"""Study 02, Experiment 2: seed set, task stream, practice cycle, replay and curriculum (plan §3 and §5).

Nothing here imports a tensor library, and nothing here can reach an evaluation split: tasks come from
`ccl.s2.load_practice` only (tests/test_e2.py enforces both). Rollouts are produced by whatever `run`
callable the caller passes in (on a GPU, `ccl.engine.Engine.run`); this module decides which tasks are
offered, which attempts are accepted, and what enters a buffer.

A practice arm is resumable. Every chunk of attempts is written once, atomically, with every attempt's
trace; a resumed cycle reads the chunks that exist instead of sampling them again, so the episodes a buffer
holds never depend on where a job was interrupted.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from . import compose as C
from . import s2, verify
from .data import episode_to_examples
from .evaluate import episode_metrics
from .runtime import Episode
from .tasks import Task

SEED_FAMILIES, SEED_INSTANCES = 50, 8  # 400 oracle episodes of one or two hops
PASSES, PASS_TASKS = 9, 1600  # one stream per replicate: nine balanced draws of 1,600 families
CYCLES, CYCLE_ACCEPT, CYCLE_TASK_CAP = 6, 480, 2400
ATTEMPTS, TEMPERATURE, TOP_P = 4, 0.8, 0.95
ARMS = ("practice", "practice+curriculum", "practice-unverified")
CHUNK = 200  # tasks attempted between two looks at the accepted count
MIN_BUCKET = 30


# ------------------------------------------------------------------ seed set and stream

def seed_families(fams: list[dict], replicate: int) -> list[dict]:
    return C.balanced_order(fams, f"e2|seed|{replicate}", (1, 2), horizon=SEED_FAMILIES)[:SEED_FAMILIES]


def seed_tasks(fams: list[dict], tools: list[str], replicate: int) -> list[Task]:
    return [C.make_task(f, i, "train", "practice", tools) for f in seed_families(fams, replicate)
            for i in range(SEED_INSTANCES)]


def _review():
    """The review's selection check (scripts/s2_review.py, A7), loaded from where it was written."""
    spec = importlib.util.spec_from_file_location("s2_review", s2.ROOT / "scripts/s2_review.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.selections


class Stream:
    """The practice tasks of one replicate, in the order every arm reads them.

    Pass `p` is a fresh balanced draw of 1,600 practice families (`e2|stream|r|p`) with a new instance of
    each, so every pass has the composition of an Experiment 1 training subset. A task that fails the
    selection check is dropped and recorded in `dropped`; `tasks[i]` is the i-th task offered."""

    def __init__(self, fams: list[dict], tools: list[str], replicate: int):
        self.fams, self.tools, self.replicate = fams, tools, replicate
        self.tasks: list[Task] = []
        self.family: list[dict] = []
        self.dropped: list[dict] = []
        self._passes, self._check = 0, _review()

    def _extend(self) -> None:
        p = self._passes
        assert p < PASSES, "the stream of this replicate is used up"
        order = C.balanced_order(self.fams, f"e2|stream|{self.replicate}|{p}", s2.E1_HOPS, PASS_TASKS)[:PASS_TASKS]
        for f in order:
            t = C.make_task(f, 1000 * (self.replicate + 1) + p, "train", "practice", self.tools)
            verdict, _ = self._check(t)
            if verdict in ("ok", "skipped"):
                self.tasks.append(t)
                self.family.append(f)
            else:
                self.dropped.append({"pass": p, "family": f["id"], "task_id": t.task_id, "why": verdict})
        self._passes += 1

    def get(self, i: int) -> Task:
        while i >= len(self.tasks):
            self._extend()
        return self.tasks[i]

    def first(self, n: int) -> list[Task]:
        return [self.get(i) for i in range(n)]


def tasks_sha256(tasks: list[Task]) -> str:
    h = hashlib.sha256()
    for t in tasks:
        h.update((json.dumps(t.to_dict(), ensure_ascii=False, sort_keys=True) + "\n").encode())
    return h.hexdigest()


# ------------------------------------------------------------------ acceptance and replay

PROTOCOL = verify.PROTOCOL  # the acceptance rule in force for Experiment 2 (2 since amendment 1)


def well_formed(task: Task, ep: Episode, protocol: int = PROTOCOL) -> bool:
    """The acceptance rule without the validator: a final answer and no parse error, unknown tool or
    invalid argument. What `practice-unverified` may keep. Under protocol 2 an episode that used the
    shortcut is kept out of this buffer too (the check reads the trace, not the validator)."""
    return ep.trace.termination == "final" and s2.rejection(task, ep) in ([], ["not_success"]) \
        and not (protocol >= 2 and verify.shortcut(ep.trace.to_dict()))


def choose(rows: list[dict], verified: bool = True, key: str = "") -> dict | None:
    """Of the attempt rows of one task (`attempt_row`), the one that may enter a buffer, or None.

    Verified: the common acceptance rule; of several passing attempts the one with the fewest calls, the
    earliest on a tie. Unverified: one well-formed attempt at random (seeded by `key`), pass or fail."""
    if verified:
        ok = [r for r in rows if not r["rejection"]]
        return min(ok, key=lambda r: (r["n_calls"], r["attempt"])) if ok else None
    ok = [r for r in rows if r["well_formed"]]
    return random.Random(f"e2|unverified|{key}|{rows[0]['task_id']}").choice(ok) if ok else None


def replay(task: Task, trace: dict, verified: bool = True, protocol: int = PROTOCOL) -> tuple[Episode | None, str]:
    """Run a recorded attempt again, from its recorded assistant turns, against a fresh simulator.

    The replayed episode must see exactly the observations that were recorded, end the same way and (for a
    verified arm) pass the acceptance rule again. This guards against a fault in the rollout path being
    mistaken for a success. Returns (episode, "ok") or (None, reason)."""
    ep = Episode(task, trace["model"])
    for t in trace["turns"]:
        if ep.done:
            return None, "recorded turns continue after the episode ended"
        ep.step(t["raw"], gen_tokens=t.get("gen_tokens", 0))
    if not ep.done or ep.trace.termination != trace["termination"]:
        return None, f"ends as {ep.trace.termination or 'unfinished'}, recorded {trace['termination']}"
    if ep.trace.messages != trace["messages"]:
        return None, "the observations differ from the recorded ones"
    if verified and s2.rejection(task, ep, protocol=protocol):
        return None, "rejected on replay: " + ",".join(s2.rejection(task, ep, protocol=protocol))
    if not verified and not well_formed(task, ep, protocol):
        return None, "not well-formed on replay"
    return ep, "ok"


def by_elimination(trace: dict) -> bool:
    """True if an attempt reached a record by trying the candidates of a list one after another.

    In the simulated worlds a record that is not on the gold path refers to records that do not exist. An
    attempt can therefore skip the selection rule a request states: it asks for the referenced record of one
    listed candidate after another until one is found. That shows in the trace as a call that ends in
    "not found" and a later successful call of the same tool, the two arguments being the same field of two
    different records of one earlier list. A mistyped identifier that is then corrected is not counted. The
    validator accepts such an episode; this check reads the trace only, never the gold graph."""
    calls = [c for c in trace["calls"] if c["outcome"] != "limit"]
    for k, c in enumerate(calls):
        if c["outcome"] == "ok" or c["result"].get("error") != "not_found":
            continue
        bad = {str(v) for v in (c["args"] or {}).values()}
        good = {str(v) for d in calls[k + 1:] if d["tool"] == c["tool"] and d["outcome"] == "ok"
                for v in (d["args"] or {}).values()} - bad
        for e in calls[:k]:
            for rows in (e["result"] or {}).values() if e["outcome"] == "ok" else ():
                if isinstance(rows, list) and all(isinstance(r, dict) for r in rows):
                    for f in {f for r in rows for f in r}:
                        col = {str(r.get(f)) for r in rows}
                        if col & bad and col & good:
                            return True
    return False


def by_empty_list(trace: dict) -> bool:
    """The same shortcut as `by_elimination`, through empty lists: a candidate off the gold path has no
    records listed under it, so listing under one candidate after another shows which one is on the path.
    A list call that returns nothing, and a later call of the same tool that returns something, the two
    arguments being the same field of two records of one earlier list. Written after the pilot's audit, which
    found this form by reading; anything it counts is labelled post-hoc."""
    calls = [c for c in trace["calls"] if c["outcome"] != "limit"]

    def rows_of(c):
        return next((v for v in (c["result"] or {}).values() if isinstance(v, list)), None) if c["outcome"] == "ok" else None
    for k, c in enumerate(calls):
        if rows_of(c) != []:
            continue
        bad = {str(v) for v in (c["args"] or {}).values()}
        good = {str(v) for d in calls[k + 1:] if d["tool"] == c["tool"] and rows_of(d)
                for v in (d["args"] or {}).values()} - bad
        for e in calls[:k]:
            rows = rows_of(e)
            if rows and all(isinstance(r, dict) for r in rows):
                for f in {f for r in rows for f in r}:
                    col = {str(r.get(f)) for r in rows}
                    if col & bad and col & good:
                        return True
    return False


def shortcut_v1(trace: dict) -> bool:
    """The two forms as the first pilot's report counted them. The acceptance rule of protocol 2 uses
    `ccl.verify.shortcut`, one definition that covers both and does not depend on tool or order."""
    return by_elimination(trace) or by_empty_list(trace)


def unobserved_identifier(task: Task, trace: dict) -> bool:
    """True if a call uses an identifier that was neither in the request nor in an earlier observation."""
    import re
    ident = re.compile(r"^[A-Z]{1,3}-\d{2,6}$")
    known = task.request
    calls = [c for c in trace["calls"] if c["outcome"] != "limit"]
    for turn in sorted({c["turn"] for c in calls}):
        now = [c for c in calls if c["turn"] == turn]
        for c in now:
            flat = [y for x in (c["args"] or {}).values() for y in (x if isinstance(x, list) else [x])]
            if any(isinstance(x, str) and ident.match(x.strip()) and x.strip() not in known for x in flat):
                return True
        known += " " + json.dumps([c["result"] for c in now])
    return False


def audit_sample(accepted: list[dict], n: int = 200, seed: str = "e2|audit") -> list[dict]:
    """The verifier audit's sample (plan §5): `n` accepted episodes, half of them from those that are not
    strictly successful (all of those if there are fewer), each with the weight that carries it back to the
    population. `accepted` holds one dict per accepted episode with at least `strict`; the order of the
    result is the order of `accepted`."""
    rng = random.Random(seed)
    loose = [i for i, x in enumerate(accepted) if not x["strict"]]
    clean = [i for i, x in enumerate(accepted) if x["strict"]]
    n_loose = min(len(loose), n // 2)
    n_clean = min(len(clean), n - n_loose)
    take = {i: len(loose) / n_loose for i in rng.sample(loose, n_loose)}
    take.update({i: len(clean) / n_clean for i in rng.sample(clean, n_clean)})
    return [{**accepted[i], "weight": take[i]} for i in sorted(take)]


def attempt_row(index: int, ep: Episode, protocol: int = PROTOCOL) -> dict:
    """One attempt as it is stored. `rejection` is under `protocol`; `rejection_v1` is what the rule of the
    proposal says, kept so that both can be counted from the same attempts."""
    m = episode_metrics(ep.task, ep)
    return {"index": index, "task_id": ep.task.task_id, "attempt": ep.attempt, "success": m["success"],
            "protocol": protocol, "rejection": s2.rejection(ep.task, ep, m, protocol),
            "rejection_v1": s2.rejection(ep.task, ep, m), "well_formed": well_formed(ep.task, ep, protocol),
            "strict": bool(m["success"]) and s2.dependency_safe(ep.task, ep) and m["unnecessary_calls"] == 0,
            "n_calls": m["n_calls"], "turns": m["turns"], "gen_tokens": m["gen_tokens"],
            "trace": ep.trace.to_dict()}


# ------------------------------------------------------------------ curriculum

def shape_class(f: dict) -> str:
    s = f["shape"]
    return next((k for k, w in (("twin", "twin"), ("fanout", "fanout"), ("action", "act"), ("list", "list"))
                 if w in s), "plain")


def mod_class(f: dict) -> str:
    return {"none": "none", "retry": "fault", "fallback": "fault"}.get(f["mod"], "stop")


def buckets(fams: list[dict]) -> dict[str, str]:
    """family id -> bucket. Families are grouped by hops, shape class and modifier class; a group with
    fewer than 30 practice families is merged into the group of the same shape and modifier class that is
    nearest in hops (the lower on a tie), smallest first, until none can be merged further."""
    group = {f["id"]: (f["hops"], shape_class(f), mod_class(f)) for f in fams}
    size = Counter(group.values())
    merged: dict[tuple, tuple] = {}
    while True:
        small = sorted((n, g) for g, n in size.items() if n < MIN_BUCKET)
        target = None
        for n, g in small:
            near = sorted((abs(o[0] - g[0]), o[0], o) for o in size if o != g and o[1:] == g[1:])
            if near:
                target = (g, near[0][2])
                break
        if target is None:
            break
        g, to = target
        size[to] += size.pop(g)
        merged[g] = to
    def final(g):
        while g in merged:
            g = merged[g]
        return g
    members = defaultdict(set)
    for g in set(group.values()):
        members[final(g)].add(g[0])
    name = {g: f"h{'-'.join(map(str, sorted(members[g])))}|{g[1]}|{g[2]}" for g in members}
    return {fid: name[final(g)] for fid, g in group.items()}


def weights(history: dict[str, tuple[int, int]], names) -> dict[str, float]:
    """`p(1 - p) + 0.05` per bucket, p being accepted/attempted tasks over the last two cycles with one
    pseudo-task at 0.5, so that an untried bucket starts at 0.5."""
    out = {}
    for b in names:
        acc, att = history.get(b, (0, 0))
        p = (acc + 0.5) / (att + 1)
        out[b] = p * (1 - p) + 0.05
    return out


# ------------------------------------------------------------------ one practice arm

class Practice:
    """One practice arm of one replicate. `run(tasks, seed)` must return `ATTEMPTS` episodes per task,
    adjacent and in order, each with `.attempt`."""

    def __init__(self, out: Path, replicate: int, arm: str, fams: list[dict], tools: list[str],
                 protocol: int | None = None):
        assert arm in ARMS
        self.out, self.replicate, self.arm = Path(out), replicate, arm
        self.stream = Stream(fams, tools, replicate)
        self.bucket = buckets(fams)
        self.state_path = self.out / "state.json"
        if self.state_path.exists():  # an arm keeps the protocol it was started under (1 before amendment 1)
            self.state = json.loads(self.state_path.read_text())
            stored = self.state.get("protocol", 1)
            assert protocol in (None, stored), f"{self.out} was run under protocol {stored}, not {protocol}"
            self.protocol = stored
        else:
            self.protocol = PROTOCOL if protocol is None else protocol
            self.state = {"protocol": self.protocol, "cycles": []}

    def _save(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.rename(self.state_path)

    def _bucket_of(self, i: int) -> str:
        self.stream.get(i)
        return self.bucket[self.stream.family[i]["id"]]

    def _history(self, cycle: int) -> dict[str, tuple[int, int]]:
        """Accepted and attempted tasks per bucket over the two cycles before `cycle`."""
        h: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for c in self.state["cycles"]:
            if cycle - 2 <= c["cycle"] < cycle:
                acc = set(c["accepted"])
                for i in c["offered"]:
                    h[self._bucket_of(i)][0] += i in acc
                    h[self._bucket_of(i)][1] += 1
        return {b: tuple(v) for b, v in h.items()}

    def _take(self, n: int, used: set[int], w: dict[str, float] | None, rng: random.Random) -> list[int]:
        """The next `n` stream tasks to offer. Without a curriculum: stream order. With one: a fifth in
        stream order regardless, the rest by drawing a bucket in proportion to its weight and taking that
        bucket's earliest unused task."""
        limit = PASSES * PASS_TASKS
        free = (i for i in range(limit) if i not in used)
        if w is None:
            return [next(free) for _ in range(n)]
        take: list[int] = []
        horizon = min(limit, max(used, default=-1) + 1 + 4 * PASS_TASKS)  # look ahead at most four passes
        queue = defaultdict(list)
        for i in range(horizon):
            if i not in used:
                queue[self._bucket_of(i)].append(i)
        names = sorted(queue)
        for k in range(n):
            if k % 5 == 0 or not names:
                i = next(j for j in range(limit) if j not in used and j not in take)
                b = self._bucket_of(i)
                if i in queue[b]:
                    queue[b].remove(i)
            else:
                b = rng.choices(names, weights=[w.get(x, 0.3) for x in names])[0]
                i = queue[b].pop(0)
            take.append(i)
            names = [x for x in names if queue[x]]
        return take

    def cycle(self, k: int, run, max_tasks: int | None = None) -> dict:
        """Cycle `k`: offer tasks until `CYCLE_ACCEPT` are accepted or the task cap is reached. Returns the
        cycle record and stores it in the arm's state. With `max_tasks` it is a probe: it stops there and
        stores nothing, so a later job continues the cycle from the probe's attempt files."""
        complete, max_tasks = max_tasks is None, CYCLE_TASK_CAP if max_tasks is None else max_tasks
        assert k == len(self.state["cycles"]) + 1, f"cycle {k} does not follow the {len(self.state['cycles'])} stored"
        verified = self.arm != "practice-unverified"
        used = {i for c in self.state["cycles"] for i in c["offered"]}
        w = weights(self._history(k), sorted(set(self.bucket.values()))) if self.arm.endswith("curriculum") else None
        rng = random.Random(f"e2|curriculum|{self.replicate}|{k}")
        seed = f"e2|{self.replicate}|{self.arm}|{k}"
        offered: list[int] = []
        accepted: list[int] = []
        rows: list[dict] = []  # SFT rows of the accepted episodes, in the order accepted
        counts = Counter()
        chunk = 0
        while len(accepted) < CYCLE_ACCEPT and len(offered) < max_tasks:
            rate = max(len(accepted) / len(offered), 0.2) if offered else 0.2
            n = min(CHUNK, max_tasks - len(offered), max(16, math.ceil((CYCLE_ACCEPT - len(accepted)) / rate)))
            idx = self._take(n, used | set(offered), w, rng)
            path = self.out / f"attempts/c{k}_{chunk:03d}.jsonl.gz"
            if path.exists():
                recs = s2.read_gz(path)
                assert [r["index"] for r in recs[::ATTEMPTS]] == idx, f"{path} does not hold the tasks now drawn"
                assert all(r.get("protocol", 1) == self.protocol for r in recs), \
                    f"{path} was written under another protocol; its stored verdicts are not this arm's"
            else:
                eps = run([self.stream.get(i) for i in idx], seed)
                assert len(eps) == ATTEMPTS * len(idx)
                recs = [attempt_row(idx[j // ATTEMPTS], e, self.protocol) for j, e in enumerate(eps)]
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                with gzip.open(tmp, "wt") as f:
                    for r in recs:
                        f.write(json.dumps(r) + "\n")
                tmp.rename(path)
            for j, i in enumerate(idx):
                task = self.stream.get(i)
                mine = recs[j * ATTEMPTS:(j + 1) * ATTEMPTS]
                assert all(r["task_id"] == task.task_id for r in mine)
                offered.append(i)
                counts["attempts"] += len(mine)
                counts["any_success"] += any(r["success"] for r in mine)
                # what the rule of the proposal would have admitted, and why protocol 2 turned attempts away
                counts["tasks_admissible_under_protocol_1"] += any(not r.get("rejection_v1", r["rejection"]) for r in mine)
                counts["attempts_rejected_for_shortcut"] += sum("shortcut" in r["rejection"] for r in mine)
                counts["attempts_rejected_for_answer_parts"] += sum("answer_parts" in r["rejection"] for r in mine)
                pick = choose(mine, verified, seed)
                if pick is None:
                    continue
                ep, why = replay(task, pick["trace"], verified, self.protocol)
                if ep is None:
                    counts["replay_failed"] += 1
                    continue
                accepted.append(i)
                counts["accepted_strict"] += pick["strict"]
                counts["accepted_success"] += bool(pick["success"])
                for r in episode_to_examples(ep, self.arm):
                    r["meta"].update(cycle=k, index=i, sample=pick["attempt"])
                    rows.append(r)
                if len(accepted) == CYCLE_ACCEPT:
                    break
            chunk += 1
        rec = {"cycle": k, "protocol": self.protocol, "offered": offered, "accepted": accepted, "complete": complete,
               "yield": len(accepted) / max(1, len(offered)), **counts, "chunks": chunk,
               "rows_sha256": s2.rows_sha256(rows)}
        if rec["complete"]:
            (self.out / "buffer").mkdir(parents=True, exist_ok=True)
            with gzip.open(self.out / f"buffer/c{k}.jsonl.gz", "wt") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            self.state["cycles"].append(rec)
            self._save()
        return {**rec, "rows": rows}


# ------------------------------------------------------------------ what a set of tasks covers

def diversity(fams: list[dict], tasks: list[Task] | None = None, reference: list[dict] | None = None) -> dict:
    """The structural-diversity table of plan §3.3 for a list of families (one entry per task, so repeats
    count). With `reference` (for example the seed families): how much of the list lies outside it."""
    n = len(fams)
    if not n:
        return {"tasks": 0}
    d = {"tasks": n, "families": len({f["id"] for f in fams}),
         "base_graphs": len({g for f in fams for g in f["graphs"]}),
         "tools": len({t for f in fams for t in f["tools"]}), "edges": len({e for f in fams for e in f["edges"]}),
         "hops": {str(h): round(c / n, 3) for h, c in sorted(Counter(f["hops"] for f in fams).items())},
         "modifier": {m: round(c / n, 3) for m, c in sorted(Counter(mod_class(f) for f in fams).items())},
         "shape": {m: round(c / n, 3) for m, c in sorted(Counter(shape_class(f) for f in fams).items())},
         "mean_gold_calls": round(sum(f["calls"] for f in fams) / n, 3)}
    if tasks is not None:
        sel = [sum(t.startswith("rule=") for t in x.tags) for x in tasks]
        d["mean_selections"] = round(sum(sel) / n, 3)
        d["selections"] = {str(k): round(c / n, 3) for k, c in sorted(Counter(sel).items())}
    if reference is not None:
        ids = {f["id"] for f in reference}
        tools = {t for f in reference for t in f["tools"]}
        edges = {e for f in reference for e in f["edges"]}
        d["outside_reference"] = {
            "new_family": round(sum(f["id"] not in ids for f in fams) / n, 3),
            "new_tool": round(sum(bool(set(f["tools"]) - tools) for f in fams) / n, 3),
            "new_edge": round(sum(bool(set(f["edges"]) - edges) for f in fams) / n, 3),
            "deeper": round(sum(f["hops"] > max(r["hops"] for r in reference) for f in fams) / n, 3)}
    return d
