"""An adversarial evaluation for Study 02, Experiment 2: `dev` tasks on worlds whose candidate records are complete.

The generated worlds build only the records on a task's gold path in full. A record that is merely a
candidate of a list refers to records that do not exist and has nothing listed under it, so an agent can find
the right candidate by asking for each one's continuation instead of applying the selection rule the request
states (`ccl.verify.shortcut`). `twin` rebuilds a task on a world where that no longer works: for the tools of
the task's gold path, every reference leads to a record and every record has records listed under it, to the
depth of the task. Whatever the gold path asks of the chosen record can be asked of every other candidate with
the same kind of result. For the other tools of the catalogue nothing changes, and there the chosen record
and the other candidates were already alike (neither has anything behind it).

A twin has the same request, catalogue, gold graph, faults and answer as its original, and every original
record unchanged, so an agent that applies the selection rule sees exactly the same episode on both. Any
difference in success between a task and its twin comes from what the agent does off the gold path.

`adversarial_dev` is the set used in Experiment 2: the twin of every `dev` task in which some call depends on
a list selection (350 of 600). It is an evaluation set only, built from `dev` and from nothing else, and
no training code reads it. Generation is deterministic; the set's SHA-256 is recorded in
`configs/study02/e2/protocol2.json` and checked by the tests.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import replace
from datetime import date

from . import compose as C
from . import s2
from .tasks import Task, task_hash
from .tools import SIMULATORS, ToolError
from .world2 import ENTITIES, TOOLS2

SALT = "ccl-study02-e2-adversarial-1"
CHILDREN = 2  # records created under a record that has none
VERSION = 2  # 1: the set of amendment 1; 2 (amendment 2): also repaired along every walk (`complete_world`)
MAX_RECORDS = 10_000  # a guard against a completion that runs away: an error, never a truncation (largest: 3,708)


def depends_on_selection(task: Task) -> bool:
    """Does some gold call take its argument from a record chosen out of a list? Only then can a candidate's
    continuation be probed."""
    lists = {g.id for g in task.gold if TOOLS2[g.tool].kind == "list"}
    return any(set(g.deps) & lists for g in task.gold)


def complete_world(task: Task, rounds: int | None = None, repair: bool = False) -> tuple[dict, dict]:
    """A copy of the task's world, completed for the tools of its gold path, and what was added.

    One round: (a) for every record and every reference field whose target type some gold-path tool takes as
    an identifier, create the referenced record if it does not exist; (b) for every list tool of the gold path
    and every record of its parent type with nothing listed under it, create two records under it. New
    records get fresh identifiers, unused names and random attribute values, and their own references point
    at identifiers that the next round resolves. `rounds` defaults to the task's depth.

    A round fills lists only under records that existed when it began, so some records created late have
    nothing behind them, and with `depth` rounds alone (version 1 of the set) a walk from another candidate
    along the rest of the gold path's calls met a missing record or an empty list in 12 of the 350 `dev`
    twins (review of PR #3). With `repair` (version 2), every such walk is then followed (`leaks`) and what it
    misses is created, until no walk misses anything. A further full round would do the same for `dev` but
    uses up the identifier space on six-hop tasks.

    Records under a gold-path parent are never added to, so every list on the gold path returns what it
    returned before and every selection is unchanged."""
    world = copy.deepcopy(task.world)
    rng = random.Random(f"{SALT}|{task.task_id}")
    b = C.Builder(rng, "eval")
    b.w, b.today = world, date.fromisoformat(task.today)
    tables = [e for e in ENTITIES if world.get(e)]
    b.used = {rid for e in tables for rid in world[e]} | {r["name"] for e in tables for r in world[e].values() if "name" in r} \
        | {r[ref.field] for e in tables for r in world[e].values() for ref in ENTITIES[e].refs}
    keys = [TOOLS2[k] for k in sorted({k for g in task.gold for k in (g.tool, g.fallback) if k})]
    # entity types whose identifier some gold-path tool accepts
    addressable = {t.parent if t.kind == "list" else t.entity for t in keys} | \
        {kind[4:] for t in keys if t.action for _, kind in t.action.params if kind.startswith("ref:")}
    lists = sorted((t for t in keys if t.kind == "list"), key=lambda t: t.key)
    added = {"records": 0, "references_resolved": 0, "lists_filled": 0}
    serial = [0]

    def make(ent, rid=None, **fixed) -> dict:
        rec = {ent.id_field: rid or b.uid(ent)}
        if ent.named:
            try:
                rec["name"] = b._name(ent.named, False)
            except RuntimeError:  # the name pool is used up in a large world
                serial[0] += 1
                rec["name"] = f"{rng.choice(C.NAMES[ent.named][:18]) if ent.named != 'person' else 'Alex Morgan'} {1000 + serial[0]}"
                b.used.add(rec["name"])
        for a in ent.attrs:
            rec[a.name] = b._value(a, rec)
        for r in ent.refs:
            rec[r.field] = b.uid(ENTITIES[r.target])
        rec.update(fixed)
        world[ent.name][rec[ent.id_field]] = rec
        added["records"] += 1
        return rec

    for _ in range(task.depth if rounds is None else rounds):
        before = {name: list(world[name].values()) for name in sorted(ENTITIES)}  # a round works on what exists at its start
        for name, recs in before.items():  # (a) references
            for rec in recs:
                for r in ENTITIES[name].refs:
                    if r.target in addressable and rec[r.field] not in world[r.target]:
                        make(ENTITIES[r.target], rid=rec[r.field])
                        added["references_resolved"] += 1
        for t in lists:  # (b) lists
            have = {r[t.ref_field] for r in world[t.entity].values()}
            for parent in before[t.parent]:
                pid = parent[ENTITIES[t.parent].id_field]
                if pid not in have:
                    for _ in range(CHILDREN):
                        make(ENTITIES[t.entity], **{t.ref_field: pid})
                    added["lists_filled"] += 1
        assert added["records"] <= MAX_RECORDS, f"{task.task_id}: {added['records']} records added"
    added["repaired"] = 0
    for _ in range(50 if repair else 0):
        todo = leaks(task, world)
        if not todo:
            break
        for key, args in todo:
            t, value = TOOLS2[key], next(iter(args.values()))
            if t.kind == "list" and value in world[t.parent]:
                if not any(r[t.ref_field] == value for r in world[t.entity].values()):
                    for _ in range(CHILDREN):
                        make(ENTITIES[t.entity], **{t.ref_field: value})
                    added["repaired"] += 1
            elif value not in world[t.parent if t.kind == "list" else t.entity]:
                make(ENTITIES[t.parent if t.kind == "list" else t.entity], rid=value)
                added["repaired"] += 1
        assert added["records"] <= MAX_RECORDS, f"{task.task_id}: {added['records']} records added"
    else:
        assert not repair, f"{task.task_id}: walks still meet missing records"
    return world, added


def leaks(task: Task, world: dict) -> list[tuple[str, dict]]:
    """The calls that come back "not found" or empty when the gold path's calls are followed from another
    candidate: at every list of the gold path whose pick a later call uses, every other row is taken in turn,
    and from then on every row of every later list. Only get and list calls with one argument are reported
    (what `complete_world` can create); the gold path itself is not walked off before its first selection."""
    w = {**copy.deepcopy(world), "_log": []}
    gold, out = {}, []
    for g in task.gold:  # what the gold path sees: per call its record, or its rows and the row it picks
        if TOOLS2[g.tool].kind == "action":
            break
        try:
            res = SIMULATORS[g.tool](w, dict(g.args))
        except ToolError:
            break
        rows = next((v for v in res.values() if isinstance(v, list)), None)
        if rows is None:
            gold[g.id] = (res, None)
        else:
            later = {str(v) for h in task.gold if g.id in h.deps for v in h.args.values()}
            ref = TOOLS2[g.tool].ref_field
            cand = [r for r in rows if {str(v) for k, v in r.items() if k != ref} & later]
            gold[g.id] = (cand[0] if len(cand) == 1 else None, rows)
    seen = set()

    def walk(i: int, mine: dict, off: bool) -> None:
        if i == len(task.gold) or task.gold[i].id not in gold:
            return
        g = task.gold[i]
        args = dict(g.args)
        for d in g.deps:  # an argument that came from a field of a dependency's record comes from this walk's record
            src = gold.get(d, (None,))[0]
            if d in mine and src is not None:
                for k, v in g.args.items():
                    f = next((f for f, x in src.items() if x == v), None)
                    if f is not None:
                        args[k] = mine[d].get(f, v)
        try:
            res = SIMULATORS[g.tool](w, args)
        except ToolError as e:
            if off and e.code == "not_found" and len(args) == 1 and TOOLS2[g.tool].kind in ("get", "list"):
                seen.add((g.tool, json.dumps(args, sort_keys=True)))
            return
        rows = next((v for v in res.values() if isinstance(v, list)), None)
        if rows is None:
            walk(i + 1, {**mine, g.id: res}, off)
        elif off:
            if not rows and len(args) == 1 and TOOLS2[g.tool].kind == "list":
                seen.add((g.tool, json.dumps(args, sort_keys=True)))
            for r in rows:
                walk(i + 1, {**mine, g.id: r}, True)
        else:
            pick = gold[g.id][0]
            walk(i + 1, {**mine, g.id: pick or (rows[0] if rows else {})}, False)
            for r in rows if pick is not None else []:
                if r != pick:
                    walk(i + 1, {**mine, g.id: r}, True)

    walk(0, {}, False)
    for key, a in sorted(seen):
        out.append((key, json.loads(a)))
    return out


def twin(task: Task, version: int = VERSION) -> Task:
    """The task on its completed world, in split `<split>_adv` (`dev_adv` for a `dev` task).
    `twin_of=<task id>` in its tags names the original. Version 1 is the set of amendment 1; version 2 also
    repairs what a walk from another candidate still misses (`complete_world`)."""
    world, _ = complete_world(task, repair=version >= 2)
    split = f"{task.split}_adv"
    t = replace(task, world=world, split=split, tags=list(task.tags) + [f"twin_of={task.task_id}"], task_id="")
    t.task_id = f"{split}-{task.template}-{task_hash(t)[:6]}-{task.task_id.split('-')[-1]}"
    return t


def twins(tasks: list[Task], version: int = VERSION) -> list[Task]:
    """The twin of every task in which some call depends on a list selection. The rule is the same for every
    evaluation split; which splits it is applied to is decided by the code that scores them."""
    return [twin(t, version) for t in tasks if depends_on_selection(t) and t.expected_status in ("done", "blocked")]


def adversarial_dev(version: int = VERSION) -> list[Task]:
    return twins(s2.load_split("dev"), version)


def selection_picks(task: Task, trace: dict) -> list[tuple[bool, float]]:
    """For every list of the gold path whose pick a later gold call uses, and which the episode obtained: did
    the first later call that uses a row of it use the gold row? Returns (gold row taken, 1 / number of rows)
    per such selection; the second is what picking at random would give. An evaluation measure only (it reads
    the gold graph): it tells an agent that applies the selection rule from one that gets further down the
    chain and picks by luck, which neither the validator nor a complete world can do (review of PR #3)."""
    w = {**copy.deepcopy(task.world), "_log": []}
    calls = [c for c in trace["calls"] if c["outcome"] != "limit"]
    out = []
    for g in task.gold:
        users = [h for h in task.gold if g.id in h.deps]
        if TOOLS2[g.tool].kind != "list" or not users:
            continue
        try:
            rows = next(v for v in SIMULATORS[g.tool](w, dict(g.args)).values() if isinstance(v, list))
        except (ToolError, StopIteration):
            continue
        used = {str(v) for h in users for v in h.args.values()}
        ref = TOOLS2[g.tool].ref_field
        fields = {f for r in rows for f, v in r.items() if f != ref and str(v) in used}
        got = next((i for i, c in enumerate(calls) if c["outcome"] == "ok" and c["tool"] == g.tool and c["args"] == g.args), None)
        if got is None or len(rows) < 2 or len(fields) != 1:
            continue
        f = fields.pop()
        column = {str(r[f]) for r in rows}
        first = next((v for c in calls[got + 1:] for v in map(str, (c["args"] or {}).values()) if v in column), None)
        if first is not None:
            out.append((first in used, 1 / len(rows)))
    return out
