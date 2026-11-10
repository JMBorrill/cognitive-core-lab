"""Study 02, Experiment 2, Stage 1: what was measured, against the gates of the plan.

    python scripts/s2_e2_stage1.py     # results/study02/e2/stage1.{json,md}

Reads the result files that scripts/s2_launch.py pulled into results/study02/runs/e2 and the Stage 1 ledger.
No GPU, no sealed split. Sections follow plan §7, Stage 1: (a) engine, (c) seed adapter and yield probe,
(d) teacher qualification; item (b) was not run (see the plan's clarifications).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccl import loop as L  # noqa: E402
from ccl import s2  # noqa: E402
from ccl.oracle import OraclePolicy  # noqa: E402
from ccl.runtime import run_batch  # noqa: E402

RUNS = ROOT / "results/study02/runs/e2"
OUT = ROOT / "results/study02/e2"
E1_OWN_SECONDS = 1.62  # Experiment 1, 1.5B student, own-family tasks (experiments/study02_e2_signal/budget.md)
GATE = {"seconds_per_episode": 0.36, "changed_share": 0.01, "success_points": 1.0, "yield": 0.20,
        "seed_below_points": 15.0}


def result(name: str) -> dict | None:
    p = RUNS / name / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def pc(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def table(head: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + \
        ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


# ------------------------------------------------------------------ (a) engine

def engine(tag: str = "") -> dict:
    """The calibration of the engine as used (`tag` ""), or of the revision R6 that failed (`tag` "2")."""
    runs = {}
    for name in ("e1_s15_D1600_r0", "e1_s15_D25_r0"):
        r = result(f"calib{tag}_{name}")
        if r:
            n = r["episodes_compared"]
            runs[name] = {**{k: r[k] for k in ("episodes", "seconds_per_episode", "success", "success_experiment_1",
                                              "changed_outcome", "same_text_as_full_re_encoding")},
                          "episodes_compared": n, "changed_share": r["changed_outcome"] / n,
                          "success_difference_points": 100 * (r["success"] - r["success_experiment_1"]),
                          "speed_up": E1_OWN_SECONDS / r["seconds_per_episode"],
                          "tokens_read_per_episode": r["engine"]["fed_tokens"] / r["engine"]["episodes"],
                          "tokens_written_per_episode": r["engine"]["gen_tokens"] / r["engine"]["episodes"],
                          "out_of_memory_splits": r["engine"]["oom_splits"], "gpu": r.get("gpu")}
    if not runs:
        return {}
    worst = lambda k, f=max: f(v[k] for v in runs.values())  # noqa: E731
    passed = {"speed": worst("seconds_per_episode") <= GATE["seconds_per_episode"],
              "flips": worst("changed_share") <= GATE["changed_share"],
              "success": max(abs(v["success_difference_points"]) for v in runs.values()) <= GATE["success_points"]}
    return {"runs": runs, "criteria": passed, "passed": all(passed.values()) and len(runs) == 2,
            "outcomes_equivalent": passed["flips"] and passed["success"],
            "seconds_per_episode": sum(v["seconds_per_episode"] for v in runs.values()) / len(runs),
            "speed_up": E1_OWN_SECONDS / (sum(v["seconds_per_episode"] for v in runs.values()) / len(runs))}


# ------------------------------------------------------------------ (c) seed adapter and yield probe

def seed() -> dict:
    r = result("r0_seed")
    if not r:
        return {}
    train = RUNS / "r0_seed/train_result.json"  # the seed job's record; the dev job wrote result.json after it
    if train.exists():
        r = {**json.loads(train.read_text()), **r}
    rows = [json.loads(l) for l in open(RUNS / "r0_seed/metrics.jsonl")] if (RUNS / "r0_seed/metrics.jsonl").exists() else []
    by = defaultdict(list)
    for x in rows:
        by[x["hops"]].append(x["success"])
    return {**{k: r.get(k) for k in ("seed_set_sha256", "steps", "seconds_per_step", "train_seconds", "peak_mem_gb",
                                     "adapter_sha256", "dev_success", "dev_tasks", "dev_seconds", "dev_engine")},
            "dev_by_hops": {str(h): sum(v) / len(v) for h, v in sorted(by.items())},
            "dev_strict": sum(x["strict_success"] for x in rows) / len(rows) if rows else None,
            "dev_seconds_per_episode": r["dev_seconds"] / r["dev_tasks"] if r.get("dev_tasks") else None}


def headroom(seed_dev: float) -> dict:
    """Criterion (c), second half: seed `dev` success at least 15 points below the 1,600-family adapters'.
    Those adapters have no `dev` rows (item (b) was not run), so their `dev` success is bracketed from
    Experiment 1's committed rows: for the 0.5B student `dev` lies between `test_comp` and `test_id` at both
    levels of D, and the 1.5B student has both of those."""
    import gzip
    e1 = ROOT / "results/study02/runs/e1"

    def mean(tag, d, reps, split):
        v = []
        for r in reps:
            with gzip.open(e1 / f"e1_{tag}_D{d}_r{r}/metrics.jsonl.gz", "rt") as f:
                x = [json.loads(l)["success"] for l in f if f'"split": "{split}"' in l]
            v.append(sum(x) / len(x))
        return sum(v) / len(v)
    small = {d: {k: mean("s05", d, range(5), f"v2/{k}") for k in ("dev", "test_comp", "test_id")} for d in (25, 1600)}
    lo, hi = mean("s15", 1600, range(3), "v2/test_comp"), mean("s15", 1600, range(3), "v2/test_id")
    return {"seed_dev": seed_dev, "comparator_measured": False, "bracket_for_1600_family_dev": [lo, hi],
            "bracket_holds_for_0.5B": all(v["test_comp"] < v["dev"] < v["test_id"] for v in small.values()),
            "0.5B": small, "margin_points_at_bracket": [100 * (lo - seed_dev), 100 * (hi - seed_dev)],
            "met_under_the_estimate": 100 * (lo - seed_dev) >= GATE["seed_below_points"]}


def calls_of(trace: dict) -> list[tuple]:
    return sorted((c["tool_key"] or c["tool"], json.dumps(c["canonical_args"], sort_keys=True))
                  for c in trace["calls"] if c["outcome"] != "limit")


def probe() -> dict:
    r = result("r0_practice")
    if not r:
        return {}
    fams, tools = s2.load_practice()
    st = L.Stream(fams, tools, 0)
    recs = [x for f in sorted((RUNS / "r0_practice/attempts").glob("c1_*.jsonl.gz")) for x in s2.read_gz(f)]
    offered = r["cycle"]["offered"]
    accepted = set(r["cycle"]["accepted"])
    recs = [x for x in recs if x["index"] in set(offered)]
    by_task = defaultdict(list)
    for x in recs:
        by_task[x["index"]].append(x)
    task = {i: st.get(i) for i in offered}
    fam = {i: st.family[i] for i in offered}
    nsel = {i: sum(t.startswith("rule=") for t in task[i].tags) for i in offered}
    bucket = L.buckets(fams)
    seed_f = L.seed_families(fams, 0)
    seed_ids, seed_edges = {f["id"] for f in seed_f}, {e for f in seed_f for e in f["edges"]}
    seed_tools = {t for f in seed_f for t in f["tools"]}

    def novelty(i):
        f = fam[i]
        if f["id"] in seed_ids:
            return "0 a seed family"
        if f["hops"] > 2:
            return "3 deeper than any seed family"
        if set(f["tools"]) - seed_tools or set(f["edges"]) - seed_edges:
            return "2 seed depth, a tool or edge the seed lacks"
        return "1 seed depth, tools and edges all in the seed"

    def split(key) -> dict:
        g = defaultdict(lambda: [0, 0, 0, 0])
        for i in offered:
            k = g[str(key(i))]
            k[0] += 1
            k[1] += i in accepted
            k[2] += sum(x["success"] for x in by_task[i])
            k[3] += len(by_task[i])
        return {k: {"offered": v[0], "accepted": v[1], "yield": v[1] / v[0], "attempt_success": v[2] / v[3]}
                for k, v in sorted(g.items())}
    # what the accepted episodes look like beside the oracle's solution of the same task
    oracle = {e.task.task_id: e for e in run_batch([task[i] for i in offered if i in accepted], OraclePolicy(), batch_size=512)}
    same_calls = fewer_turns = more_turns = strict = extra = 0
    picks = {}
    for i in offered:
        if i in accepted:
            p = L.choose(by_task[i])
            picks[i] = p
            o = oracle[task[i].task_id]
            oc = sorted((c.tool_key, json.dumps(c.canonical_args, sort_keys=True)) for c in o.trace.calls)
            same_calls += calls_of(p["trace"]) == oc
            extra += len(calls_of(p["trace"])) > len(oc)
            fewer_turns += p["turns"] < len(o.trace.turns)
            more_turns += p["turns"] > len(o.trace.turns)
            strict += p["strict"]
    n_acc = len(accepted)
    admissible = [x for x in recs if not x["rejection"]]
    elim = {"accepted_episodes": sorted(i for i in picks if L.by_elimination(picks[i]["trace"])),
            "admissible_attempts": sum(L.by_elimination(x["trace"]) for x in admissible),
            "of_admissible_attempts": len(admissible)}
    distinct_ok = [len({json.dumps([t["raw"] for t in x["trace"]["turns"]]) for x in by_task[i] if not x["rejection"]})
                   for i in offered if i in accepted]
    stops = Counter(x["trace"]["termination"] for x in recs)
    why = Counter(k for x in recs for k in (x["rejection"] or ["accepted"]))
    acc_idx = [i for i in offered if i in accepted]
    div = {"offered": L.diversity([fam[i] for i in offered], [task[i] for i in offered], seed_f),
           "accepted by practice": L.diversity([fam[i] for i in acc_idx], [task[i] for i in acc_idx], seed_f),
           "oracle arm, same number of tasks": L.diversity([fam[i] for i in offered[:n_acc]],
                                                           [task[i] for i in offered[:n_acc]], seed_f),
           "seed set": L.diversity([f for f in seed_f for _ in range(L.SEED_INSTANCES)])}
    return {"tasks": len(offered), "accepted": n_acc, "yield": n_acc / len(offered),
            "attempts": len(recs), "attempt_success": sum(x["success"] for x in recs) / len(recs),
            "attempt_accepted": sum(not x["rejection"] for x in recs) / len(recs),
            "tasks_with_a_success": sum(any(x["success"] for x in by_task[i]) for i in offered) / len(offered),
            "termination": dict(stops), "rejection_reasons": dict(why),
            "seconds_per_attempt": r["seconds_per_attempt"], "rollout_seconds": r["rollout_seconds"],
            "tokens_read_per_attempt": r["engine"]["fed_tokens"] / r["engine"]["episodes"],
            "tokens_written_per_attempt": r["engine"]["gen_tokens"] / r["engine"]["episodes"],
            "stream_dropped": len(r["stream_dropped"]),
            "by_hops": split(lambda i: fam[i]["hops"]), "by_selections": split(lambda i: nsel[i]),
            "by_novelty": split(novelty), "by_bucket": split(lambda i: bucket[fam[i]["id"]]),
            "diversity": div,
            "accepted_episodes": {"n": n_acc, "strict": strict, "same_calls_as_oracle": same_calls,
                                  "more_calls_than_oracle": extra, "fewer_turns_than_oracle": fewer_turns,
                                  "more_turns_than_oracle": more_turns,
                                  "tasks_with_two_or_more_distinct_passing_attempts": sum(d > 1 for d in distinct_ok),
                                  "found_by_elimination": elim},
            "tasks_per_cycle_at_this_yield": L.CYCLE_ACCEPT / max(n_acc / len(offered), 1e-9),
            "replay": json.loads((RUNS / "r0_practice/replay.json").read_text())
            if (RUNS / "r0_practice/replay.json").exists() else None}


# ------------------------------------------------------------------ training benchmark, teacher, books

def bench() -> dict:
    r = result("r0_bench")
    return {} if not r else {"episodes": r["episodes"], "settings": r["settings"]}


def teachers(prices: dict) -> dict:
    out = {}
    for d in sorted(RUNS.glob("teacher_*")):
        r = result(d.name)
        if not r:
            continue
        attempts = sum(p["attempts"] for p in r["passes"].values())
        flavor = r.get("accelerator") or ""
        usd_h = prices.get(d.name)
        rows = s2.read_gz(d / "attempts.jsonl.gz")
        ok = [x for x in rows if not x["rejection"]]
        hops = {x["index"]: x["trace"] for x in ok}
        depth = {}
        for x in rows:
            depth[x["index"]] = None
        fams, tools = s2.load_practice()
        st = L.Stream(fams, tools, 0)
        for i in depth:
            depth[i] = st.get(i).depth

        def rate(keep) -> dict:
            good = {x["index"] for x in ok if keep(x)}
            by = defaultdict(list)
            for i, h in depth.items():
                by[h].append(i in good)
            return {"overall": len(good) / len(depth), **{str(h): sum(v) / len(v) for h, v in sorted(by.items())}}
        t = {"accepted_strict": sum(x["strict"] for x in ok), "accepted_by_elimination": sum(L.by_elimination(x["trace"]) for x in ok),
             "accepted_n": len(ok), "acceptance_without_elimination": rate(lambda x: not L.by_elimination(x["trace"])),
             "acceptance_strict_only": rate(lambda x: x["strict"])}
        t = {**t, "model": r["model"], "complete": r["complete"], "acceptance": r["acceptance"],
             "acceptance_by_hops": r["acceptance_by_hops"], "qualifies": r["qualifies"],
             "greedy_acceptance": r["passes"]["greedy"]["accepted"] / r["tasks"],
             "attempts": attempts, "rollout_seconds": r["rollout_seconds"], "load_seconds": r["load_seconds"],
             "seconds_per_attempt": r["rollout_seconds"] / attempts, "flavor": flavor, "usd_per_hour": usd_h,
             "replay": json.loads((d / "replay.json").read_text()) if (d / "replay.json").exists() else None}
        if usd_h and r["accepted"]:
            t["usd_per_accepted_demonstration"] = r["rollout_seconds"] / 3600 * usd_h / r["accepted"]
            t["usd_for_14400_demonstrations"] = 14400 * t["usd_per_accepted_demonstration"]
        out[d.name] = t
    return out


def books() -> dict:
    spec = __import__("importlib.util").util.spec_from_file_location("s2_launch", ROOT / "scripts/s2_launch.py")
    launch = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    budget, path = launch.load_budget("e2_stage1")
    ledger = launch.read_ledger(path)
    b = launch.books(ledger)
    jobs = {}
    for e in ledger:
        j = jobs.setdefault(e.get("key"), {})
        if e["event"] == "reserve":
            j.update(flavor=e["flavor"], timeout_min=e["timeout_min"], reserved_usd=e["reserved_usd"],
                     price_per_min=e["price_per_min"], commit=e["commit"][:10])
        elif e["event"] == "settled":
            j.update(stage=e["stage"], running_secs=e["running_secs"], cost_usd=e["cost_usd"], message=e.get("message"))
        elif e["event"] == "provider_cost":
            j.update(cost_usd=e["cost_usd"])
    jobs.pop(None, None)
    return {"cap_usd": budget["cap_usd"], "reserve_usd": budget["reserve_usd"], "settled_usd": b["settled_usd"],
            "provider_used_usd": b["provider_used_usd"], "open_reservations_usd": b["reserved_usd"],
            "committed_usd": b["committed_usd"], "jobs": jobs,
            "prices": {k: 60 * j["price_per_min"] for k, j in jobs.items() if "price_per_min" in j}}


def freeze() -> None:
    """The inputs of the main run that are fixed before a pilot (plan, "Frozen inputs"), and what the task
    stream covers. CPU only; about two minutes."""
    from ccl.sft import file_sha256
    fams, tools = s2.load_practice()
    bucket = L.buckets(fams)
    reps = {}
    for r in range(5):
        st = L.Stream(fams, tools, r)
        tasks = st.first(L.PASSES * L.PASS_TASKS)
        seed_f = L.seed_families(fams, r)
        rows, stats = s2.oracle_rows(L.seed_tasks(fams, tools, r))
        skeleton = Counter()
        for t in tasks:  # the gold tool sequence of a task, ignoring every value
            skeleton[(t.template,)] += 1
        reps[str(r)] = {"seed_set_sha256": s2.rows_sha256(rows), "seed_episodes": stats["episodes"],
                        "seed_families": [f["id"] for f in seed_f],
                        "stream_sha256": L.tasks_sha256(tasks), "stream_tasks": len(tasks),
                        "stream_dropped_by_selection_check": len(st.dropped),
                        "stream": L.diversity(st.family, tasks, seed_f),
                        "first_cycle_of_the_oracle_arm": L.diversity(st.family[:L.CYCLE_ACCEPT], tasks[:L.CYCLE_ACCEPT], seed_f),
                        "seed": L.diversity([f for f in seed_f for _ in range(L.SEED_INSTANCES)]),
                        "times_a_family_recurs_in_the_stream": dict(sorted(Counter(skeleton.values()).items())),
                        "distinct_requests": len({t.request for t in tasks})}
    out = {"data_v2_manifest_sha256": file_sha256(str(s2.V2 / "manifest.json")),
           "student": {"repo": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"},
           "practice_space": {"families": len(fams), "tools_in_gold_graphs": len({t for f in fams for t in f["tools"]}),
                              "edges": len({e for f in fams for e in f["edges"]})},
           "replicates": reps,
           "curriculum_buckets": dict(sorted(Counter(bucket.values()).items()))}
    stage1 = OUT / "stage1.json"
    if stage1.exists():
        res = json.loads(stage1.read_text())
        out["engine_calibration"] = {k: res["engine"].get(k) for k in ("passed", "criteria", "seconds_per_episode", "speed_up")}
        out["teacher_qualification"] = {k: {x: v[x] for x in ("model", "acceptance", "acceptance_by_hops", "qualifies", "complete")}
                                        for k, v in res["teachers"].items()}
    path = ROOT / "configs/study02/e2/frozen.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")


def main() -> None:
    if sys.argv[1:] == ["freeze"]:
        return freeze()
    OUT.mkdir(parents=True, exist_ok=True)
    money = books()
    # a teacher's job key (e2s1_teacher_qwen3_32b) names its result directory (teacher_Qwen3-32B) up to case and dashes
    by_key = {k.split("e2s1_")[-1]: v for k, v in money["prices"].items() if "teacher_" in k}
    price = {d.name: by_key.get(d.name.lower().replace("-", "_")) for d in RUNS.glob("teacher_*")}
    res = {"engine": engine(), "engine_revision_r6": engine("2"), "seed": seed(), "bench": bench(),
           "probe": probe(), "teachers": teachers(price), "books": money}
    if res["seed"].get("dev_success") is not None:
        res["headroom"] = headroom(res["seed"]["dev_success"])
    (OUT / "stage1.json").write_text(json.dumps(res, indent=1) + "\n")
    e, s, b, p, t = res["engine"], res["seed"], res["bench"], res["probe"], res["teachers"]
    md = ["# Experiment 2, Stage 1: measurements (generated by scripts/s2_e2_stage1.py)", ""]
    h = res.get("headroom")
    if e and p and h:
        r0 = list(e["runs"].values())
        ok = lambda b: "met" if b else "**not met**"  # noqa: E731
        md += ["## Gates of plan §7, Stage 1", ""] + table(["", "criterion", "measured", ""], [
            ["(a)", "at most 0.36 s per episode (four times faster)",
             " and ".join(f"{v['seconds_per_episode']:.3f}" for v in r0) + f" s ({e['speed_up']:.1f} times faster)",
             ok(e["criteria"]["speed"]) + ": between 2 and 4 times, so re-cost and ask again"],
            ["(a)", "at most 1% of episodes change outcome", " and ".join(pc(v["changed_share"]) for v in r0), ok(e["criteria"]["flips"])],
            ["(a)", "success within 1 point of Experiment 1",
             " and ".join(f"{v['success_difference_points']:+.2f}" for v in r0) + " points", ok(e["criteria"]["success"])],
            ["(b)", "the 1.5B scoring Experiment 1 left out", "not run: approved for planning only", "-"],
            ["(c)", "at least 20% of probed tasks accepted", f"{p['accepted']} of {p['tasks']} ({pc(p['yield'])})",
             ok(p["yield"] >= GATE["yield"])],
            ["(c)", "seed `dev` success at least 15 points below the 1,600-family adapters'",
             f"seed {pc(h['seed_dev'])}; the comparator is part of (b) and not measured. Estimated between "
             f"{pc(h['bracket_for_1600_family_dev'][0])} and {pc(h['bracket_for_1600_family_dev'][1])}",
             ("met under the estimate" if h["met_under_the_estimate"] else "**not met under the estimate**")
             + f", by {h['margin_points_at_bracket'][0]:.0f} points or more"],
            ["(d)", "a teacher accepted on at least 60% of 200 tasks and 40% of the four-hop ones",
             "; ".join(f"{v['model']['repo'].split('/')[-1]}: {pc(v['acceptance'])} and {pc(v['acceptance_by_hops'].get('4'))}"
                       + ("" if v["complete"] else " (incomplete)") for v in t.values()) or "-",
             "a teacher qualifies" if any(v["qualifies"] for v in t.values()) else "**no teacher qualifies**"]])
    if e:
        md += ["## (a) Rollout engine on the GPU", "",
               "Own-family tasks of two Experiment 1 adapters, scored again with the new engine and compared with "
               "the committed Experiment 1 rows. Pass: at most 0.36 s per episode, at most 1% of episodes changing "
               "outcome, success within 1 point.", ""]
        md += table(["adapter", "episodes", "s per episode", "Experiment 1", "speed-up", "outcome changed",
                     "success", "Experiment 1", "same text as full re-encoding (24 tasks)"],
                    [[k, v["episodes"], f"{v['seconds_per_episode']:.3f}", f"{E1_OWN_SECONDS:.2f}",
                      f"{v['speed_up']:.1f}x", f"{v['changed_outcome']} ({pc(v['changed_share'])})", pc(v["success"]),
                      pc(v["success_experiment_1"]), v["same_text_as_full_re_encoding"]] for k, v in e["runs"].items()])
        md += [f"Criteria: speed {'met' if e['criteria']['speed'] else 'NOT met'}, changed outcomes "
               f"{'met' if e['criteria']['flips'] else 'NOT met'}, success {'met' if e['criteria']['success'] else 'NOT met'}. "
               f"**Calibration {'passed' if e['passed'] else 'did not pass'}**: the engine reproduces Experiment 1's "
               f"outcomes and is {e['speed_up']:.1f} times faster, where four times was required. The plan's rule for "
               "between two and four times is to re-cost and ask again.", ""]
        r6 = res["engine_revision_r6"]
        if r6:
            md += ["### Engine revision R6: failed, switched off (plan, deviation 1)", "",
                   "One revision was tried after the first calibration: attention per key-value head during decoding. "
                   "It was faster and wrong, and the job stopped itself. Nothing else in Stage 1 used it.", ""]
            md += table(["adapter", "episodes", "s per episode", "outcome changed", "success", "Experiment 1",
                         "same text as full re-encoding (24 tasks)"],
                        [[k, v["episodes"], f"{v['seconds_per_episode']:.3f}", f"{v['changed_outcome']} ({pc(v['changed_share'])})",
                          pc(v["success"]), pc(v["success_experiment_1"]), v["same_text_as_full_re_encoding"]]
                         for k, v in r6["runs"].items()])
    if s:
        md += ["## (c) Seed adapter of replicate 0", "",
               f"400 oracle episodes of 50 one- and two-hop families (SHA-256 `{s['seed_set_sha256'][:16]}…`), "
               f"{s['steps']} steps at {s['seconds_per_step']} s per step.", ""]
        if s.get("dev_success") is not None:
            md += table(["", "all", *[f"{h} hops" for h in s["dev_by_hops"]], "strict"],
                        [["`dev` success", pc(s["dev_success"]), *[pc(v) for v in s["dev_by_hops"].values()],
                          pc(s["dev_strict"])]])
    if b:
        md += ["## Training cost on a cycle-1 buffer", "",
               f"24 optimizer steps of 8 episodes on {b['episodes']} episodes (seed set and oracle episodes for the "
               "first 480 stream tasks). Gradients are the same in the three settings.", ""]
        md += table(["gradient checkpointing", "s per step", "tokens per step", "peak GPU memory (GB)", "loss, last step"],
                    [[k, v.get("seconds_per_step", "out of memory"), v.get("tokens_per_step", "-"),
                      f"{v['peak_mem_gb']:.1f}" if "peak_mem_gb" in v else "-",
                      f"{v['last_loss']:.4f}" if "last_loss" in v else "-"] for k, v in b["settings"].items()])
    if p:
        md += ["## (c) Yield probe: the first 400 stream tasks of replicate 0, four sampled attempts each", "",
               f"**Yield: {p['accepted']} of {p['tasks']} tasks accepted ({pc(p['yield'])}); the gate is 20%.** "
               f"Single attempts pass the validator {pc(p['attempt_success'])} of the time and are accepted "
               f"{pc(p['attempt_accepted'])}; {pc(p['tasks_with_a_success'])} of tasks have at least one passing attempt. "
               f"{p['seconds_per_attempt']:.3f} s per sampled attempt ({p['tokens_read_per_attempt']:.0f} tokens read, "
               f"{p['tokens_written_per_attempt']:.0f} written). At this yield a cycle of 480 accepted episodes needs "
               f"{p['tasks_per_cycle_at_this_yield']:.0f} tasks (cap 2,400).", ""]
        for title, key in (("By number of hops", "by_hops"), ("By number of list selections", "by_selections"),
                           ("By distance from the seed set", "by_novelty")):
            md += [f"### {title}", ""] + table(
                ["", "tasks offered", "accepted", "yield", "single-attempt success"],
                [[k, v["offered"], v["accepted"], pc(v["yield"]), pc(v["attempt_success"])] for k, v in p[key].items()])
        md += ["### What the accepted set covers (plan §3.3)", ""]
        cols = list(p["diversity"])
        d = p["diversity"]
        get = lambda c, *ks: (lambda v: v if not isinstance(v, float) else f"{v:.2f}")(  # noqa: E731
            __import__("functools").reduce(lambda a, k: a.get(k, "-") if isinstance(a, dict) else "-", ks, d[c]))
        rows = [["tasks", *[get(c, "tasks") for c in cols]], ["distinct families", *[get(c, "families") for c in cols]],
                ["distinct base graphs", *[get(c, "base_graphs") for c in cols]],
                ["tools covered (of 79)", *[get(c, "tools") for c in cols]],
                ["edges covered (of 229)", *[get(c, "edges") for c in cols]],
                *[[f"share with {h} hops", *[get(c, "hops", str(h)) for c in cols]] for h in (1, 2, 3, 4)],
                ["gold calls per task", *[get(c, "mean_gold_calls") for c in cols]],
                ["list selections per task", *[get(c, "mean_selections") for c in cols]],
                ["share outside the seed: new family", *[get(c, "outside_reference", "new_family") for c in cols]],
                ["share outside the seed: new tool", *[get(c, "outside_reference", "new_tool") for c in cols]],
                ["share outside the seed: new edge", *[get(c, "outside_reference", "new_edge") for c in cols]],
                ["share deeper than the seed", *[get(c, "outside_reference", "deeper") for c in cols]]]
        md += table(["", *cols], rows)
        a = p["accepted_episodes"]
        md += ["### The accepted episodes beside the oracle's solution of the same task", "",
               f"Of {a['n']} accepted episodes: {a['strict']} strictly successful (no unnecessary or "
               f"dependency-unsafe call); {a['same_calls_as_oracle']} make exactly the oracle's calls, "
               f"{a['more_calls_than_oracle']} make more; {a['fewer_turns_than_oracle']} take fewer turns than the "
               f"oracle and {a['more_turns_than_oracle']} more. {a['tasks_with_two_or_more_distinct_passing_attempts']} "
               "tasks have two or more different passing attempts.", "",
               f"**Found by elimination: {len(a['found_by_elimination']['accepted_episodes'])} of {a['n']} accepted episodes** "
               f"(stream tasks {a['found_by_elimination']['accepted_episodes']}), and "
               f"{a['found_by_elimination']['admissible_attempts']} of {a['found_by_elimination']['of_admissible_attempts']} "
               "admissible attempts. In the simulated worlds a record off the gold path refers to records that do not "
               "exist, so an attempt can ask for the referenced record of one listed candidate after another until one "
               "is found, without applying the selection rule the request states (`ccl.loop.by_elimination`). The "
               "validator accepts it: the right record was reached and the right fact stated. Such episodes are not "
               "strictly successful.", ""]
        if p["replay"]:
            md += [f"CPU replay in a separate process: {p['replay']['accepted_attempts_replayed']} admissible attempts "
                   f"replayed, {p['replay']['failed']} failed.", ""]
        md += ["### By curriculum bucket", ""] + table(
            ["bucket", "tasks offered", "accepted", "yield"],
            [[k, v["offered"], v["accepted"], pc(v["yield"])] for k, v in p["by_bucket"].items()])
    if t:
        md += ["## (d) Teacher qualification", "",
               "200 practice tasks, 50 per number of hops, few-shot prompt, one greedy attempt and one sampled retry. "
               "Qualifies at 60% overall and 40% at four hops.", ""]
        md += table(["model", "accepted", *[f"{h} hops" for h in (1, 2, 3, 4)], "greedy only", "qualifies",
                     "s per attempt", "USD per accepted demonstration", "USD for 14,400"],
                    [[v["model"]["repo"], pc(v["acceptance"]), *[pc(v["acceptance_by_hops"].get(str(h))) for h in (1, 2, 3, 4)],
                      pc(v["greedy_acceptance"]), ("yes" if v["qualifies"] else "no") + ("" if v["complete"] else " (incomplete)"),
                      f"{v['seconds_per_attempt']:.2f}",
                      f"{v['usd_per_accepted_demonstration']:.4f}" if "usd_per_accepted_demonstration" in v else "-",
                      f"{v['usd_for_14400_demonstrations']:.0f}" if "usd_for_14400_demonstrations" in v else "-"]
                     for v in t.values()])
        md += ["How the accepted demonstrations were reached. The qualification rule is the pre-registered acceptance "
               "rule; the other two rows are stricter readings, reported beside it.", ""]
        md += table(["model", "", "accepted", *[f"{h} hops" for h in (1, 2, 3, 4)]],
                    [[v["model"]["repo"].split("/")[-1], label, pc(x["overall"]), *[pc(x[str(h)]) for h in (1, 2, 3, 4)]]
                     for v in t.values() for label, x in (
                         ("as pre-registered", {"overall": v["acceptance"], **v["acceptance_by_hops"]}),
                         (f"without the {v['accepted_by_elimination']} demonstrations found by elimination", v["acceptance_without_elimination"]),
                         (f"strictly successful only ({v['accepted_strict']} of {v['accepted_n']})", v["acceptance_strict_only"]))])
    m = res["books"]
    md += ["## Money", ""] + table(
        ["job", "flavour", "outcome", "billed minutes", "reserved USD", "billed USD"],
        [[k, j.get("flavor"), j.get("stage", "open"), round(j.get("cost_usd", 0) / j["price_per_min"]) if "cost_usd" in j else "-",
          f"{j.get('reserved_usd', 0):.2f}", f"{j['cost_usd']:.2f}" if "cost_usd" in j else "-"] for k, j in m["jobs"].items()])
    md += [f"Stage 1 total: **{m['settled_usd']:.2f} USD** billed for closed jobs; provider-reported above the baseline: "
           f"{m['provider_used_usd']:.2f} USD; open reservations {m['open_reservations_usd']:.2f} USD. "
           f"Cap {m['cap_usd']:.2f} USD.", ""]
    (OUT / "stage1.md").write_text("\n".join(md))
    print(f"wrote {OUT / 'stage1.md'}")


if __name__ == "__main__":
    main()
