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

def engine() -> dict:
    runs = {}
    for name in ("e1_s15_D1600_r0", "e1_s15_D25_r0"):
        r = result(f"calib_{name}")
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
            "seconds_per_episode": sum(v["seconds_per_episode"] for v in runs.values()) / len(runs),
            "speed_up": E1_OWN_SECONDS / (sum(v["seconds_per_episode"] for v in runs.values()) / len(runs))}


# ------------------------------------------------------------------ (c) seed adapter and yield probe

def seed() -> dict:
    r = result("r0_seed")
    if not r:
        return {}
    rows = [json.loads(l) for l in open(RUNS / "r0_seed/metrics.jsonl")] if (RUNS / "r0_seed/metrics.jsonl").exists() else []
    by = defaultdict(list)
    for x in rows:
        by[x["hops"]].append(x["success"])
    return {**{k: r.get(k) for k in ("seed_set_sha256", "steps", "seconds_per_step", "train_seconds", "peak_mem_gb",
                                     "adapter_sha256", "dev_success", "dev_tasks", "dev_seconds", "dev_engine")},
            "dev_by_hops": {str(h): sum(v) / len(v) for h, v in sorted(by.items())},
            "dev_strict": sum(x["strict_success"] for x in rows) / len(rows) if rows else None,
            "dev_seconds_per_episode": r["dev_seconds"] / r["dev_tasks"] if r.get("dev_tasks") else None}


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
                                  "tasks_with_two_or_more_distinct_passing_attempts": sum(d > 1 for d in distinct_ok)},
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
        t = {"model": r["model"], "complete": r["complete"], "acceptance": r["acceptance"],
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    money = books()
    price = {f"teacher_{k.split('teacher_')[1]}": v for k, v in money["prices"].items() if "teacher_" in k}
    names = {"teacher_qwen3_14b": "teacher_Qwen3-14B", "teacher_qwen3_30b_a3b": "teacher_Qwen3-30B-A3B-Instruct-2507"}
    price = {names.get(k, k): v for k, v in price.items()}
    res = {"engine": engine(), "seed": seed(), "bench": bench(), "probe": probe(), "teachers": teachers(price),
           "books": money}
    (OUT / "stage1.json").write_text(json.dumps(res, indent=1) + "\n")
    e, s, b, p, t = res["engine"], res["seed"], res["bench"], res["probe"], res["teachers"]
    md = ["# Experiment 2, Stage 1: measurements (generated by scripts/s2_e2_stage1.py)", ""]
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
               f"**Calibration {'passed' if e['passed'] else 'did not pass'}.**", ""]
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
               "tasks have two or more different passing attempts.", ""]
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
