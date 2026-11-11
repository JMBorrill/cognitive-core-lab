"""Study 02, Experiment 2, Stage 2: the pilot (replicate 0, cycles 1 and 2 of `practice`) against its gates.

    python scripts/s2_e2_stage2.py     # results/study02/e2/stage2.{json,md}

Reads what scripts/s2_launch.py pulled into results/study02/runs/e2, the Stage 2 ledger, and the audit's
verdicts. No GPU, no sealed split. Gates (plan §7): in both cycles at least 20% of attempted tasks accepted;
at most 4 false accepts in 200 audited episodes; cost per cycle within twice the model's; no fall of more
than 5 points on `dev` between cycles.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccl import loop as L  # noqa: E402
from ccl import s2  # noqa: E402

RUNS = ROOT / "results/study02/runs/e2"
OUT = ROOT / "results/study02/e2"
ARM = "r0_practice"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pc(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def table(head: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + \
        ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


def dev_rows(d: Path) -> list[dict]:
    p = d / "metrics.jsonl"
    return [r for r in map(json.loads, open(p)) if r["split"] == "v2/dev"] if p.exists() else []


def by_hops(rows: list[dict]) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[r["hops"]].append(r["success"])
    return {"all": sum(r["success"] for r in rows) / len(rows), **{str(h): sum(v) / len(v) for h, v in sorted(g.items())}}


def shortcuts(d: Path) -> dict:
    """Per scored episode of a run: was a success reached by the shortcut (`ccl.loop.by_elimination`, by a
    "not found"; `ccl.loop.by_empty_list`). Derived from the run's traces when they are here (pull with
    --traces) and written to `shortcuts.json`, which is committed; otherwise read from that file."""
    import gzip
    out = d / "shortcuts.json"
    if (d / "traces.jsonl.gz").exists():
        flags = {}
        with gzip.open(d / "traces.jsonl.gz", "rt") as f:
            for x in map(json.loads, f):
                flags[f"{x['split']}|{x['task_id']}"] = [L.by_elimination(x["trace"]), L.by_empty_list(x["trace"])]
        out.write_text(json.dumps(dict(sorted(flags.items()))) + "\n")
    return json.loads(out.read_text()) if out.exists() else {}


def by_hops_clean(d: Path, split: str = "v2/dev") -> dict:
    """Success by hops, and success not counting episodes reached by the shortcut."""
    flags = shortcuts(d)
    rows = [r for r in map(json.loads, open(d / "metrics.jsonl")) if r["split"] == split]
    g = defaultdict(lambda: [0, 0.0, 0, 0])
    for r in rows:
        nf, em = flags.get(f"{split}|{r['task_id']}", [False, False])
        for k in (str(r["hops"]), "all"):
            g[k][0] += 1
            g[k][1] += r["success"]
            g[k][2] += bool(r["success"]) and (nf or em)
            g[k][3] += bool(r["success"]) and nf
    return {k: {"episodes": v[0], "success": v[1] / v[0], "successes": int(v[1]), "by_shortcut": v[2], "by_not_found": v[3],
                "success_without_shortcut": (v[1] - v[2]) / v[0]} for k, v in sorted(g.items())} if flags else {}


def diff_interval(a: tuple[int, int], b: tuple[int, int]) -> dict:
    """Difference of two independent proportions b - a with a 95% normal interval (different tasks in each)."""
    pa, pb = a[0] / a[1], b[0] / b[1]
    se = math.sqrt(pa * (1 - pa) / a[1] + pb * (1 - pb) / b[1])
    return {"difference": pb - pa, "lo": pb - pa - 1.96 * se, "hi": pb - pa + 1.96 * se}


def cycles() -> list[dict]:
    fams, tools = s2.load_practice()
    arm = L.Practice(RUNS / ARM, 0, "practice", fams, tools)
    seed_f = L.seed_families(fams, 0)
    out = []
    for c in arm.state["cycles"]:
        k = c["cycle"]
        recs = [x for f in sorted((RUNS / ARM / "attempts").glob(f"c{k}_*.jsonl.gz")) for x in s2.read_gz(f)]
        offered, accepted = c["offered"], set(c["accepted"])
        by = defaultdict(list)
        for x in recs:
            if x["index"] in set(offered):
                by[x["index"]].append(x)
        task = {i: arm.stream.get(i) for i in offered}
        fam = {i: arm.stream.family[i] for i in offered}

        def split(key) -> dict:
            g = defaultdict(lambda: [0, 0, 0, 0])
            for i in offered:
                v = g[str(key(i))]
                v[0] += 1
                v[1] += i in accepted
                v[2] += sum(x["success"] for x in by[i])
                v[3] += len(by[i])
            return {name: {"offered": v[0], "accepted": v[1], "yield": v[1] / v[0], "attempt_success": v[2] / v[3]}
                    for name, v in sorted(g.items())}
        picks = {i: L.choose(by[i]) for i in offered if i in accepted}
        admissible = [x for i in offered for x in by[i] if not x["rejection"]]
        cut = {i for i, p in picks.items() if L.shortcut(p["trace"])}
        clean = defaultdict(lambda: [0, 0])
        for i in offered:
            clean[str(fam[i]["hops"])][0] += 1
            clean[str(fam[i]["hops"])][1] += i in accepted and i not in cut
        acc = [i for i in offered if i in accepted]
        lo = (k - 1) * L.CYCLE_ACCEPT  # the oracle arm keeps the next 480 stream tasks in every cycle
        oracle = [arm.stream.get(i) and i for i in range(lo, lo + L.CYCLE_ACCEPT)]
        res = json.loads((RUNS / ARM / f"c{k}/result.json").read_text())
        probe = json.loads((RUNS / ARM / "result.json").read_text()) if k == 1 and (RUNS / ARM / "result.json").exists() else {}
        rollout = res.get("rollout_seconds_this_job", 0) + probe.get("rollout_seconds", 0)
        sampled = res.get("attempts_sampled_this_job", 0) + probe.get("attempts_sampled_now", 0)
        out.append({
            "cycle": k, "tasks_offered": len(offered), "accepted": len(acc), "yield": len(acc) / len(offered),
            "attempt_success": sum(x["success"] for i in offered for x in by[i]) / sum(len(by[i]) for i in offered),
            "by_hops": split(lambda i: fam[i]["hops"]),
            "by_selections": split(lambda i: sum(t.startswith("rule=") for t in task[i].tags)),
            "accepted_strict": sum(p["strict"] for p in picks.values()),
            "accepted_by_elimination": sum(L.by_elimination(p["trace"]) for p in picks.values()),
            "accepted_by_empty_list": sum(L.by_empty_list(p["trace"]) for p in picks.values()),
            "accepted_by_shortcut": len(cut),
            "accepted_by_shortcut_by_hops": dict(sorted(Counter(str(fam[i]["hops"]) for i in cut).items())),
            "by_hops_without_shortcut": {h: {"offered": v[0], "accepted": v[1], "yield": v[1] / v[0]} for h, v in sorted(clean.items())},
            "admissible_by_elimination": sum(L.by_elimination(x["trace"]) for x in admissible),
            "admissible_attempts": len(admissible), "replay_failed": c.get("replay_failed", 0),
            "diversity": {"offered": L.diversity([fam[i] for i in offered], [task[i] for i in offered], seed_f),
                          "accepted by practice": L.diversity([fam[i] for i in acc], [task[i] for i in acc], seed_f),
                          "oracle arm, this cycle": L.diversity([arm.stream.family[i] for i in oracle],
                                                               [arm.stream.get(i) for i in oracle], seed_f)},
            "rollout_seconds": rollout, "attempts_sampled": sampled,
            "seconds_per_attempt": rollout / sampled if sampled else None,
            "steps": res.get("steps"), "train_seconds": res.get("train_seconds"), "seconds_per_step": res.get("seconds_per_step"),
            "dev_seconds": res.get("dev_seconds_this_job"), "dev": by_hops(dev_rows(RUNS / ARM / f"c{k}")) if dev_rows(RUNS / ARM / f"c{k}") else None,
            "adapter_sha256": res.get("adapter_sha256"), "train_data_sha256": res.get("train_data_sha256")})
    return out


def audit() -> dict:
    p = RUNS / ARM / "audit.json"
    if not p.exists():
        return {}
    a = json.loads(p.read_text())
    v = RUNS / ARM / "audit_verdicts.json"
    verdicts = json.loads(v.read_text()) if v.exists() else {}
    total = sum(x["weight"] for x in a["sample"])
    false = {"read": 0.0, "with_elimination": 0.0}
    kinds = Counter()
    n = Counter()
    for x in a["sample"]:
        key = f"{x['cycle']}:{x['index']}"
        human = verdicts.get(key, {}).get("false_accept")
        elim = "found_by_elimination" in x["flags"] or "unobserved_identifier" in x["flags"]
        n["flagged"] += bool(x["flags"])
        n["read"] += human is not None
        n["false_by_reading"] += bool(human)
        n["by_elimination_or_unobserved"] += elim
        if human:
            kinds[verdicts[key].get("kind", "?")] += 1
        false["read"] += x["weight"] * bool(human)
        false["with_elimination"] += x["weight"] * bool(human or elim)
    return {"accepted_episodes": a["accepted_episodes"], "not_strict": a["not_strict"], "sampled": a["sampled"],
            **n, "verdicts_recorded": len(verdicts), "false_accepts_by_kind": dict(sorted(kinds.items())),
            "false_accept_rate_by_reading": false["read"] / total,
            "false_accept_rate_counting_elimination": false["with_elimination"] / total,
            "limit": 0.02}


def comparator() -> dict:
    """`dev` of Experiment 1's three 1,600-family adapters, if it was scored (after the fact, labelled so)."""
    v = [by_hops(dev_rows(d)) for d in sorted(RUNS.glob("e1dev_e1_s15_D1600_r*")) if dev_rows(d)]
    return {} if not v else {"adapters": len(v), **{k: sum(x[k] for x in v) / len(v) for k in v[0]}}


def r6() -> dict:
    out = {}
    for d in sorted(RUNS.glob("calib3_*")):
        r = json.loads((d / "result.json").read_text())
        out[d.name] = {k: r[k] for k in ("episodes", "seconds_per_episode", "success", "success_experiment_1",
                                         "changed_outcome", "episodes_compared", "same_text_as_full_re_encoding")}
    if out:
        v = list(out.values())
        out["passed"] = len(v) == 2 and all(x["seconds_per_episode"] <= 0.36 and x["changed_outcome"] <= 0.01 * x["episodes_compared"]
                                            and abs(x["success"] - x["success_experiment_1"]) <= 0.01 for x in v)
    return out


def books() -> dict:
    launch = load("s2_launch")
    try:
        budget, path = launch.load_budget("e2_stage2")
    except FileNotFoundError:
        return {}
    ledger = launch.read_ledger(path)
    b = launch.books(ledger)
    jobs = {}
    for e in ledger:
        j = jobs.setdefault(e.get("key"), {})
        if e["event"] == "reserve":
            j.update(flavor=e["flavor"], reserved_usd=e["reserved_usd"], price_per_min=e["price_per_min"])
        elif e["event"] == "settled":
            j.update(stage=e["stage"], cost_usd=e["cost_usd"])
        elif e["event"] == "provider_cost":
            j.update(cost_usd=e["cost_usd"])
    jobs.pop(None, None)
    return {"cap_usd": budget["cap_usd"], "settled_usd": b["settled_usd"], "provider_used_usd": b["provider_used_usd"],
            "open_reservations_usd": b["reserved_usd"], "jobs": jobs}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cyc = cycles()
    S1 = json.loads((OUT / "stage1.json").read_text())
    R = load("s2_e2_rebudget")
    x = R.inputs()
    seed_dev = by_hops(dev_rows(RUNS / "r0_seed"))
    clean_dev = {"seed adapter": by_hops_clean(RUNS / "r0_seed"),
                 **{f"after cycle {c['cycle']}": by_hops_clean(RUNS / ARM / f"c{c['cycle']}") for c in cyc if c["dev"]}}
    oracle_ref = {d.name.replace("calib_", ""): by_hops_clean(d, "own") for d in sorted(RUNS.glob("calib_e1_s15_*"))}
    strict = [c["accepted_strict"] / c["accepted"] for c in cyc]
    # what the cost model, at Stage 1's measurements, expects one cycle to take in GPU-seconds
    model = [{"rollouts": min(L.CYCLE_ACCEPT / x["yield"], L.CYCLE_TASK_CAP) * L.ATTEMPTS * x["attempt_seconds"],
              "training": (L.SEED_FAMILIES * L.SEED_INSTANCES + c["cycle"] * L.CYCLE_ACCEPT) / 4 * x["train_seconds_per_step"],
              "dev": 600 * x["seconds_per_episode"]["v2/dev"]} for c in cyc]
    for c, m in zip(cyc, model):
        c["gpu_seconds"] = (c["rollout_seconds"] or 0) + (c["train_seconds"] or 0) + (c["dev_seconds"] or 0)
        c["model_gpu_seconds"] = sum(m.values())
        c["cost_ratio_to_model"] = c["gpu_seconds"] / c["model_gpu_seconds"]
    dev = [seed_dev["all"]] + [c["dev"]["all"] for c in cyc if c["dev"]]
    au = audit()
    gates = {"yield": all(c["yield"] >= 0.20 for c in cyc) and len(cyc) == 2,
             "audit": bool(au) and au["false_accept_rate_counting_elimination"] <= au["limit"] and au["read"] == au["flagged"],
             "cost": all(c["cost_ratio_to_model"] <= 2 for c in cyc),
             "dev": all(b >= a - 0.05 for a, b in zip(dev, dev[1:])) and len(dev) == 3}
    climb, climb_clean = {}, {}
    if len(cyc) == 2:
        for h in ("1", "2", "3", "4"):
            a, b = cyc[0]["by_hops"].get(h), cyc[1]["by_hops"].get(h)
            if a and b:
                climb[h] = diff_interval((a["accepted"], a["offered"]), (b["accepted"], b["offered"]))
                a, b = cyc[0]["by_hops_without_shortcut"][h], cyc[1]["by_hops_without_shortcut"][h]
                climb_clean[h] = diff_interval((a["accepted"], a["offered"]), (b["accepted"], b["offered"]))
    res = {"cycles": cyc, "seed_dev": seed_dev, "dev_by_adapter": dev, "comparator_dev_1600_families": comparator(),
           "audit": au, "gates": gates, "passed": all(gates.values()), "yield_change_by_hops": climb,
           "yield_change_by_hops_without_shortcut": climb_clean, "dev_without_shortcut": clean_dev,
           "oracle_trained_reference_own_family": oracle_ref,
           "not_strict_share_by_cycle": [1 - x for x in strict],
           "strictness_warning": {"rise_points": 100 * (strict[0] - strict[1]) if len(strict) == 2 else None, "stops_at": 10.0},
           "engine_revision_r6": r6(), "books": books()}
    if len(cyc) == 2:  # the remaining stages again, with what two cycles measured in place of one probe
        y = sum(c["accepted"] for c in cyc) / sum(c["tasks_offered"] for c in cyc)
        att = sum(c["rollout_seconds"] for c in cyc) / sum(c["attempts_sampled"] for c in cyc)
        step = sum(c["train_seconds"] for c in cyc) / sum(c["steps"] for c in cyc)
        x2 = {**x, "train_seconds_per_step": step}
        res["rebudget"] = {"yield": y, "attempt_seconds": att, "train_seconds_per_step": step,
                           "by_the_plans_rules": {k: v for k, v in R.cost(x2, y, att).items() if k.endswith("_usd") and "stage1" not in k and not isinstance(v, (list, dict))},
                           "sealed_on_new_engine": {k: v for k, v in R.cost(x2, y, att, True).items() if k.endswith("_usd") and "stage1" not in k and not isinstance(v, (list, dict))}}
    (OUT / "stage2.json").write_text(json.dumps(res, indent=1) + "\n")

    ok = lambda b: "met" if b else "**not met**"  # noqa: E731
    md = ["# Experiment 2, Stage 2: the pilot (generated by scripts/s2_e2_stage2.py)", "",
          "Replicate 0, `practice`, cycles 1 and 2. Cycle 1's rollouts are the seed adapter's; cycle 2's are those of "
          "the adapter trained on the seed set and cycle 1's accepted episodes.", "", "## Gates of plan §7, Stage 2", ""]
    md += table(["criterion", "measured", ""], [
        ["in both cycles at least 20% of attempted tasks accepted", " and ".join(pc(c["yield"]) for c in cyc), ok(gates["yield"])],
        ["verifier audit: at most 4 false accepts in 200 (2%)",
         (f"{au['sampled']} audited, {au['flagged']} differ from the oracle and were read ({au['read']} verdicts): "
          f"{pc(au['false_accept_rate_by_reading'])} by reading, {pc(au['false_accept_rate_counting_elimination'])} "
          "counting episodes found by elimination or with an unobserved identifier") if au else "not run", ok(gates["audit"])],
        ["cost per cycle within twice the model's", " and ".join(f"{c['cost_ratio_to_model']:.2f}x" for c in cyc), ok(gates["cost"])],
        ["no fall of more than 5 points on `dev` between cycles", " → ".join(pc(v) for v in dev), ok(gates["dev"])]])
    md += [f"**Pilot {'passed' if res['passed'] else 'did not pass'}.**", ""]
    if au:
        md += ["The audit counts, as recorded before the pilot, every sampled episode found by elimination (a \"not found\" "
               "on one candidate, then another) or using an unobserved identifier. By reading, the same shortcut also "
               f"appears through empty lists, and one episode states a fact for the wrong record: "
               f"{au['false_by_reading']} of {au['sampled']} sampled episodes are false accepts by reading "
               f"({pc(au['false_accept_rate_by_reading'])} weighted; kinds: {au['false_accepts_by_kind']}).", ""]
    w = res["strictness_warning"]
    if w["rise_points"] is not None:
        md += [f"Accepted but not strictly successful: {' and '.join(pc(x) for x in res['not_strict_share_by_cycle'])} of the two "
               f"cycles' accepted episodes, a rise of {w['rise_points']:.1f} points. The plan's warning stops a run above 10.", ""]
    md += ["## Does practice climb?", "",
           "Yield by number of hops. The two cycles see different tasks of the same stream, so the comparison is "
           "between proportions on different tasks.", ""]
    md += table(["hops", *[f"cycle {c['cycle']}: offered" for c in cyc][:1], "accepted", "yield",
                 *(["cycle 2: offered", "accepted", "yield", "change (95% interval)"] if len(cyc) == 2 else [])],
                [[h, *[z for c in cyc for z in (c["by_hops"].get(h, {}).get("offered", 0), c["by_hops"].get(h, {}).get("accepted", 0),
                                                 pc(c["by_hops"].get(h, {}).get("yield")))],
                  *([f"{100 * climb[h]['difference']:+.1f} [{100 * climb[h]['lo']:+.1f}, {100 * climb[h]['hi']:+.1f}]"] if h in climb else [])]
                 for h in ("1", "2", "3", "4")])
    if climb_clean:
        md += ["The same without accepted episodes that reached their record by the shortcut (either form):", ""]
        md += table(["hops", "cycle 1 yield", "cycle 2 yield", "change (95% interval)", "accepted by shortcut, cycles 1 and 2"],
                    [[h, *[f"{pc(c['by_hops_without_shortcut'][h]['yield'])} ({c['by_hops_without_shortcut'][h]['accepted']} of {c['by_hops_without_shortcut'][h]['offered']})" for c in cyc],
                      f"{100 * climb_clean[h]['difference']:+.1f} [{100 * climb_clean[h]['lo']:+.1f}, {100 * climb_clean[h]['hi']:+.1f}]",
                      " and ".join(str(c["accepted_by_shortcut_by_hops"].get(h, 0)) for c in cyc)] for h in ("1", "2", "3", "4")])
    md += ["By number of list selections:", ""]
    sels = sorted({k for c in cyc for k in c["by_selections"]})
    md += table(["selections", *[f"cycle {c['cycle']} yield (accepted of offered)" for c in cyc]],
                [[k, *[f"{pc(c['by_selections'][k]['yield'])} ({c['by_selections'][k]['accepted']} of {c['by_selections'][k]['offered']})"
                       if k in c["by_selections"] else "-" for c in cyc]] for k in sels])
    md += ["## `dev` by adapter", ""]
    cmp_ = res["comparator_dev_1600_families"]
    rows = [["seed adapter", *[pc(seed_dev.get(k)) for k in ("all", "1", "2", "3", "4")]]]
    rows += [[f"after cycle {c['cycle']} ({400 + 480 * c['cycle']} episodes, {c['steps']} steps)", *[pc(c["dev"].get(k)) for k in ("all", "1", "2", "3", "4")]]
             for c in cyc if c["dev"]]
    if cmp_:
        rows.append([f"Experiment 1, 1,600 families, oracle labels (mean of {cmp_['adapters']}; scored after the fact)",
                     *[pc(cmp_.get(k)) for k in ("all", "1", "2", "3", "4")]])
    md += table(["adapter", "all", "1 hop", "2", "3", "4"], rows)
    if all(clean_dev.values()):
        md += ["### `dev` without successes reached by the shortcut", "",
               "The validator accepts the shortcut on `dev` as it does in practice. Success, and success not counting "
               "episodes that reached a record by trying candidates in turn (in brackets: how many successes those are).", ""]
        md += table(["adapter", "all", "1 hop", "2", "3", "4"],
                    [[k, *[f"{pc(v[h]['success'])} → {pc(v[h]['success_without_shortcut'])} ({v[h]['by_shortcut']})" for h in ("all", "1", "2", "3", "4")]]
                     for k, v in clean_dev.items()])
        if any(oracle_ref.values()):
            md += ["For comparison, adapters trained on oracle episodes (Experiment 1, own-family tasks, the Stage 1 "
                   "calibration's traces): " + "; ".join(f"`{k}` {v['all']['by_shortcut']} of {v['all']['successes']} successes"
                                                         for k, v in oracle_ref.items() if v) + ".", ""]
    md += ["## What each cycle's buffer holds (plan §3.3)", ""]
    for c in cyc:
        d = c["diversity"]
        cols = list(d)
        g = lambda col, *ks: (lambda v: f"{v:.2f}" if isinstance(v, float) else v)(  # noqa: E731
            __import__("functools").reduce(lambda a, k: a.get(k, "-") if isinstance(a, dict) else "-", ks, d[col]))
        md += [f"### Cycle {c['cycle']}", ""] + table(["", *cols], [
            ["tasks", *[g(x, "tasks") for x in cols]], ["distinct families", *[g(x, "families") for x in cols]],
            ["tools covered (of 79)", *[g(x, "tools") for x in cols]], ["edges covered (of 229)", *[g(x, "edges") for x in cols]],
            *[[f"share with {h} hops", *[g(x, "hops", str(h)) for x in cols]] for h in (1, 2, 3, 4)],
            ["list selections per task", *[g(x, "mean_selections") for x in cols]],
            ["share needing an edge the seed lacks", *[g(x, "outside_reference", "new_edge") for x in cols]],
            ["share deeper than the seed", *[g(x, "outside_reference", "deeper") for x in cols]]])
    md += ["## Accepted episodes", ""] + table(
        ["cycle", "accepted", "strictly successful", "found by elimination (not found)", "through empty lists (post-hoc)",
         "either", "admissible attempts found by elimination", "failed replay"],
        [[c["cycle"], c["accepted"], f"{c['accepted_strict']} ({pc(c['accepted_strict'] / max(1, c['accepted']))})",
          f"{c['accepted_by_elimination']} ({pc(c['accepted_by_elimination'] / max(1, c['accepted']))})",
          c["accepted_by_empty_list"], f"{c['accepted_by_shortcut']} ({pc(c['accepted_by_shortcut'] / max(1, c['accepted']))})",
          f"{c['admissible_by_elimination']} of {c['admissible_attempts']}", c["replay_failed"]] for c in cyc])
    md += ["## Throughput and cost per cycle", ""] + table(
        ["cycle", "tasks offered", "s per attempt", "rollouts (s)", "training (s)", "s per step", "dev (s)", "GPU-seconds", "model", "ratio"],
        [[c["cycle"], c["tasks_offered"], f"{c['seconds_per_attempt']:.3f}" if c["seconds_per_attempt"] else "-",
          f"{c['rollout_seconds']:.0f}", f"{c['train_seconds'] or 0:.0f}", c["seconds_per_step"], f"{c['dev_seconds'] or 0:.0f}",
          f"{c['gpu_seconds']:.0f}", f"{c['model_gpu_seconds']:.0f}", f"{c['cost_ratio_to_model']:.2f}"] for c in cyc])
    e = res["engine_revision_r6"]
    if e:
        md += ["## Engine revision R6, corrected: one calibration", ""] + table(
            ["adapter", "s per episode", "outcomes changed", "success", "Experiment 1", "same text (24 tasks)"],
            [[k.replace("calib3_", ""), f"{v['seconds_per_episode']:.3f}", f"{v['changed_outcome']} of {v['episodes_compared']}",
              pc(v["success"]), pc(v["success_experiment_1"]), v["same_text_as_full_re_encoding"]] for k, v in e.items() if k != "passed"])
        md += [f"**R6 {'passes' if e['passed'] else 'does not pass'} the calibration** (at most 0.36 s per episode, at most 1% of "
               "outcomes changed, success within 1 point).", ""]
    if "rebudget" in res:
        rb = res["rebudget"]
        md += ["## Remaining stages, with the pilot's two cycles as inputs (USD)", "",
               f"Yield {pc(rb['yield'])} over both cycles, {rb['attempt_seconds']:.3f} s per attempt, "
               f"{rb['train_seconds_per_step']:.2f} s per training step; otherwise as in "
               "[budget_stage1.md](../../../experiments/study02_e2_signal/budget_stage1.md).", ""]
        md += table(["", "Stage 3A", "Stage 3B", "3A + 3B", "extension"], [
            [k.replace("_", " "), f"{v['stage3_block_a_usd']:.0f}", f"{v['stage3_block_b_usd']:.0f}",
             f"{v['stage3_block_a_usd'] + v['stage3_block_b_usd']:.0f}", f"{v['extension_usd']:.0f}"]
            for k, v in (("by the plan's rules", rb["by_the_plans_rules"]), ("sealed splits also on the new engine", rb["sealed_on_new_engine"]))])
    m = res["books"]
    if m:
        md += ["## Money", ""] + table(["job", "flavour", "outcome", "reserved USD", "billed USD"], [
            [k, j.get("flavor"), j.get("stage", "open"), f"{j.get('reserved_usd', 0):.2f}", f"{j['cost_usd']:.2f}" if "cost_usd" in j else "-"]
            for k, j in m["jobs"].items()])
        md += [f"Stage 2 total: **{m['settled_usd']:.2f} USD** billed; provider-reported above the stage's baseline: "
               f"{m['provider_used_usd']:.2f} USD. Cap {m['cap_usd']:.2f} USD.", ""]
    (OUT / "stage2.md").write_text("\n".join(md))
    print(f"wrote {OUT / 'stage2.md'}")


if __name__ == "__main__":
    main()
