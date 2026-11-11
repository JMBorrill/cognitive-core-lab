"""Study 02, Experiment 2: the repeated pilot under protocol 2 (amendment 1), against its gates.

    python scripts/s2_e2_stage2r.py     # results/study02/e2/stage2r.{json,md}

Replicate 0, `practice`, cycles 1 and 2, on the corrected engine (R6), with the acceptance rule of protocol 2:
no episode reached by the shortcut and no multi-part answer with a fact given for the wrong part enters a
buffer. Reads what scripts/s2_launch.py pulled into results/study02/runs/e2, the stage's ledger and the audit's
verdicts. No GPU; `dev` and the adversarial set built from it are the only evaluation data.

Reported on `dev`, separately, as pre-registered: success by the original validator (comparable with
Experiments 0 and 1 and with the first pilot), success with each fact given for its own part, success
without the shortcut, and success on the adversarial set (complete candidate records) beside the same
adapter's success on the 350 original tasks those were built from.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccl import adversarial  # noqa: E402
from ccl import loop as L  # noqa: E402
from ccl import s2, verify  # noqa: E402

RUNS = ROOT / "results/study02/runs/e2"
OUT = ROOT / "results/study02/e2"
ARM, SEED, BUDGET = "r0_practice_p2", "r0_seed_p2", "e2_stage2r"
MEASURES = (("success", "original validator"), ("success_parts", "each fact for its own part"),
            ("success_clean", "and no shortcut"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S2 = load("s2_e2_stage2")
pc, table, diff_interval = S2.pc, S2.table, S2.diff_interval


def rows_of(d: Path, split: str) -> list[dict]:
    p = d / "metrics.jsonl"
    return [r for r in map(json.loads, open(p)) if r["split"] == split] if p.exists() else []


def rate(rows: list[dict], key: str, hops=None) -> float | None:
    v = [r[key] for r in rows if hops is None or r["hops"] == hops]
    return sum(v) / len(v) if v else None


def dev_table(d: Path) -> dict:
    rows = rows_of(d, "v2/dev")
    if not rows:
        return {}
    return {"episodes": len(rows), "shortcut_successes": sum(bool(r["success"]) and r["shortcut"] for r in rows),
            "successes": int(sum(r["success"] for r in rows)), "episodes_using_the_shortcut": sum(r["shortcut"] for r in rows),
            "wrong_part_successes": sum(bool(r["success"]) and not r["check_parts"] for r in rows),
            **{k: {"all": rate(rows, k), **{str(h): rate(rows, k, h) for h in (1, 2, 3, 4)}} for k, _ in MEASURES}}


def adversarial_table(d: Path, twin_of: dict) -> dict:
    """The adapter on the 350 twins, and on the 350 `dev` tasks they were built from."""
    adv = {r["task_id"]: r for r in rows_of(d, "v2/dev_adv")}
    dev = {r["task_id"]: r for r in rows_of(d, "v2/dev")}
    if not adv:
        return {}
    pairs = [(dev[twin_of[t]], r) for t, r in adv.items()]
    out = {"tasks": len(pairs), "complete": len(pairs) == len(twin_of)}
    for h in (None, 2, 3, 4):
        sub = [(o, a) for o, a in pairs if h is None or a["hops"] == h]
        if sub:
            n = len(sub)
            out["all" if h is None else str(h)] = {
                "tasks": n, "original_success": sum(o["success"] for o, _ in sub) / n,
                "original_success_parts": sum(o["success_parts"] for o, _ in sub) / n,
                "original_success_clean": sum(o["success_clean"] for o, _ in sub) / n,
                "adversarial_success_parts": sum(a["success_parts"] for _, a in sub) / n,
                "solved_on_original_only": sum(bool(o["success_parts"]) and not a["success_parts"] for o, a in sub),
                "solved_on_twin_only": sum(bool(a["success_parts"]) and not o["success_parts"] for o, a in sub),
                "original_shortcut_successes": sum(bool(o["success"]) and o["shortcut"] for o, _ in sub)}
    return out


def cycles() -> list[dict]:
    fams, tools = s2.load_practice()
    arm = L.Practice(RUNS / ARM, 0, "practice", fams, tools)
    assert arm.protocol == 2
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
            g = defaultdict(lambda: [0, 0, 0])
            for i in offered:
                v = g[str(key(i))]
                v[0] += 1
                v[1] += i in accepted
                v[2] += any(not x["rejection_v1"] for x in by[i])
            return {n: {"offered": v[0], "accepted": v[1], "yield": v[1] / v[0], "admissible_under_protocol_1": v[2]}
                    for n, v in sorted(g.items())}
        picks = {i: L.choose(by[i]) for i in offered if i in accepted}
        attempts = [x for i in offered for x in by[i]]
        v1_only = [i for i in offered if i not in accepted and any(not x["rejection_v1"] for x in by[i])]
        acc = [i for i in offered if i in accepted]
        lo = (k - 1) * L.CYCLE_ACCEPT
        res = json.loads((RUNS / ARM / f"c{k}/result.json").read_text())
        out.append({
            "cycle": k, "tasks_offered": len(offered), "accepted": len(acc), "yield": len(acc) / len(offered),
            "tasks_admissible_under_protocol_1": sum(any(not x["rejection_v1"] for x in by[i]) for i in offered),
            "tasks_lost_to_protocol_2": len(v1_only),
            "attempts": len(attempts), "attempts_admissible_under_protocol_1": sum(not x["rejection_v1"] for x in attempts),
            "attempts_rejected_for_shortcut": sum("shortcut" in x["rejection"] for x in attempts),
            "attempts_rejected_for_answer_parts": sum("answer_parts" in x["rejection"] for x in attempts),
            "attempts_using_the_shortcut": sum(verify.shortcut(x["trace"]) for x in attempts),
            "by_hops": split(lambda i: fam[i]["hops"]),
            "by_selections": split(lambda i: sum(t.startswith("rule=") for t in task[i].tags)),
            "accepted_strict": sum(p["strict"] for p in picks.values()),
            "buffer_shortcut": sum(verify.shortcut(p["trace"]) for p in picks.values()),
            "buffer_wrong_parts": sum(not verify.parts_ok(task[i], p["trace"]["final"]) for i, p in picks.items()),
            "buffer_uses_two_candidates": sum(verify.enumerates(p["trace"]) for p in picks.values()),
            "replay_failed": c.get("replay_failed", 0),
            "diversity": {"offered": L.diversity([fam[i] for i in offered], [task[i] for i in offered], seed_f),
                          "accepted by practice": L.diversity([fam[i] for i in acc], [task[i] for i in acc], seed_f),
                          "oracle arm, this cycle": L.diversity([arm.stream.family[i] for i in range(lo, lo + L.CYCLE_ACCEPT)],
                                                               [arm.stream.get(i) for i in range(lo, lo + L.CYCLE_ACCEPT)], seed_f)},
            "rollout_seconds": res.get("rollout_seconds_this_job"), "attempts_sampled": res.get("attempts_sampled_this_job"),
            "seconds_per_attempt": res.get("seconds_per_attempt"), "steps": res.get("steps"),
            "train_seconds": res.get("train_seconds"), "seconds_per_step": res.get("seconds_per_step"),
            "dev_seconds": res.get("dev_seconds_this_job"), "adv_seconds": res.get("adv_seconds_this_job"),
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
    false, n, kinds = 0.0, Counter(), Counter()
    for x in a["sample"]:
        key = f"{x['cycle']}:{x['index']}"
        human = verdicts.get(key, {}).get("false_accept")
        must = bool(x["flags"]) or bool(x.get("blind"))
        violates = "shortcut" in x["flags"] or "answer_parts" in x["flags"]  # the rule itself should have kept it out
        n["to_read"] += must
        n["flagged"] += bool(x["flags"])
        n["blind"] += bool(x.get("blind"))
        n["read"] += must and human is not None
        n["false_accepts"] += bool(human) or violates
        n["false_among_blind"] += bool(x.get("blind")) and bool(human)
        n["rule_violations"] += violates
        false += x["weight"] * (bool(human) or violates)
        if human:
            kinds[verdicts[key].get("kind", "?")] += 1
    return {"accepted_episodes": a["accepted_episodes"], "not_strict": a["not_strict"], "sampled": a["sampled"], **n,
            "flags": dict(sorted(Counter(f for x in a["sample"] for f in x["flags"]).items())),
            "false_accepts_by_kind": dict(sorted(kinds.items())), "false_accept_rate": false / total, "limit": 0.02,
            "complete": n["read"] == n["to_read"]}


def books() -> dict:
    launch = load("s2_launch")
    try:
        budget, path = launch.load_budget(BUDGET)
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
    twin_of = {t.task_id: next(x for x in t.tags if x.startswith("twin_of="))[8:] for t in adversarial.adversarial_dev()}
    adapters = {"seed adapter": RUNS / SEED, **{f"after cycle {c['cycle']}": RUNS / ARM / f"c{c['cycle']}" for c in cyc}}
    dev = {k: dev_table(d) for k, d in adapters.items()}
    adv = {k: adversarial_table(d, twin_of) for k, d in adapters.items()}
    first = json.loads((OUT / "stage2.json").read_text())  # the first pilot (protocol 1), for the comparison only
    x = load("s2_e2_rebudget").inputs()
    model = [min(L.CYCLE_ACCEPT / x["yield"], L.CYCLE_TASK_CAP) * L.ATTEMPTS * x["attempt_seconds"]
             + (L.SEED_FAMILIES * L.SEED_INSTANCES + c["cycle"] * L.CYCLE_ACCEPT) / 4 * x["train_seconds_per_step"]
             + 600 * x["seconds_per_episode"]["v2/dev"] for c in cyc]
    for c, m in zip(cyc, model):
        c["gpu_seconds"] = sum(c[k] or 0 for k in ("rollout_seconds", "train_seconds", "dev_seconds", "adv_seconds"))
        c["model_gpu_seconds"], c["cost_ratio_to_model"] = m, c["gpu_seconds"] / m
    series = [dev[k]["success"]["all"] for k in adapters if dev[k]]
    au = audit()
    gates = {"yield": len(cyc) == 2 and all(c["yield"] >= 0.20 for c in cyc),
             "audit": bool(au) and au["complete"] and au["false_accept_rate"] <= au["limit"],
             "cost": bool(cyc) and all(c["cost_ratio_to_model"] <= 2 for c in cyc),
             "dev": len(series) == 3 and all(b >= a - 0.05 for a, b in zip(series, series[1:]))}
    climb = {}
    if len(cyc) == 2:
        for h in ("1", "2", "3", "4"):
            a, b = cyc[0]["by_hops"][h], cyc[1]["by_hops"][h]
            climb[h] = diff_interval((a["accepted"], a["offered"]), (b["accepted"], b["offered"]))
    res = {"protocol": 2, "arm": ARM, "cycles": cyc, "dev": dev, "adversarial": adv, "audit": au, "gates": gates,
           "passed": all(gates.values()), "yield_change_by_hops": climb, "books": books(),
           "buffers_free_of_the_shortcut": all(c["buffer_shortcut"] == 0 and c["buffer_wrong_parts"] == 0 for c in cyc),
           "first_pilot": {"dev": first["dev_by_adapter"], "dev_without_shortcut": [v["all"]["success_without_shortcut"]
                                                                                    for v in first["dev_without_shortcut"].values()],
                           "yield": [c["yield"] for c in first["cycles"]],
                           "accepted_by_shortcut": [c["accepted_by_shortcut"] for c in first["cycles"]],
                           "seconds_per_attempt": [c["seconds_per_attempt"] for c in first["cycles"]]}}
    r6 = [json.loads((d / "result.json").read_text())["seconds_per_episode"] for d in sorted(RUNS.glob("calib3_*"))]
    r4 = [json.loads((d / "result.json").read_text())["seconds_per_episode"] for d in sorted(RUNS.glob("calib_e1_s15_*"))]
    if len(cyc) == 2 and r6 and r4:  # the remaining stages once more, with this pilot and the corrected engine as inputs
        R = load("s2_e2_rebudget")
        gain = (sum(r4) / len(r4)) / (sum(r6) / len(r6))  # greedy scoring, R6 against the engine Stage 1 measured
        y = sum(c["accepted"] for c in cyc) / sum(c["tasks_offered"] for c in cyc)
        att = sum(c["rollout_seconds"] for c in cyc) / sum(c["attempts_sampled"] for c in cyc)
        step = sum(c["train_seconds"] for c in cyc) / sum(c["steps"] for c in cyc)
        x2 = {**x, "train_seconds_per_step": step, "seconds_per_episode": {k: v / gain for k, v in x["seconds_per_episode"].items()},
              "untrained_seconds_per_episode": x["untrained_seconds_per_episode"] / gain}
        keep = lambda v: {k: z for k, z in v.items() if k.endswith("_usd") and "stage1" not in k and not isinstance(z, (list, dict))}  # noqa: E731
        res["rebudget"] = {"yield": y, "attempt_seconds": att, "train_seconds_per_step": step, "greedy_speed_gain_of_r6": gain,
                           "by_the_plans_rules": keep(R.cost(x2, y, att)), "sealed_on_new_engine": keep(R.cost(x2, y, att, True))}
    (OUT / "stage2r.json").write_text(json.dumps(res, indent=1) + "\n")

    ok = lambda b: "met" if b else "**not met**"  # noqa: E731
    md = ["# Experiment 2: the repeated pilot under protocol 2 (generated by scripts/s2_e2_stage2r.py)", "",
          "Replicate 0, `practice`, cycles 1 and 2, on the corrected engine. Acceptance rule of amendment 1: no episode "
          "reached by the shortcut, and no multi-part answer with a fact given for the wrong part, enters a buffer.", "",
          "## Gates", ""]
    md += table(["criterion", "measured", ""], [
        ["in both cycles at least 20% of attempted tasks accepted", " and ".join(pc(c["yield"]) for c in cyc) or "-", ok(gates["yield"])],
        ["verifier audit, as pre-registered: at most 2% false accepts",
         (f"{au['sampled']} audited; {au['flagged']} flagged and {au['blind']} unflagged read ({au['read']} of {au['to_read']} verdicts); "
          f"{au['false_accepts']} false accepts, {pc(au['false_accept_rate'])} weighted") if au else "not run", ok(gates["audit"])],
        ["cost per cycle within twice the model's", " and ".join(f"{c['cost_ratio_to_model']:.2f}x" for c in cyc) or "-", ok(gates["cost"])],
        ["no fall of more than 5 points on `dev` between cycles (original validator)", " → ".join(pc(v) for v in series), ok(gates["dev"])]])
    md += [f"**Pilot {'passed' if res['passed'] else 'did not pass'}.**", ""]
    if cyc:
        md += ["## What protocol 2 kept out", ""]
        md += table(["cycle", "tasks offered", "accepted", "yield", "admissible under protocol 1", "tasks lost to protocol 2",
                     "attempts using the shortcut", "attempts with a wrong part", "in the buffer: shortcut / wrong part"],
                    [[c["cycle"], c["tasks_offered"], c["accepted"], pc(c["yield"]), c["tasks_admissible_under_protocol_1"],
                      c["tasks_lost_to_protocol_2"], f"{c['attempts_using_the_shortcut']} of {c['attempts']}",
                      c["attempts_rejected_for_answer_parts"], f"{c['buffer_shortcut']} / {c['buffer_wrong_parts']}"] for c in cyc])
        md += [f"First pilot (protocol 1): yield {' and '.join(pc(v) for v in res['first_pilot']['yield'])}; "
               f"{' and '.join(str(v) for v in res['first_pilot']['accepted_by_shortcut'])} accepted episodes reached by the shortcut.", ""]
        md += ["## Does practice climb without the shortcut?", "",
               "Yield by number of hops under protocol 2. The two cycles see different tasks of the same stream.", ""]
        md += table(["hops", "cycle 1", *(["cycle 2", "change (95% interval)"] if len(cyc) == 2 else [])],
                    [[h, *[f"{pc(c['by_hops'][h]['yield'])} ({c['by_hops'][h]['accepted']} of {c['by_hops'][h]['offered']})" for c in cyc],
                      *([f"{100 * climb[h]['difference']:+.1f} [{100 * climb[h]['lo']:+.1f}, {100 * climb[h]['hi']:+.1f}]"] if climb else [])]
                     for h in ("1", "2", "3", "4")])
        sels = sorted({k for c in cyc for k in c["by_selections"]})
        md += ["By number of list selections:", ""] + table(
            ["selections", *[f"cycle {c['cycle']}" for c in cyc]],
            [[k, *[f"{pc(c['by_selections'][k]['yield'])} ({c['by_selections'][k]['accepted']} of {c['by_selections'][k]['offered']})"
                   if k in c["by_selections"] else "-" for c in cyc]] for k in sels])
    if any(dev.values()):
        md += ["## `dev`, three ways", "",
               "600 tasks, unchanged. The first row of each adapter is the measure of Experiments 0 and 1 and of the first pilot.", ""]
        rows = []
        for name, d in dev.items():
            for key, label in MEASURES if d else ():
                rows.append([name if key == "success" else "", label, *[pc(d[key][h]) for h in ("all", "1", "2", "3", "4")],
                             (f"{d['shortcut_successes']} of {d['successes']} by shortcut, {d['wrong_part_successes']} with a wrong part; "
                              f"{d['episodes_using_the_shortcut']} of {d['episodes']} episodes show the shortcut" if key == "success" else "")])
        md += table(["adapter", "success", "all", "1 hop", "2", "3", "4", "of the successes"], rows)
        fp = res["first_pilot"]
        md += ["First pilot (protocol 1, engine R1-R4), original validator: " + " → ".join(pc(v) for v in fp["dev"])
               + "; without shortcut successes: " + " → ".join(pc(v) for v in fp["dev_without_shortcut"]) + ".", ""]
    if any(adv.values()):
        md += ["## The adversarial set: complete candidate records", "",
               "350 `dev` tasks in which a call depends on a list selection, rebuilt so that every candidate's continuation "
               "exists. Request, gold path and answer are those of the original, and an agent that applies the selection "
               "rule sees the same episode on both. Success with each fact for its own part, on the twins and on their 350 originals.", ""]
        rows = []
        for name, a in adv.items():
            for h in ("all", "2", "3", "4") if a else ():
                v = a.get(h)
                if v:
                    rows.append([name if h == "all" else "", h if h != "all" else "all", v["tasks"], pc(v["original_success_parts"]),
                                 pc(v["original_success_clean"]), pc(v["adversarial_success_parts"]),
                                 f"{v['solved_on_original_only']} / {v['solved_on_twin_only']}", v["original_shortcut_successes"]])
        md += table(["adapter", "hops", "tasks", "original tasks", "original, no shortcut", "**adversarial twins**",
                     "solved on the original only / on the twin only", "original successes by shortcut"], rows)
    if cyc:
        md += ["## What each cycle's buffer holds (plan §3.3)", ""]
        for c in cyc:
            d = c["diversity"]
            cols = list(d)
            g = lambda col, *ks: (lambda v: f"{v:.2f}" if isinstance(v, float) else v)(  # noqa: E731
                __import__("functools").reduce(lambda a, k: a.get(k, "-") if isinstance(a, dict) else "-", ks, d[col]))
            md += [f"### Cycle {c['cycle']}", ""] + table(["", *cols], [
                ["tasks", *[g(z, "tasks") for z in cols]], ["distinct families", *[g(z, "families") for z in cols]],
                ["tools covered (of 79)", *[g(z, "tools") for z in cols]], ["edges covered (of 229)", *[g(z, "edges") for z in cols]],
                *[[f"share with {h} hops", *[g(z, "hops", str(h)) for z in cols]] for h in (1, 2, 3, 4)],
                ["list selections per task", *[g(z, "mean_selections") for z in cols]],
                ["share deeper than the seed", *[g(z, "outside_reference", "deeper") for z in cols]]])
        md += ["## Accepted episodes", ""] + table(
            ["cycle", "accepted", "strictly successful", "use two candidates of a list (read in the audit)", "failed replay"],
            [[c["cycle"], c["accepted"], f"{c['accepted_strict']} ({pc(c['accepted_strict'] / max(1, c['accepted']))})",
              c["buffer_uses_two_candidates"], c["replay_failed"]] for c in cyc])
    if au:
        md += ["## The audit", "",
               f"{au['accepted_episodes']} accepted episodes, {au['not_strict']} of them not strictly successful; {au['sampled']} "
               f"sampled. Flags: {au['flags']}. Read: {au['flagged']} flagged and {au['blind']} unflagged. False accepts by "
               f"reading or by a failed rule check: {au['false_accepts']} ({au['false_accepts_by_kind']}; among the unflagged: "
               f"{au['false_among_blind']}; episodes in a buffer that fail the rule's own checks: {au['rule_violations']}).", ""]
    if cyc:
        md += ["## Throughput and cost per cycle", ""] + table(
            ["cycle", "tasks offered", "s per attempt", "rollouts (s)", "training (s)", "s per step", "dev + adversarial (s)", "GPU-seconds", "model", "ratio"],
            [[c["cycle"], c["tasks_offered"], c["seconds_per_attempt"], f"{c['rollout_seconds'] or 0:.0f}", f"{c['train_seconds'] or 0:.0f}",
              c["seconds_per_step"], f"{(c['dev_seconds'] or 0) + (c['adv_seconds'] or 0):.0f}", f"{c['gpu_seconds']:.0f}",
              f"{c['model_gpu_seconds']:.0f}", f"{c['cost_ratio_to_model']:.2f}"] for c in cyc])
        md += ["First pilot, engine R1-R4: " + " and ".join(f"{v:.3f}" for v in res["first_pilot"]["seconds_per_attempt"])
               + " s per attempt. The model is the cost model at Stage 1's measurements, which were made on that engine.", ""]
    if "rebudget" in res:
        rb = res["rebudget"]
        md += ["## The main run's cost with this pilot as input (USD; not approved, not run)", "",
               f"Yield {pc(rb['yield'])} over both cycles, {rb['attempt_seconds']:.3f} s per attempt, {rb['train_seconds_per_step']:.2f} s per "
               f"training step, greedy scoring {rb['greedy_speed_gain_of_r6']:.2f} times faster than on the engine Stage 1 costed; otherwise "
               "the cost model of [budget_stage1.md](../../../experiments/study02_e2_signal/budget_stage1.md). The adversarial set and a "
               "second evaluation of sealed splits are not in the model.", ""]
        md += table(["", "Stage 3A", "Stage 3B", "3A + 3B", "extension"], [
            [k, f"{v['stage3_block_a_usd']:.0f}", f"{v['stage3_block_b_usd']:.0f}", f"{v['stage3_block_a_usd'] + v['stage3_block_b_usd']:.0f}",
             f"{v['extension_usd']:.0f}"] for k, v in (("sealed splits on the Experiment 1 scoring path (the plan's rule)", rb["by_the_plans_rules"]),
                                                      ("sealed splits on the corrected engine", rb["sealed_on_new_engine"]))])
    m = res["books"]
    if m:
        md += ["## Money", ""] + table(["job", "flavour", "outcome", "reserved USD", "billed USD"], [
            [k, j.get("flavor"), j.get("stage", "open"), f"{j.get('reserved_usd', 0):.2f}", f"{j['cost_usd']:.2f}" if "cost_usd" in j else "-"]
            for k, j in m["jobs"].items()])
        md += [f"Total: **{m['settled_usd']:.2f} USD** billed; provider-reported above the stage's baseline: "
               f"{m['provider_used_usd']:.2f} USD; open reservations {m['open_reservations_usd']:.2f} USD. Hard cap {m['cap_usd']:.2f} USD.", ""]
    (OUT / "stage2r.md").write_text("\n".join(md))
    print(f"wrote {OUT / 'stage2r.md'}")


if __name__ == "__main__":
    main()
