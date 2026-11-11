"""Study 02, Experiment 2: the record that amendment 2 asks for before Stage 3 (no model, no test split).

    python scripts/s2_e2_stage3.py      # writes results/study02/e2/stage3_prep.{json,md}

It reads committed results only: the engine calibrations, the two pilots' attempts, the repeated pilot's
audit, the manifest of complete-world twins (counts and hashes; no task), the teacher re-qualification, and
the cost model. The only evaluation tasks it opens are `dev` and its twins, for the selection table.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccl import s2  # noqa: E402

RUNS = ROOT / "results/study02/runs/e2"
OUT = ROOT / "results/study02/e2"
ALPHA = 0.05


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def upper(k: int, n: int, alpha: float = ALPHA) -> float:
    """One-sided exact (Clopper-Pearson) upper bound for a share, k found in n looked at."""
    if n == 0:
        return 1.0
    if k == 0:
        return 1 - alpha ** (1 / n)
    from math import comb
    lo, hi = k / n, 1.0
    for _ in range(60):  # the p at which seeing k or fewer has probability alpha
        p = (lo + hi) / 2
        if sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1)) > alpha:
            lo = p
        else:
            hi = p
    return hi


def reads_needed(limit: float, found: int = 0, alpha: float = ALPHA) -> int:
    n = found + 1
    while upper(found, n, alpha) > limit:
        n += 1
    return n


# ------------------------------------------------------------------ the engine

def metrics(d: str, split: str | None = None) -> dict:
    return {r["task_id"]: r for r in map(json.loads, open(RUNS / d / "metrics.jsonl")) if split is None or r["split"] == split}


def attempts(arm: str, cycle: int) -> dict:
    return {(r["index"], r["attempt"]): r for f in sorted((RUNS / arm / "attempts").glob(f"c{cycle}_*.jsonl.gz"))
            for r in s2.read_gz(f)}


def engine() -> dict:
    """What is known about the corrected engine (R1-R4 with R6 in float32) against the two it replaces."""
    calib = {}
    for adapter in ("e1_s15_D1600_r0", "e1_s15_D25_r0"):
        a, b = metrics(f"calib_{adapter}"), metrics(f"calib3_{adapter}")
        ra, rb = (json.loads((RUNS / f"{p}_{adapter}" / "result.json").read_text()) for p in ("calib", "calib3"))
        calib[adapter] = {
            "episodes": len(a), "experiment_1_success": ra["success_experiment_1"],
            "r1_r4": {"success": ra["success"], "changed_against_experiment_1": ra["changed_outcome"],
                      "same_text_as_full_re_encoding": ra["same_text_as_full_re_encoding"],
                      "seconds_per_episode": ra["seconds_per_episode"]},
            "r6": {"success": rb["success"], "changed_against_experiment_1": rb["changed_outcome"],
                   "same_text_as_full_re_encoding": rb["same_text_as_full_re_encoding"],
                   "seconds_per_episode": rb["seconds_per_episode"]},
            "r6_against_r1_r4": {"outcome_differs": sum(a[k]["success"] != b[k]["success"] for k in a),
                                 "length_or_calls_differ": sum((a[k]["gen_tokens"], a[k]["n_calls"]) != (b[k]["gen_tokens"], b[k]["n_calls"]) for k in a)}}
    a, b = metrics("r0_seed", "v2/dev"), metrics("r0_seed_p2", "v2/dev")
    dev = {"tasks": len(a), "r1_r4_success": sum(r["success"] for r in a.values()) / len(a),
           "r6_success": sum(r["success"] for r in b.values()) / len(b),
           "fail_to_success": sum(bool(not a[k]["success"] and b[k]["success"]) for k in a),
           "success_to_fail": sum(bool(a[k]["success"] and not b[k]["success"]) for k in a),
           "length_or_calls_differ": sum((a[k]["gen_tokens"], a[k]["n_calls"]) != (b[k]["gen_tokens"], b[k]["n_calls"]) for k in a)}
    p, q = attempts("r0_practice", 1), attempts("r0_practice_p2", 1)
    both = sorted(set(p) & set(q))
    text = lambda r: [t["raw"] for t in r["trace"]["turns"]]  # noqa: E731
    sampled = {"attempts_with_the_same_task_seed_and_attempt_number": len(both),
               "identical_text": sum(text(p[k]) == text(q[k]) for k in both),
               "outcome_differs": sum(p[k]["success"] != q[k]["success"] for k in both),
               "r1_r4_success": sum(p[k]["success"] for k in both) / len(both),
               "r6_success": sum(q[k]["success"] for k in both) / len(both)}
    bad = json.loads((RUNS / "calib2_e1_s15_D1600_r0/result.json").read_text())
    return {"calibration_greedy_own_family_tasks": calib, "seed_adapter_on_dev_greedy": dev,
            "cycle_1_sampled_attempts": sampled,
            "first_version_of_r6_rejected": {"changed_against_experiment_1": bad["changed_outcome"], "episodes": bad["episodes"],
                                             "same_text_as_full_re_encoding": bad["same_text_as_full_re_encoding"],
                                             "cause": "attention scores rounded to bfloat16 before the softmax; not visible on a CPU, "
                                                      "where the tests run in float32"},
            "not_measured": "the corrected engine against itself (the same adapter and tasks scored twice); Experiment 0 "
                            "measured 0 of 720 for the Experiment 1 path"}


# ------------------------------------------------------------------ the audit's denominators

def audit() -> dict:
    a = json.loads((OUT / "stage2r.json").read_text())["audit"]
    N, sample, nonstrict = a["accepted_episodes"], a["sampled"], a["not_strict"]
    strict_pop, strict_sample = N - nonstrict, sample - nonstrict
    assert a["flagged"] == nonstrict == a["flags"].get("calls_differ_from_the_oracle"), "the flagged are the not strictly successful"
    w = strict_pop / N
    return {
        "accepted_episodes": N, "sampled_and_checked_by_program": sample, "read": a["read"],
        "read_because_flagged": a["flagged"], "read_blind": a["blind"], "checked_by_program_only": sample - a["read"],
        "false_accepts_found": a["false_accepts"],
        "strata": {"not_strictly_successful": {"population": nonstrict, "sampled": nonstrict, "read": nonstrict, "false": 0,
                                               "note": "a census: every one was sampled, flagged and read"},
                   "strictly_successful": {"population": strict_pop, "sampled": strict_sample, "flagged_by_program": 0,
                                           "read": a["blind"], "false": a["false_among_blind"]}},
        "what_the_program_establishes": "for each of the unflagged episodes: its calls are the oracle's calls, its answer "
                                        "states no value the reference answer lacks, it uses no identifier it had not "
                                        "observed and it passes the rule's own checks",
        "upper_bound_95_one_sided": {
            "if_the_programs_check_is_relied_on": {"strict_stratum": upper(0, strict_sample),
                                                   "all_accepted": w * upper(0, strict_sample)},
            "from_reading_alone": {"strict_stratum": upper(0, a["blind"]), "all_accepted": w * upper(0, a["blind"])}},
        "reads_needed_for_a_2_percent_bound_from_reading_alone": {"none_found": reads_needed(0.02), "one_found": reads_needed(0.02, 1)},
        "first_pilot": (lambda f: {"sampled_and_checked_by_program": f["sampled"], "read": f["read"],
                                   "false_accepts": f["false_by_reading"], "weighted_rate": f["false_accept_rate_by_reading"]})(
            json.loads((OUT / "stage2.json").read_text())["audit"]),
        "reader": "the assistant that ran the pilot, with a written verdict per episode; no person and no second reader"}


# ------------------------------------------------------------------ the teacher

def teacher() -> dict:
    out = {}
    for d in sorted(RUNS.glob("teacher_*")):
        r = json.loads((d / "result.json").read_text())
        row = {"repo": r["model"]["repo"], "revision": r["model"]["revision"], "protocol_of_the_run": r.get("protocol", 1),
               "complete": r["complete"]}
        rows = list(s2.read_gz(d / "attempts.jsonl.gz"))
        hop = {}
        for x in rows:
            hop.setdefault(x["task_id"], x)
        e2 = load("s2_e2")
        idx, tasks = e2.teacher_tasks()
        depth = {t.task_id: t.depth for t in tasks}
        by = {t.task_id: t for t in tasks}
        from ccl.runtime import Episode
        ok = {1: set(), 2: set()}
        why = {"shortcut": 0, "answer_parts": 0}
        for x in rows:
            ep = Episode(by[x["task_id"]], "replay", fewshot=True)
            for t in x["trace"]["turns"]:
                ep.step(t["raw"])
            for proto in (1, 2):
                rej = s2.rejection(by[x["task_id"]], ep, protocol=proto)
                if not rej:
                    ok[proto].add(x["task_id"])
                elif proto == 2:
                    for k in why:
                        why[k] += k in rej
        for proto in (1, 2):
            four = [t for t in depth if depth[t] == 4]
            row[f"protocol_{proto}"] = {"accepted": len(ok[proto]), "acceptance": len(ok[proto]) / len(tasks),
                                        "by_hops": {str(h): sum(t in ok[proto] for t in depth if depth[t] == h) /
                                                    sum(1 for t in depth if depth[t] == h) for h in (1, 2, 3, 4)},
                                        "four_hop": sum(t in ok[proto] for t in four) / len(four)}
        row["attempts"] = len(rows)
        row["attempts_rejected_under_protocol_2_for"] = why
        p2 = row["protocol_2"]
        row["qualifies_under_protocol_2"] = bool(r["complete"] and r.get("protocol", 1) == 2 and p2["acceptance"] >= 0.60 and p2["four_hop"] >= 0.40)
        row["note"] = None if r.get("protocol", 1) == 2 else "collected under protocol 1: tasks solved by the shortcut got no retry"
        out[d.name] = row
    return out


# ------------------------------------------------------------------ cost

def cost() -> dict:
    """Stage 3 at the repeated pilot's measurements, with what amendment 2 adds: twins of the test tasks for
    every final adapter and the adversarial `dev` set at every `dev` scoring."""
    R, B = load("s2_e2_rebudget"), load("s2_e2_budget")
    rb = json.loads((OUT / "stage2r.json").read_text())["rebudget"]
    x = R.inputs()
    sec = {k: v / rb["greedy_speed_gain_of_r6"] for k, v in x["seconds_per_episode"].items()}
    tw = json.loads((ROOT / "configs/study02/e2/twins.json").read_text())["splits"]
    c = [json.loads((RUNS / "r0_practice_p2" / f"c{k}" / "result.json").read_text()) for k in (1, 2)]
    ratio = max((r["adv_seconds_this_job"] / r["adv_tasks"]) / (r["dev_seconds_this_job"] / r["dev_tasks"]) for r in c)
    old = B.ASSUME["job_overhead"]
    B.ASSUME["job_overhead"] = x["job_overhead"]
    try:
        final_twins = sum(v["twins"] * sec[k[:-4]] * ratio for k, v in tw.items() if k != "v2/dev_adv")
        dev_twins = tw["v2/dev_adv"]["twins"] * sec["v2/dev"] * ratio
        finals = {"a": 3 * 6, "b": 2 * 4}  # final adapters: seed, oracle, practice, +curriculum, unverified, matched; then the four main arms
        devs = {"a": 3 * (6 + 6 + 6 + 2 + 1 + 1), "b": 2 * (6 + 6 + 2 + 1)}
        extra = {k: B.usd(finals[k] * final_twins + devs[k] * dev_twins) for k in "ab"}
        repeat = B.usd(2 * 600 * sec["v2/dev"] + 300)
    finally:
        B.ASSUME["job_overhead"] = old
    out = {}
    for name, base in (("test splits on the corrected engine", rb["sealed_on_new_engine"]),
                       ("test splits on the Experiment 1 path (the plan's rule while the engine is not approved)", rb["by_the_plans_rules"])):
        a, b = base["stage3_block_a_usd"], base["stage3_block_b_usd"]
        scale = 1.0 if "corrected" in name else rb["greedy_speed_gain_of_r6"] * x["e1"]["seconds_per_episode"]["v2/test_comp"] / x["seconds_per_episode"]["v2/test_comp"]
        ea, eb = extra["a"] * scale, extra["b"] * scale
        out[name] = {"block_a_usd": a, "block_b_usd": b, "twins_block_a_usd": ea, "twins_block_b_usd": eb,
                     "block_a_with_twins_usd": a + ea, "block_b_with_twins_usd": b + eb, "both_usd": a + b + ea + eb}
    return {"inputs": {"yield": rb["yield"], "seconds_per_attempt": rb["attempt_seconds"], "train_seconds_per_step": rb["train_seconds_per_step"],
                       "twin_episode_against_original": ratio, "twin_episodes_per_final_adapter": sum(v["twins"] for k, v in tw.items() if k != "v2/dev_adv"),
                       "test_episodes_per_final_adapter": sum(v["tasks_in_split"] for k, v in tw.items() if k != "v2/dev_adv"),
                       "final_adapters": finals, "dev_scorings": devs},
            "scenarios": out, "engine_repeat_check_usd": repeat,
            "contingency_in_every_figure": 1.25}


def selection() -> dict:
    """First picks at list selections against picking at random (review of PR #3). Rollouts are committed; the
    `dev` traces are in the bucket (`scripts/s2_launch.py pull e2 --traces`), and without them the numbers
    of the last run that had them are kept."""
    from ccl import adversarial, loop
    fams, tools = s2.load_practice()
    stream = loop.Stream(fams, tools, 0)

    def tally(pairs):
        pairs = list(pairs)
        return {"selections": len(pairs), "gold_first_pick": sum(a for a, _ in pairs) / len(pairs), "chance": sum(c for _, c in pairs) / len(pairs)}
    out = {"rollouts": {}, "greedy": {}}
    for arm in ("r0_practice", "r0_practice_p2"):
        for k in (1, 2):
            out["rollouts"][f"{arm} cycle {k}"] = tally(p for r in attempts(arm, k).values() for p in adversarial.selection_picks(stream.get(r["index"]), r["trace"]))
    adapters = {"seed adapter": "r0_seed_p2", "after cycle 1": "r0_practice_p2/c1", "after cycle 2": "r0_practice_p2/c2"}
    if all((RUNS / d / "traces.jsonl.gz").exists() for d in adapters.values()):
        sets = {"v2/dev": {t.task_id: t for t in adversarial.s2.load_split("dev")}, "v2/dev_adv": {t.task_id: t for t in adversarial.adversarial_dev(version=1)}}
        for name, d in adapters.items():
            rows = [x["trace"] for x in s2.read_gz(RUNS / d / "traces.jsonl.gz")]
            for split, tasks in sets.items():
                out["greedy"][f"{name}, {split}"] = tally(p for t in rows if t["task_id"] in tasks for p in adversarial.selection_picks(tasks[t["task_id"]], t))
    elif (OUT / "stage3_prep.json").exists():
        out["greedy"] = json.loads((OUT / "stage3_prep.json").read_text()).get("selection", {}).get("greedy", {})
    return out


def main() -> None:
    res = {"engine": engine(), "audit": audit(), "twins": json.loads((ROOT / "configs/study02/e2/twins.json").read_text())["splits"],
           "teacher": teacher(), "cost": cost(), "selection": selection()}
    (OUT / "stage3_prep.json").write_text(json.dumps(res, indent=1) + "\n")
    e, a, c = res["engine"], res["audit"], res["cost"]
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    md = ["# Experiment 2: record for Stage 3 (generated by scripts/s2_e2_stage3.py)", "",
          "No model was run for this file except the teacher re-qualification it reports. No test split was scored.", "",
          "## 1. The corrected engine", "",
          "| greedy, 400 own-family tasks | success | outcomes changed against Experiment 1 | same text as full re-encoding | s per episode |",
          "|---|---|---|---|---|"]
    for k, v in e["calibration_greedy_own_family_tasks"].items():
        md.append(f"| {k}, Experiment 1 path | {pct(v['experiment_1_success'])} | — | — | 1.62 |")
        for name, key in (("R1-R4", "r1_r4"), ("R1-R4 and R6 (corrected)", "r6")):
            r = v[key]
            md.append(f"| {k}, {name} | {pct(r['success'])} | {r['changed_against_experiment_1']} | {r['same_text_as_full_re_encoding']} | {r['seconds_per_episode']:.2f} |")
        d = v["r6_against_r1_r4"]
        md.append(f"| {k}, R6 against R1-R4 | | {d['outcome_differs']} outcomes differ, {d['length_or_calls_differ']} episodes differ in length or calls | | |")
    d, s = e["seed_adapter_on_dev_greedy"], e["cycle_1_sampled_attempts"]
    md += ["", f"Seed adapter on `dev` ({d['tasks']} tasks, greedy): {pct(d['r1_r4_success'])} on R1-R4, {pct(d['r6_success'])} on R6; "
           f"{d['fail_to_success']} tasks changed from failure to success and {d['success_to_fail']} the other way; "
           f"{d['length_or_calls_differ']} episodes differ in length or calls.", "",
           f"Sampled attempts of cycle 1 (same task, seed and attempt number on both engines): {s['attempts_with_the_same_task_seed_and_attempt_number']:,} "
           f"pairs, {s['identical_text']:,} with identical text ({pct(s['identical_text'] / s['attempts_with_the_same_task_seed_and_attempt_number'])}), "
           f"{s['outcome_differs']} with a different outcome; success {pct(s['r1_r4_success'])} and {pct(s['r6_success'])}.", "",
           f"The first version of R6 changed {e['first_version_of_r6_rejected']['changed_against_experiment_1']} of "
           f"{e['first_version_of_r6_rejected']['episodes']} outcomes and was rejected by the calibration guard. "
           f"Cause: {e['first_version_of_r6_rejected']['cause']}.", "", f"Not measured: {e['not_measured']}.", "",
           "## 2. The repeated pilot's audit: what was checked by program and what was read", "",
           "| | episodes |", "|---|---|",
           f"| accepted in the two cycles | {a['accepted_episodes']} |",
           f"| sampled, each checked by program | {a['sampled_and_checked_by_program']} |",
           f"| of these, read: flagged by the program | {a['read_because_flagged']} |",
           f"| of these, read: unflagged, drawn blind | {a['read_blind']} |",
           f"| checked by program only, not read | {a['checked_by_program_only']} |",
           f"| false accepts found | {a['false_accepts_found']} |", "",
           f"The {a['read_because_flagged']} flagged episodes are all {a['strata']['not_strictly_successful']['population']} accepted episodes that are not strictly "
           "successful: that stratum was read in full. Of the "
           f"{a['strata']['strictly_successful']['population']} strictly successful ones, {a['strata']['strictly_successful']['sampled']} were sampled, "
           f"none was flagged, and {a['read_blind']} were read.", "",
           "| one-sided 95% upper bound on the share of false accepts | strictly successful stratum | all accepted |", "|---|---|---|"]
    for k, v in a["upper_bound_95_one_sided"].items():
        md.append(f"| {k.replace('_', ' ')} | {pct(v['strict_stratum'])} | {pct(v['all_accepted'])} |")
    md += ["", f"The program establishes, {a['what_the_program_establishes']}. Reading alone supports the 2% limit only with "
           f"{a['reads_needed_for_a_2_percent_bound_from_reading_alone']['none_found']} episodes read and none found false "
           f"({a['reads_needed_for_a_2_percent_bound_from_reading_alone']['one_found']} if one is found). Reader: {a['reader']}.", "",
           "## 3. Complete-world twins", "", "| split | tasks | twins | by hops | SHA-256 |", "|---|---|---|---|---|"]
    for k, v in res["twins"].items():
        md.append(f"| `{k}` | {v['tasks_in_split']} | {v['twins']} | {', '.join(f'{h}: {n}' for h, n in v['hops'].items())} | `{v['sha256'][:16]}…` |")
    md += ["", "For every twin the oracle's transcript is the one it has on the original (checked when the manifest was built; "
           "no model was run and no task was printed).", "", "## 4. Teacher qualification under protocol 2", "",
           "| model | run under | attempts | protocol 1: all, four hops | protocol 2: all, four hops | rejected for shortcut, wrong part | qualifies (60%, 40%) |",
           "|---|---|---|---|---|---|---|"]
    for k, v in res["teacher"].items():
        w = v["attempts_rejected_under_protocol_2_for"]
        md.append(f"| `{v['repo']}` | protocol {v['protocol_of_the_run']} | {v['attempts']} | {pct(v['protocol_1']['acceptance'])}, {pct(v['protocol_1']['four_hop'])} | "
                  f"{pct(v['protocol_2']['acceptance'])}, {pct(v['protocol_2']['four_hop'])} | {w['shortcut']}, {w['answer_parts']} | "
                  f"{'**yes**' if v['qualifies_under_protocol_2'] else 'no'}{'' if not v['note'] else ' (' + v['note'] + ')'} |")
    md += ["", "## 5. Cost of Stage 3 at the repeated pilot's measurements", "",
           f"Yield {pct(c['inputs']['yield'])}, {c['inputs']['seconds_per_attempt']:.2f} s per attempt, {c['inputs']['train_seconds_per_step']:.2f} s per step; "
           f"a twin episode takes {c['inputs']['twin_episode_against_original']:.2f} times an original one (measured on `dev`). "
           f"Per final adapter: {c['inputs']['test_episodes_per_final_adapter']:,} test episodes and {c['inputs']['twin_episodes_per_final_adapter']:,} twin episodes. "
           "Every figure includes job overhead and a 25% contingency.", "",
           "| | 3A | 3B | twins in 3A | twins in 3B | 3A with twins | 3B with twins | both |", "|---|---|---|---|---|---|---|---|"]
    for k, v in c["scenarios"].items():
        md.append(f"| {k} | {v['block_a_usd']:.0f} | {v['block_b_usd']:.0f} | {v['twins_block_a_usd']:.1f} | {v['twins_block_b_usd']:.1f} | "
                  f"**{v['block_a_with_twins_usd']:.0f}** | **{v['block_b_with_twins_usd']:.0f}** | {v['both_usd']:.0f} |")
    md += ["", f"A repeat of the corrected engine against itself (seed adapter, `dev`, twice): about {c['engine_repeat_check_usd']:.2f} USD.", ""]
    md += ["## 6. Does practice improve the selection itself?", "",
           "At a list of the gold path whose pick a later call uses: the share of first picks that take the gold row, against "
           "picking at random (the oracle: 100%).", "", "| | selections | gold row picked first | at random |", "|---|---|---|---|"]
    for group in ("rollouts", "greedy"):
        for k, v in res["selection"][group].items():
            md.append(f"| {'sampled attempts, ' if group == 'rollouts' else ''}{k} | {v['selections']:,} | {pct(v['gold_first_pick'])} | {pct(v['chance'])} |")
    md += [""]
    (OUT / "stage3_prep.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
