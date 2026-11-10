"""Experiment 2 planning numbers: cost model and statistical power, from what Experiment 1 measured.

    python scripts/s2_e2_budget.py     # experiments/study02_e2_signal/budget.{json,md}

Nothing here spends money or reads a sealed split. Unit costs come from the committed run records of
Experiment 1 (A10G, the scoring and training code as it ran). Everything that is an assumption and not a
measurement is in `DESIGN` or `ASSUME` below and is listed as such in the output.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/study02/runs/e1"
OUT = ROOT / "experiments/study02_e2_signal"

# The design of experiments/study02_e2_signal/PLAN.md, as counts.
DESIGN = {
    "replicates": 5, "replicates_extended": 8, "control_replicates": 3,
    "seed_episodes": 400, "cycles": 6, "episodes_per_cycle": 480, "task_cap_per_cycle": 2400, "attempts_per_task": 4,
    "episodes_per_step": 8, "passes": 2,
    "static_arm_cycles_trained": (3, 6),  # oracle, teacher: one adapter at mid-run and one at the end
    "splits_final": {"v2/test_comp": 1200, "v2/test_depth": 400, "v2/test_tool": 400, "v2/test_domain": 400,
                     "v2/test_id": 400, "v1": 720},
    "dev": 600, "sharpening_tasks": 300, "sharpening_attempts": 16, "audit_rescoring_arms": 4,
}
# Assumptions (not measured). Each is checked by a stage gate before the money that depends on it is spent.
ASSUME = {
    "usd_per_gpu_hour": 1.00,          # a10g-small, the price Experiment 1 was billed at
    "job_overhead": 1.10,              # start-up, installation, model loading, per-minute rounding
    "contingency": 1.25,               # failed and repeated jobs; Experiment 1 lost 10% of its bill to them
    "sampled_attempt_vs_greedy": 1.25,  # a sampled practice attempt against a greedy episode of the same mix
    "unverified_well_formed_rate": 0.9,
    "teacher_acceptance": 0.6, "teacher_attempts_per_task": 1.5,
    "teacher_usd_per_attempt": (0.0009, 0.0040),  # an efficient engine or hosted endpoint; the E1 loop on a larger GPU
}
ENGINES = {"E1 engine, as measured": 1.0, "Stage 1 target (4x)": 4.0, "stretch (10x)": 10.0}
YIELDS = {"floor allowed by the pilot gate": 0.20, "planning value": 0.40, "optimistic": 0.60}


def run_json(name: str, file: str = "run.json") -> dict:
    return json.loads((RUNS / name / file).read_text())


def measured() -> dict:
    """Seconds per episode by split and seconds per optimizer step for the 1.5B student on an A10G."""
    per: dict = {"s15": defaultdict(list), "s05": defaultdict(list)}
    for d in sorted(RUNS.glob("e1_s*_D*_r*")):
        tag = d.name.split("_")[1]
        for s in run_json(d.name)["scoring"]:
            if s["run_now"] == s["tasks"] and s["seconds"] > 0:  # whole split in one invocation
                per[tag][s["split"]].append(s["seconds"] / s["run_now"])
    s15 = {k: float(np.mean(v)) for k, v in per["s15"].items()}
    s05 = {k: float(np.mean(v)) for k, v in per["s05"].items()}
    shared = [k for k in ("v2/test_comp", "v2/test_id") if k in s15 and k in s05]
    ratio = float(np.mean([s15[k] / s05[k] for k in shared]))
    v1 = [k for k in s15 if k.startswith("v1/")]
    sizes = {"v1/eval_id": 200, "v1/eval_ood": 240, "v1/eval_unseen": 120, "v1/eval_renamed": 160}
    sec = {k: s15[k] for k in ("v2/test_comp", "v2/test_id", "own") if k in s15}
    sec["v1"] = float(sum(s15[k] * sizes[k] for k in v1) / sum(sizes[k] for k in v1))
    for k in ("v2/dev", "v2/test_depth", "v2/test_tool", "v2/test_domain"):  # not scored for the 1.5B student
        sec[k] = s05[k] * ratio
    steps = []
    for d in sorted(RUNS.glob("e1_s15_D*_r*")):
        t = run_json(d.name, "train_run.json")["summary"]
        steps.append(t["train_time_s"] / (t["steps"] - t["resumed_from"]))
    base = [run_json(n)["scoring"] for n in ("e1_s15_fewshot",) if (RUNS / n / "run.json").exists()]
    untrained = float(np.mean([s["seconds"] / s["run_now"] for sc in base for s in sc if s["run_now"]])) if base else None
    return {"seconds_per_episode": sec, "scaled_from_0.5B": ["v2/dev", "v2/test_depth", "v2/test_tool", "v2/test_domain"],
            "ratio_1.5B_to_0.5B": ratio, "train_seconds_per_step": float(np.mean(steps)),
            "untrained_seconds_per_episode": untrained}


def gpu_seconds(m: dict, yield_: float, speed: float) -> dict:
    """GPU-seconds of every unit of work in the design, for one replicate, at a given yield and engine speed-up."""
    D, sec = DESIGN, m["seconds_per_episode"]
    step = m["train_seconds_per_step"]
    per_step = D["episodes_per_step"] / D["passes"]
    steps_at = lambda k: (D["seed_episodes"] + k * D["episodes_per_cycle"]) / per_step  # noqa: E731
    final = sum(n * sec[k] for k, n in D["splits_final"].items()) / speed
    dev = D["dev"] * sec["v2/dev"] / speed
    attempt = sec["v2/test_id"] * ASSUME["sampled_attempt_vs_greedy"] / speed  # the stream has test_id's depth mix
    tasks = min(D["episodes_per_cycle"] / yield_, D["task_cap_per_cycle"])
    cyc = range(1, D["cycles"] + 1)
    practice = {"rollouts": D["cycles"] * tasks * D["attempts_per_task"] * attempt,
                "training": sum(steps_at(k) for k in cyc) * step, "scoring": D["cycles"] * dev + final}
    unverified = {**practice, "rollouts": D["cycles"] * D["episodes_per_cycle"] / ASSUME["unverified_well_formed_rate"]
                  * D["attempts_per_task"] * attempt}
    static = {"rollouts": 0.0, "training": sum(steps_at(k) for k in D["static_arm_cycles_trained"]) * step,
              "scoring": len(D["static_arm_cycles_trained"]) * dev + final}
    matched = {"rollouts": 0.0, "training": steps_at(D["cycles"]) * step, "scoring": dev + final}
    seed = {"rollouts": D["sharpening_tasks"] * D["sharpening_attempts"] * attempt, "training": steps_at(0) * step,
            "scoring": dev + final}
    return {"seed": seed, "oracle": static, "teacher": static, "practice": practice, "practice+curriculum": practice,
            "practice-unverified": unverified, "oracle-matched": matched,
            "one_practice_cycle": {"rollouts": tasks * D["attempts_per_task"] * attempt, "training": steps_at(1) * step,
                                   "scoring": dev},
            "final_scoring": final, "dev_scoring": dev, "attempt": attempt}


def usd(seconds: float) -> float:
    return seconds / 3600 * ASSUME["usd_per_gpu_hour"] * ASSUME["job_overhead"] * ASSUME["contingency"]


def stages(m: dict, yield_: float, speed: float) -> dict:
    """Cost of each stage of the plan. Blocks: A = replicates 0-2 of every arm but the teacher; B = replicates
    3-4 of the main arms; teacher = its five adapters and its demonstrations; extension = replicates 5-7 of the
    main arms, run only if the seed-to-seed spread calls for it."""
    D, g, sec = DESIGN, gpu_seconds(m, yield_, speed), m["seconds_per_episode"]
    tot = lambda a: sum(g[a].values())  # noqa: E731
    main = ["seed", "oracle", "practice", "practice+curriculum"]
    controls = ["practice-unverified", "oracle-matched"]
    # Stage 1 runs on the E1 engine except for the calibration itself, so it is costed at speed 1.
    e1_gap = 6 * sum(n * sec[k] for k, n in (("v2/dev", 600), ("v2/test_depth", 400), ("v2/test_tool", 400),
                                             ("v2/test_domain", 400)))
    baselines = 2 * 720 * (m["untrained_seconds_per_episode"] or 2.1)
    g1 = gpu_seconds(m, yield_, 1.0)
    stage1 = {"engine calibration (two adapters, own-family tasks, both engines)": 4 * 400 * sec["own"] + 1800,
              "complete the 1.5B scoring of Experiment 1 (4 splits x 6 adapters)": e1_gap,
              "v1 anchor for the two untrained 1.5B baselines": baselines,
              "seed adapter of replicate 0, dev, yield probe on 400 tasks": g1["seed"]["training"] + g1["dev_scoring"]
              + 400 * D["attempts_per_task"] * g1["attempt"]}
    pilot = 2 * sum(g["one_practice_cycle"].values()) + g["dev_scoring"]
    audit = D["audit_rescoring_arms"] * D["splits_final"]["v2/test_comp"] * sec["v2/test_comp"] / speed
    block_a = D["control_replicates"] * sum(tot(a) for a in main + controls)
    block_b = (D["replicates"] - D["control_replicates"]) * sum(tot(a) for a in main) + audit
    ext = (D["replicates_extended"] - D["replicates"]) * sum(tot(a) for a in main)
    teacher_attempts = lambda reps: (reps * D["cycles"] * D["episodes_per_cycle"] / ASSUME["teacher_acceptance"]  # noqa: E731
                                     * ASSUME["teacher_attempts_per_task"])
    gen = lambda reps: [teacher_attempts(reps) * c * ASSUME["contingency"] for c in ASSUME["teacher_usd_per_attempt"]]  # noqa: E731
    return {"stage1_usd": {k: usd(v) for k, v in stage1.items()}, "stage1_total_usd": usd(sum(stage1.values())),
            "stage2_pilot_usd": usd(pilot), "stage3_block_a_usd": usd(block_a), "stage3_block_b_usd": usd(block_b),
            "stage4_teacher_adapters_usd": usd(D["replicates"] * tot("teacher")),
            "stage4_teacher_generation_usd": gen(D["replicates"]),
            "stage4_teacher_generation_3_replicates_usd": gen(D["control_replicates"]),
            "extension_usd": usd(ext),
            "total_without_teacher_or_extension_usd": usd(sum(stage1.values()) + pilot + block_a + block_b),
            "per_replicate_usd": {a: {k: usd(v) for k, v in g[a].items()} for a in main + ["teacher"] + controls},
            "practice_attempts": int(D["replicates"] * 2 * D["cycles"]
                                     * min(D["episodes_per_cycle"] / yield_, D["task_cap_per_cycle"]) * D["attempts_per_task"])}


def power() -> dict:
    """Standard error of a difference between two arms on test_comp, from the variance Experiment 1 measured
    on the 1.5B student, and what that detects. Arms are scored on the same 600 families."""
    def cube(d):
        per = []
        for r in range(3):
            fam = defaultdict(dict)
            with gzip.open(RUNS / f"e1_s15_D{d}_r{r}" / "metrics.jsonl.gz", "rt") as f:
                for x in map(json.loads, f):
                    if x["split"] == "v2/test_comp":
                        fam[x["template"]][x["task_id"]] = x["success"]
            per.append(fam)
        fams = sorted(per[0])
        return np.array([[[p[f][t] for t in sorted(p[f])] for f in fams] for p in per])
    lo, hi = cube(25), cube(1600)
    sd_rep = {"D25": float(lo.mean((1, 2)).std(ddof=1)), "D1600": float(hi.mean((1, 2)).std(ddof=1))}
    fam_diff = (hi.mean((0, 2)) - lo.mean((0, 2)))  # per-family difference between two very different arms
    sd_fam = float(fam_diff.std(ddof=1))
    t975 = {8: 2.306, 14: 2.145}
    t80 = {8: 0.889, 14: 0.868}
    t95 = {8: 1.860, 14: 1.761}
    rows = []
    for sd in (0.02, 0.03, 0.05):
        for n in (5, 8):
            df = 2 * (n - 1)
            se = float(np.sqrt(2 * sd ** 2 / n + sd_fam ** 2 / lo.shape[1]))
            rows.append({"seed_sd": sd, "replicates_per_arm": n, "se_points": 100 * se,
                         "detectable_difference_points": 100 * (t975[df] + t80[df]) * se,
                         "non_inferiority_margin_7.5_has_80pct_power": bool((t95[df] + t80[df]) * se <= 0.075)})
    return {"replicate_sd_measured_1.5B": sd_rep, "replicates_behind_that_estimate": 3,
            "family_sd_of_arm_difference_upper_bound": sd_fam, "table": rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = measured()
    grid = {e: {y: stages(m, YIELDS[y], ENGINES[e]) for y in YIELDS} for e in ENGINES}
    pw = power()
    res = {"design": DESIGN, "assumptions": ASSUME, "measured_in_experiment_1": m, "scenarios": grid, "power": pw}
    (OUT / "budget.json").write_text(json.dumps(res, indent=1) + "\n")

    sec = m["seconds_per_episode"]
    md = ["# Experiment 2: cost model and power (generated by scripts/s2_e2_budget.py)", "",
          "No money is spent by this script. Unit costs are those Experiment 1 was billed at; the scenarios bracket "
          "what is not yet measured.", "",
          "## Measured in Experiment 1 (1.5B student, A10G at 1.00 USD per hour)", "",
          "| quantity | value | source |", "|---|---|---|",
          f"| training | {m['train_seconds_per_step']:.2f} s per step of 8 episodes (gradient checkpointing) | six runs |"]
    for k in ("v2/test_comp", "v2/test_id", "own", "v1"):
        md.append(f"| greedy scoring, `{k}` | {sec[k]:.2f} s per episode | six runs |")
    for k in m["scaled_from_0.5B"]:
        md.append(f"| greedy scoring, `{k}` | {sec[k]:.2f} s per episode | 0.5B time x {m['ratio_1.5B_to_0.5B']:.2f} "
                  f"(the 1.5B/0.5B ratio on `test_comp` and `test_id`) |")
    md += [f"| untrained 1.5B model, few-shot | {m['untrained_seconds_per_episode']:.2f} s per episode | one run |", "",
           "## Assumptions, and the gate that checks each", "",
           "| assumption | value | checked at |", "|---|---|---|",
           f"| a sampled practice attempt costs this many greedy `test_id` episodes | {ASSUME['sampled_attempt_vs_greedy']} | Stage 1 yield probe |",
           "| engine speed-up over Experiment 1 | 1x, 4x or 10x | Stage 1 calibration |",
           "| yield (accepted tasks per attempted task) | 20%, 40% or 60% | Stage 1 probe, Stage 2 pilot |",
           f"| job overhead; contingency for failed jobs | x{ASSUME['job_overhead']}; x{ASSUME['contingency']} | ledger, every stage |",
           f"| teacher: acceptance, attempts per task | {ASSUME['teacher_acceptance']:.0%}, {ASSUME['teacher_attempts_per_task']} | Stage 1 qualification |",
           f"| teacher: cost per attempt | {ASSUME['teacher_usd_per_attempt'][0]} to {ASSUME['teacher_usd_per_attempt'][1]} USD | Stage 1 qualification |",
           "", "## Cost by stage (USD, GPU only, including overhead and contingency)", "",
           "Block A: replicates 0-2 of seed, oracle, practice, practice+curriculum, practice-unverified and "
           "oracle-matched. Block B: replicates 3-4 of seed, oracle, practice and practice+curriculum, and one repeat "
           "scoring per main arm. Extension: replicates 5-7 of those four arms.", "",
           "| engine | yield | Stage 1 | Stage 2 pilot | Stage 3, block A | Stage 3, block B | **Stages 1-3** | extension | "
           "practice attempts, blocks A and B |", "|---|---|---|---|---|---|---|---|---|"]
    for e in ENGINES:
        for y in YIELDS:
            s = grid[e][y]
            md.append(f"| {e} | {YIELDS[y]:.0%} ({y}) | {s['stage1_total_usd']:.0f} | {s['stage2_pilot_usd']:.0f} | "
                      f"{s['stage3_block_a_usd']:.0f} | {s['stage3_block_b_usd']:.0f} | "
                      f"**{s['total_without_teacher_or_extension_usd']:.0f}** | {s['extension_usd']:.0f} | "
                      f"{s['practice_attempts']:,} |")
    ref = grid["Stage 1 target (4x)"]["planning value"]
    worst = grid["Stage 1 target (4x)"]["floor allowed by the pilot gate"]
    md += ["", f"Stage 4, the teacher arm: five adapters trained and scored, {ref['stage4_teacher_adapters_usd']:.0f} USD at "
               f"the Stage 1 target; demonstrations, which do not depend on the student's engine or yield, "
               f"{ref['stage4_teacher_generation_usd'][0]:.0f} to {ref['stage4_teacher_generation_usd'][1]:.0f} USD for five "
               f"replicates or {ref['stage4_teacher_generation_3_replicates_usd'][0]:.0f} to "
               f"{ref['stage4_teacher_generation_3_replicates_usd'][1]:.0f} USD for three.", "",
           "### Stage 1 in detail (run on the Experiment 1 engine, apart from the calibration itself)", "",
           "| item | USD |", "|---|---|"] + [f"| {k} | {v:.2f} |" for k, v in ref["stage1_usd"].items()]
    md += ["", "### One replicate at the Stage 1 target and the planning yield (USD)", "",
           "| arm | rollouts | training | scoring | total |", "|---|---|---|---|---|"]
    for a, v in ref["per_replicate_usd"].items():
        md.append(f"| {a} | {v['rollouts']:.2f} | {v['training']:.2f} | {v['scoring']:.2f} | {sum(v.values()):.2f} |")
    md += ["", "## Power on `test_comp` (600 families, two tasks each)", "",
           f"Measured on the 1.5B student in Experiment 1: standard deviation of replicate means "
           f"{pw['replicate_sd_measured_1.5B']['D25']:.3f} at 25 families and {pw['replicate_sd_measured_1.5B']['D1600']:.3f} "
           f"at 1,600, from three replicates each (a 95% interval for a standard deviation estimated from three values "
           f"runs from about half to six times the estimate). Per-family standard deviation of the observed difference "
           f"between those two arms, replicate and task noise included: {pw['family_sd_of_arm_difference_upper_bound']:.3f}, "
           f"used as an upper bound for arms that differ less.", "",
           "| seed sd | replicates per arm | SE of an arm difference (points) | difference detected with 80% power | "
           "non-inferiority at 7.5 points has 80% power |", "|---|---|---|---|---|"]
    for r in pw["table"]:
        md.append(f"| {r['seed_sd']:.2f} | {r['replicates_per_arm']} | {r['se_points']:.1f} | "
                  f"{r['detectable_difference_points']:.1f} | {'yes' if r['non_inferiority_margin_7.5_has_80pct_power'] else 'no'} |")
    md += ["", "Two-sided 5% tests with Student-t quantiles on 2(n - 1) degrees of freedom; one-sided 5% for "
               "non-inferiority at a true difference of zero.", ""]
    (OUT / "budget.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(json.dumps({"reference": {k: v for k, v in ref.items() if k != "per_replicate_usd"},
                      "floor_yield_at_target": {k: v for k, v in worst.items() if k.startswith(("stage2", "stage3", "total"))}},
                     indent=1))


if __name__ == "__main__":
    main()
