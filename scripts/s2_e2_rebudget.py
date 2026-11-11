"""Experiment 2: the remaining budget, recalculated from what Stage 1 measured.

    python scripts/s2_e2_rebudget.py     # experiments/study02_e2_signal/budget_stage1.{json,md}

The cost model is the one of scripts/s2_e2_budget.py, unchanged (same design counts, same formulas). What
changes is its inputs: wherever the plan had a projected speed-up or an assumed ratio, this script puts the
Stage 1 measurement (results/study02/e2/stage1.json). The plan's own table, budget.md, is left as it was.
Nothing here spends money.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/study02_e2_signal"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def inputs() -> dict:
    """Unit costs for the model: Experiment 1's, replaced by Stage 1's wherever Stage 1 measured them."""
    B = load("s2_e2_budget")
    S = json.loads((ROOT / "results/study02/e2/stage1.json").read_text())
    m = B.measured()
    speed = S["engine"]["speed_up"]  # greedy scoring, own-family tasks, both calibration adapters
    sec = {k: v / speed for k, v in m["seconds_per_episode"].items()}
    step = S["bench"]["settings"]["always"]["seconds_per_step"]
    jobs = S["books"]["jobs"]
    j = jobs["e2s1_dev_probe"]  # the one job whose useful seconds are all on record
    useful = S["seed"]["dev_seconds"] + S["probe"]["rollout_seconds"]
    overhead = (j["cost_usd"] / j["price_per_min"] * 60) / useful
    failed = sum(x.get("cost_usd", 0) for x in jobs.values() if x.get("stage") == "ERROR")
    return {"B": B, "S": S, "e1": m, "speed_up": speed, "seconds_per_episode": sec, "train_seconds_per_step": step,
            "attempt_seconds": S["probe"]["seconds_per_attempt"], "yield": S["probe"]["yield"],
            "job_overhead": overhead, "failed_share_of_stage1": failed / S["books"]["settled_usd"],
            "untrained_seconds_per_episode": m["untrained_seconds_per_episode"] / speed}


def cost(x: dict, yield_: float, attempt_seconds: float, sealed_on_new_engine: bool = False) -> dict:
    """The model's stages at these inputs. The model takes the engine as a speed-up on Experiment 1's times and
    a sampled attempt as a multiple of a greedy `test_id` episode; both are set so that it uses the measured
    seconds as they are.

    Sealed splits and the v1 anchor are costed on the Experiment 1 scoring path unless `sealed_on_new_engine`:
    the plan lets them use the new engine only if its calibration passes, and the calibration missed its speed
    criterion. `dev`, the practice rollouts and everything else use the new engine."""
    B = x["B"]
    sec = dict(x["seconds_per_episode"])
    if not sealed_on_new_engine:
        sec.update({k: v for k, v in x["e1"]["seconds_per_episode"].items() if k != "v2/dev" and k != "own"})
    m = {**x["e1"], "seconds_per_episode": sec, "train_seconds_per_step": x["train_seconds_per_step"],
         "untrained_seconds_per_episode": x["untrained_seconds_per_episode"]}
    old = dict(B.ASSUME)
    B.ASSUME.update(sampled_attempt_vs_greedy=attempt_seconds / sec["v2/test_id"], job_overhead=x["job_overhead"])
    try:
        s = B.stages(m, yield_, 1.0)
        s["remaining_usd"] = s["stage2_pilot_usd"] + s["stage3_block_a_usd"] + s["stage3_block_b_usd"]
        return s
    finally:
        B.ASSUME.clear()
        B.ASSUME.update(old)


def deferred_scoring(x: dict) -> dict:
    """Item (b) of Stage 1, approved for planning only: the 1.5B scoring Experiment 1 left out."""
    B, sec = x["B"], x["seconds_per_episode"]
    old = B.ASSUME["job_overhead"]
    B.ASSUME["job_overhead"] = x["job_overhead"]
    try:
        splits = (("v2/dev", 600), ("v2/test_depth", 400), ("v2/test_tool", 400), ("v2/test_domain", 400))
        adapters = 6 * sum(n * sec[k] for k, n in splits)
        baselines = 2 * 720 * x["untrained_seconds_per_episode"]
        e1 = x["e1"]
        return {"episodes_six_adapters": 6 * sum(n for _, n in splits), "episodes_two_baselines": 2 * 720,
                "six_adapters_usd": B.usd(adapters), "two_baselines_v1_anchor_usd": B.usd(baselines),
                "total_usd": B.usd(adapters + baselines),
                "dev_only_three_1600_family_adapters_usd": B.usd(3 * 600 * sec["v2/dev"]),
                "on_the_experiment_1_engine_usd": B.usd(6 * sum(n * e1["seconds_per_episode"][k] for k, n in splits)
                                                        + 2 * 720 * e1["untrained_seconds_per_episode"])}
    finally:
        B.ASSUME["job_overhead"] = old


def main() -> None:
    x = inputs()
    B, S, sec = x["B"], x["S"], x["seconds_per_episode"]
    y, a = x["yield"], x["attempt_seconds"]
    full = sec["v2/test_id"] * 1.25  # an attempt that runs to its end like a greedy episode (the plan's ratio)
    rows = {"**as measured in Stage 1** (seed adapter: yield 33%, 0.36 s per attempt)": (y, a),
            "yield at the pilot's floor (20%), attempts as measured": (0.20, a),
            "yield as measured, every attempt as long as a completed episode": (y, full),
            "yield at the floor and every attempt as long as a completed episode": (0.20, full),
            "yield 50% in later cycles, attempts as measured": (0.50, a)}
    grid = {k: cost(x, *v) for k, v in rows.items()}
    ref = grid[next(iter(rows))]
    fast = "as measured, sealed splits also on the new engine (needs a decision: see below)"
    grid[fast] = cost(x, y, a, sealed_on_new_engine=True)
    rows[fast] = (y, a)
    teachers = {}
    for name, t in S["teachers"].items():
        if "usd_per_accepted_demonstration" in t:
            per = t["usd_per_accepted_demonstration"] * x["job_overhead"] * B.ASSUME["contingency"]
            teachers[name] = {"qualifies": t["qualifies"], "acceptance": t["acceptance"],
                              "acceptance_four_hops": t["acceptance_by_hops"]["4"], "flavor_usd_per_hour": t["usd_per_hour"],
                              "seconds_per_attempt": t["seconds_per_attempt"], "usd_per_accepted_demonstration": per,
                              "five_replicates_usd": 5 * 2880 * per, "three_replicates_usd": 3 * 2880 * per}
    d = deferred_scoring(x)
    plan = json.loads((OUT / "budget.json").read_text())["scenarios"]["Stage 1 target (4x)"]["planning value"]
    res = {"inputs": {k: v for k, v in x.items() if k not in ("B", "S", "e1")}, "scenarios": grid, "teachers": teachers,
           "deferred_1.5B_scoring": d, "stage1_spent_usd": S["books"]["settled_usd"],
           "plan_reference_usd": {k: plan[k] for k in ("stage2_pilot_usd", "stage3_block_a_usd", "stage3_block_b_usd",
                                                         "extension_usd", "stage4_teacher_adapters_usd")}}
    (OUT / "budget_stage1.json").write_text(json.dumps(res, indent=1) + "\n")

    e1 = x["e1"]
    md = ["# Experiment 2: remaining budget from Stage 1's measurements (generated by scripts/s2_e2_rebudget.py)", "",
          "The cost model of [budget.md](budget.md), with the same design and formulas. Its projected inputs are "
          "replaced by what Stage 1 measured. A10G at 1.00 USD per hour.", "",
          "## Inputs", "",
          "| quantity | the plan assumed | Stage 1 measured | source |", "|---|---|---|---|",
          f"| greedy scoring against Experiment 1 | 4 times faster (target), 1 to 10 in the scenarios | "
          f"**{x['speed_up']:.2f} times faster**: {1.62 / x['speed_up']:.3f} s per own-family episode against 1.62 | "
          "calibration, two adapters, 800 episodes |",
          f"| a sampled practice attempt | 1.25 greedy `test_id` episodes ({e1['seconds_per_episode']['v2/test_id'] * 1.25 / 4:.2f} s "
          f"at the target) | **{a:.3f} s** ({a / sec['v2/test_id']:.2f} of a greedy `test_id` episode on this engine; "
          "most failing attempts end early) | yield probe, 1,600 attempts |",
          f"| yield, accepted tasks per attempted task | 40% (20% to 60%) | **{y:.1%}** with the seed adapter, cycle 1 | "
          "yield probe, 400 tasks |",
          f"| training | {e1['train_seconds_per_step']:.2f} s per step | **{x['train_seconds_per_step']:.2f} s per step** on a "
          "cycle-1 buffer; no saving from lighter gradient checkpointing | training benchmark |",
          f"| job overhead (start-up, installation, loading, per-minute billing) | x1.10 | **x{x['job_overhead']:.2f}** | "
          "billed against useful seconds, dev and probe job |",
          f"| contingency for failed and repeated jobs | x1.25 | x1.25 kept; {x['failed_share_of_stage1']:.0%} of Stage 1's "
          "bill went to its one failed job | ledger |", "",
          "Scoring times of the splits not timed in Stage 1 are Experiment 1's divided by the measured speed-up. "
          f"As a cross-check, `dev` took {S['seed']['dev_seconds_per_episode']:.2f} s per episode for the seed adapter, "
          f"against {sec['v2/dev']:.2f} s in the model, which is for a fully trained adapter that runs more turns.", "",
          "## Remaining stages (USD, including overhead and contingency)", "",
          f"Stage 1 has cost {S['books']['settled_usd']:.2f} USD so far. The plan's figures are those of its planning row "
          "(engine four times faster, yield 40%).", "",
          "| scenario | Stage 2 pilot | Stage 3A | Stage 3B | **Stages 2-3** | extension (replicates 5-7) | "
          "tasks attempted per cycle |", "|---|---|---|---|---|---|---|",
          f"| *the plan's planning row, for reference* | {plan['stage2_pilot_usd']:.0f} | {plan['stage3_block_a_usd']:.0f} | "
          f"{plan['stage3_block_b_usd']:.0f} | **{plan['stage2_pilot_usd'] + plan['stage3_block_a_usd'] + plan['stage3_block_b_usd']:.0f}** | "
          f"{plan['extension_usd']:.0f} | 1,200 |"]
    for k, (yy, _) in rows.items():
        s = grid[k]
        md.append(f"| {k} | {s['stage2_pilot_usd']:.0f} | {s['stage3_block_a_usd']:.0f} | {s['stage3_block_b_usd']:.0f} | "
                  f"**{s['remaining_usd']:.0f}** | {s['extension_usd']:.0f} | {min(480 / yy, 2400):,.0f} |")
    md += ["", "The first row is what the measurements say about cycle 1. Later cycles are not measured: a better "
               "adapter is accepted on more tasks (fewer attempts per cycle) and finishes more of its attempts (longer "
               "attempts). The third row is the upper bound on the second effect with no credit for the first.", "",
           "**Sealed splits.** The plan lets sealed splits be scored on the new engine only if its calibration "
           "passes. It met both outcome criteria (at most 1% of outcomes changed, success within 1 point) and "
           "missed the speed criterion, so every row but the last costs the sealed splits and the v1 anchor on the "
           "Experiment 1 path, as the rule is written. The last row shows what a decision to use the new engine for "
           f"them would save: {ref['remaining_usd'] - grid[fast]['remaining_usd']:.0f} USD over Stages 2-3.", "",
           "### One replicate, as measured (USD)", "",
           "| arm | rollouts | training | scoring | total |", "|---|---|---|---|---|"]
    for arm, v in ref["per_replicate_usd"].items():
        md.append(f"| {arm} | {v['rollouts']:.2f} | {v['training']:.2f} | {v['scoring']:.2f} | {sum(v.values()):.2f} |")
    md += ["", f"Stage 4 adapters (five teacher-arm adapters trained and scored): {ref['stage4_teacher_adapters_usd']:.0f} USD.", "",
           "## Teacher demonstrations (Stage 4, only if H2(a) holds and a teacher qualifies)", "",
           "From the qualification runs, at the acceptance and speed each model showed on 200 practice tasks, with "
           "overhead and contingency. The plan runs five replicates if the demonstrations cost at most 60 USD, "
           "otherwise three.", "",
           "| model | qualifies | accepted | at four hops | USD per hour | s per attempt | USD per accepted demonstration | "
           "five replicates (14,400) | three replicates (8,640) |", "|---|---|---|---|---|---|---|---|---|"]
    for name, t in teachers.items():
        md.append(f"| {name.replace('teacher_', '')} | {'yes' if t['qualifies'] else 'no'} | {t['acceptance']:.0%} | "
                  f"{t['acceptance_four_hops']:.0%} | {t['flavor_usd_per_hour']:.2f} | {t['seconds_per_attempt']:.1f} | "
                  f"{t['usd_per_accepted_demonstration']:.4f} | {t['five_replicates_usd']:.0f} | {t['three_replicates_usd']:.0f} |")
    md += ["", "## The 1.5B scoring Experiment 1 left out (approved for planning, not run)", "",
           f"Six adapters on `dev`, `test_depth`, `test_tool` and `test_domain` ({d['episodes_six_adapters']:,} episodes) "
           f"and the v1 anchor for the two untrained baselines ({d['episodes_two_baselines']:,} episodes).", "",
           "| item | USD |", "|---|---|",
           f"| six adapters, four splits, on the calibrated engine | {d['six_adapters_usd']:.2f} |",
           f"| two untrained baselines, v1 anchor (assumes the same speed-up for long malformed output, not measured) | "
           f"{d['two_baselines_v1_anchor_usd']:.2f} |",
           f"| **total** | **{d['total_usd']:.2f}** |",
           f"| the same on the Experiment 1 scoring path, which is what the plan's rule prescribes after a missed speed "
           f"criterion (the plan's 9.02 USD at the measured overhead) | {d['on_the_experiment_1_engine_usd']:.2f} |",
           f"| the part Stage 1's criterion (c) needs: `dev` for the three 1,600-family adapters | "
           f"{d['dev_only_three_1600_family_adapters_usd']:.2f} |", "",
           "Which of the two totals applies is the same decision as for the main run's sealed splits, above.", ""]
    (OUT / "budget_stage1.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
