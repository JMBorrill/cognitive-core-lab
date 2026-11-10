"""The review's independent check of list selections, and the Experiment 2 cost model. CPU only, no network."""
import copy
import importlib.util
from pathlib import Path

from ccl import s2
from ccl.world2 import TOOLS2

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_selection_check_passes_on_dev_tasks_and_catches_a_planted_tie_and_a_planted_wrong_pick():
    check = load("s2_review").selections
    tasks = [t for t in s2.load_split("dev") if any(TOOLS2[g.tool].kind == "list" for g in t.gold)
             and t.expected_status in ("done", "blocked")][:60]
    assert len(tasks) == 60
    swapped = 0
    for t in tasks:
        verdict, n = check(t)
        assert verdict == "ok" and n >= 1, (t.task_id, verdict)
        g = next(g for g in t.gold if TOOLS2[g.tool].kind == "list")
        tool = TOOLS2[g.tool]
        (pid,) = g.args.values()
        listed = [k for k, r in t.world[tool.entity].items() if r[tool.ref_field] == pid]
        tie = copy.deepcopy(t)  # every listed record gets the same attribute values: no rule can single one out
        for k in listed[1:]:
            tie.world[tool.entity][k].update({f: v for f, v in tie.world[tool.entity][listed[0]].items()
                                              if not f.endswith("_id")})
        assert check(tie)[0] not in ("ok", "skipped"), t.task_id
        later = {str(v) for h in t.gold[t.gold.index(g) + 1:] for v in h.args.values()}
        unused = [k for k in listed if not {str(v) for f, v in t.world[tool.entity][k].items() if f.endswith("_id")} & later]
        if later and unused and not t.answer_facts:  # leave only a record that the gold graph does not go on to use
            wrong = copy.deepcopy(t)
            for k in listed:
                if k != unused[0]:
                    wrong.world[tool.entity][k][tool.ref_field] = "ZZ-0000"
            assert check(wrong)[0] not in ("ok", "skipped"), t.task_id
            swapped += 1
    assert swapped >= 5


def test_cost_model_moves_the_right_way_and_stage_one_does_not_assume_the_new_engine():
    B = load("s2_e2_budget")
    m = B.measured()
    assert 3.0 < m["train_seconds_per_step"] < 4.5 and 1.0 < m["seconds_per_episode"]["v2/test_comp"] < 2.5
    slow, fast = B.stages(m, 0.4, 1.0), B.stages(m, 0.4, 4.0)
    low, high = B.stages(m, 0.2, 4.0), B.stages(m, 0.6, 4.0)
    for k in ("stage2_pilot_usd", "stage3_block_a_usd", "stage3_block_b_usd", "extension_usd"):
        assert fast[k] < slow[k] and high[k] < fast[k] < low[k], k
    # Stage 1 is where the engine is measured, so its cost must not depend on the speed-up it is meant to find.
    assert slow["stage1_usd"] == fast["stage1_usd"]
    # a practice arm at the pilot's yield floor attempts exactly the task cap in every cycle
    d = B.DESIGN
    assert d["episodes_per_cycle"] / 0.20 == d["task_cap_per_cycle"]
    # the final adapter of every arm takes the same number of steps: two passes over seed plus six cycles
    assert (d["seed_episodes"] + d["cycles"] * d["episodes_per_cycle"]) * d["passes"] / d["episodes_per_step"] == 820


def test_recosting_uses_stage_one_measurements_and_keeps_the_plans_rule_for_sealed_splits():
    R = load("s2_e2_rebudget")
    x = R.inputs()
    # the measured engine is between two and four times faster: the plan's "re-cost and ask again" band
    assert 2.0 < x["speed_up"] < 4.0 and 0.2 <= x["yield"] < 0.6 and 0.2 < x["attempt_seconds"] < 1.0
    assert 3.0 < x["train_seconds_per_step"] < 4.5 and 1.0 < x["job_overhead"] < 1.5
    by_rule, relaxed = R.cost(x, x["yield"], x["attempt_seconds"]), R.cost(x, x["yield"], x["attempt_seconds"], True)
    assert relaxed["remaining_usd"] < by_rule["remaining_usd"]  # sealed splits on the Experiment 1 path cost more
    floor = R.cost(x, 0.20, x["attempt_seconds"])
    assert floor["remaining_usd"] > by_rule["remaining_usd"] and floor["stage2_pilot_usd"] <= 10  # the pilot's cap
    assert R.load("s2_e2_budget").ASSUME["job_overhead"] == 1.10  # the plan's own model is left as it was
    d = R.deferred_scoring(x)
    assert d["total_usd"] < d["on_the_experiment_1_engine_usd"] and d["dev_only_three_1600_family_adapters_usd"] < 1
