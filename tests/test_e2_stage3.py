"""Experiment 2, amendment 2: the preparation of Stage 3 (twins of the test tasks, the audit's bounds, the
engine record, the teacher re-qualification). Nothing here runs a model or prints a task."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

from ccl import adversarial as A
from ccl import loop as L
from ccl import s2

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_twin_manifest_is_what_the_generator_builds_and_the_scorer_refuses_any_other_set(monkeypatch, tmp_path):
    score = load("s2_score")
    recorded = json.loads(score.TWINS.read_text())["splits"]
    assert set(recorded) == {f"v2/{s}_adv" for s in score.TWIN_SPLITS}
    amendment_1 = json.loads((ROOT / "configs/study02/e2/protocol2.json").read_text())["adversarial_dev"]
    v1, v2 = A.adversarial_dev(version=1), A.adversarial_dev()
    assert json.loads(score.TWINS.read_text())["version"] == A.VERSION == 2
    assert L.tasks_sha256(v1) == amendment_1["sha256"] != recorded["v2/dev_adv"]["sha256"] == L.tasks_sha256(v2)
    # version 1 was one step short on 12 twins (review of PR #3); version 2 repairs those and changes no other
    assert sum(bool(A.leaks(t, t.world)) for t in v1) == 12 and not any(A.leaks(t, t.world) for t in v2)
    assert sum(a.task_id == b.task_id for a, b in zip(v1, v2)) == 338 == recorded["v2/dev_adv"]["identical_to_version_1"]
    assert all(v["walks_from_other_candidates_that_miss_a_record"] == 0 for v in recorded.values())
    for split, want in recorded.items():
        tasks = score.tasks_for(split, {})  # rebuilt; the scorer checks count and hash itself
        assert len(tasks) == want["twins"] == want["oracle_transcript_identical"] and L.tasks_sha256(tasks) == want["sha256"]
        assert all(t.split == split[3:] and t.task_id.startswith(split[3:] + "-") for t in tasks)
        assert sum(want["hops"].values()) == want["twins"] <= want["tasks_in_split"]
    assert len({v["sha256"] for v in recorded.values()}) == len(recorded)
    wrong = json.loads(score.TWINS.read_text())
    wrong["splits"]["v2/dev_adv"]["sha256"] = "0" * 64
    (tmp_path / "twins.json").write_text(json.dumps(wrong))
    monkeypatch.setattr(score, "TWINS", tmp_path / "twins.json")
    with pytest.raises(AssertionError, match="not the pre-registered set"):
        score.tasks_for("v2/dev_adv", {})


def test_only_the_scorer_can_build_twins_of_test_tasks():
    text = (ROOT / "ccl/adversarial.py").read_text()
    assert "sealed" not in text and not re.search(r"test_(comp|depth|tool|domain|id)", text)
    assert set(re.findall(r"load_split\(([^)]*)\)", text)) == {'"dev"'}
    for src in ("ccl/loop.py", "ccl/engine.py", "ccl/verify.py", "scripts/s2_e2.py", "scripts/s2_e2_stage3.py"):
        text = (ROOT / src).read_text()
        assert "twins(" not in text and "s2_score" not in text, src
        assert set(re.findall(r"load_split\(([^)]*)\)", text)) <= {'"dev"'}, src
    for src in ("ccl/loop.py", "ccl/engine.py", "ccl/verify.py", "ccl/hf.py", "ccl/sft.py"):  # nor does training code import the twins
        assert "adversarial" not in (ROOT / src).read_text(), src
    # the dev twins did not move when the builder was generalised
    assert A.twin(next(t for t in s2.load_split("dev") if A.depends_on_selection(t))).split == "dev_adv"


def test_audit_bounds_and_denominators():
    S = load("s2_e2_stage3")
    assert abs(S.upper(0, 181) - 0.0164) < 2e-4 and abs(S.upper(0, 20) - 0.1391) < 2e-4
    assert abs(S.upper(1, 236) - 0.02) < 2e-4 and S.upper(0, 0) == 1.0
    assert S.reads_needed(0.02) == 149 and S.reads_needed(0.02, 1) == 236
    a = json.loads((ROOT / "results/study02/e2/stage3_prep.json").read_text())["audit"]
    assert a["read"] + a["checked_by_program_only"] == a["sampled_and_checked_by_program"] == 200
    assert a["read"] == a["read_because_flagged"] + a["read_blind"] == 39 and a["false_accepts_found"] == 0
    st = a["strata"]
    assert st["not_strictly_successful"]["population"] + st["strictly_successful"]["population"] == a["accepted_episodes"]
    ub = a["upper_bound_95_one_sided"]
    assert ub["if_the_programs_check_is_relied_on"]["all_accepted"] < 0.02 < ub["from_reading_alone"]["all_accepted"]


def test_the_record_agrees_with_the_committed_results_and_the_teacher_budget_holds():
    r = json.loads((ROOT / "results/study02/e2/stage3_prep.json").read_text())
    e = r["engine"]
    for v in e["calibration_greedy_own_family_tasks"].values():
        assert v["r6"]["changed_against_experiment_1"] <= 4 and v["r6"]["same_text_as_full_re_encoding"] == "24/24"
    assert e["first_version_of_r6_rejected"]["changed_against_experiment_1"] == 113
    d = e["seed_adapter_on_dev_greedy"]
    assert abs(d["r6_success"] - d["r1_r4_success"]) * d["tasks"] == pytest.approx(d["fail_to_success"] - d["success_to_fail"])
    c = r["cost"]["scenarios"]
    new, old = c["test splits on the corrected engine"], next(v for k, v in c.items() if "Experiment 1" in k)
    assert new["block_a_with_twins_usd"] < old["block_a_with_twins_usd"] and new["twins_block_a_usd"] > 0
    for k, v in r["teacher"].items():  # the thresholds are the plan's, whatever the result
        p2 = v["protocol_2"]
        assert v["qualifies_under_protocol_2"] == (v["complete"] and v["protocol_of_the_run"] == 2 and p2["acceptance"] >= 0.60 and p2["four_hop"] >= 0.40)
        assert p2["accepted"] <= v["protocol_1"]["accepted"]
    assert load("s2_e2").QUALIFY == {"overall": 0.60, "four_hop": 0.40}
    budget = json.loads((ROOT / "configs/study02/budget_e2_teacher2.json").read_text())
    assert budget["cap_usd"] == 2.0
    worst = 0.0
    for f in (ROOT / "configs/study02/jobs/e2").glob("e2t2_*.json"):
        spec = json.loads(f.read_text())
        assert spec["budget"] == "e2_teacher2" and "s2_score.py" not in " ".join(spec["extra"]) and "teacher" in spec["extra"][0]
        worst += (spec["timeout_min"] + 2) * budget["flavors"][spec["flavor"]] / 60
    ledger = [json.loads(l) for l in open(ROOT / budget["ledger"])]
    settled = sum(x.get("cost_usd", 0) for x in ledger if x["event"] == "settled")
    assert max(settled, 0) <= budget["cap_usd"] and all(x["reserved_usd"] <= budget["cap_usd"] for x in ledger if x["event"] == "reserve")


def test_a_cycle_refuses_chunks_of_another_protocol_and_a_stopped_arm_stays_stopped(tmp_path):
    from ccl.oracle import OraclePolicy
    from ccl.runtime import Episode
    fams, tools = s2.load_practice()

    def run(tasks, seed):
        out = []
        for t in tasks:
            for a in range(L.ATTEMPTS):
                e = Episode(t, "x")
                e.attempt = a
                while not e.done:
                    e.step(OraclePolicy().act(e))
                out.append(e)
        return out
    L.Practice(tmp_path / "arm", 0, "practice", fams, tools, protocol=1).cycle(1, run, max_tasks=32)  # a probe: chunks, no state
    with pytest.raises(AssertionError, match="another protocol"):
        L.Practice(tmp_path / "arm", 0, "practice", fams, tools).cycle(1, run, max_tasks=32)
    e2 = load("s2_e2")
    (tmp_path / "x").mkdir()
    with pytest.raises(SystemExit) as stop:
        e2.stop(tmp_path / "x", None, "cycle 1: yield 0.100 is under 20%")
    assert stop.value.code == 3 and "under 20%" in (tmp_path / "x/STOPPED").read_text()
    text = (ROOT / "scripts/s2_e2.py").read_text()
    assert text.index('assert not (out / "STOPPED").exists()') < text.index("arm = L.Practice(out, a.replicate, a.arm, fams, tools)\n    before")
    assert "sys.exit(3)" not in text.replace('print(f"[e2] stopped: {why}", flush=True)\n    sys.exit(3)', "")
    launch = (ROOT / "scripts/s2_launch.py").read_text()
    assert launch.index("provider_usage()[0] - budget[\"provider_baseline_usd\"]") < launch.index('append({"event": "reserve"')


def test_selection_picks_separate_applying_the_rule_from_a_lucky_pick():
    from ccl.oracle import OraclePolicy
    from ccl.runtime import run_batch
    tasks = [t for t in A.adversarial_dev()][:120]
    eps = run_batch(tasks, OraclePolicy(), batch_size=512)
    picks = [p for e in eps for p in A.selection_picks(e.task, e.trace.to_dict())]
    assert len(picks) > 100 and all(ok for ok, _ in picks) and all(0 < c <= 0.5 for _, c in picks)  # the oracle always takes the gold row
    rows = [s2.metrics_row(e.task, e, "v2/dev_adv", "oracle") for e in eps[:20]]
    assert all(r["picks"] == r["picks_gold"] and r["picks_chance"] <= r["picks"] for r in rows) and sum(r["picks"] for r in rows) > 0
