"""The spending cap: reservations, refusals and settlement, without any network call."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("s2_launch", ROOT / "scripts/s2_launch.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

BUDGET = {"cap_usd": 60.0, "reserve_usd": 3.0, "max_parallel": 4}
L4 = 0.80 / 60


def reserve(key, minutes, price=L4):
    return {"event": "reserve", "key": key, "flavor": "l4x1", "timeout_min": minutes, "price_per_min": price,
            "reserved_usd": L.reservation_usd(minutes, price)}


def test_reservation_is_the_worst_case_and_cost_rounds_up():
    assert L.reservation_usd(60, L4) == pytest.approx((60 + L.BILLING_SLACK_MIN) * L4, abs=1e-4)
    assert L.job_cost_usd(61, L4) == pytest.approx(3 * L4, abs=1e-4)  # started minutes, plus one for start-up
    assert L.job_cost_usd(0, L4) == 0.0
    assert L.job_cost_usd(772, L4) == pytest.approx(0.1867, abs=1e-3)  # the first job: the provider billed 0.186662
    assert L.job_cost_usd(60 * 60, L4) <= L.reservation_usd(60, L4)  # a job cannot cost more than it reserved


def test_no_job_in_the_committed_ledger_was_billed_above_what_the_launcher_now_reserves():
    """Two Experiment 1 jobs ran into their time limit and were billed one minute more than the launcher had
    reserved for them. The reservation must cover the bill of every job the provider has charged so far."""
    jobs: dict = {}
    for e in L.read_ledger():
        j = jobs.setdefault(e.get("key"), {})
        if e["event"] == "reserve":
            j.update(timeout=e["timeout_min"], price=e["price_per_min"], recorded=e["reserved_usd"])
        elif e["event"] in ("settled", "provider_cost"):
            j["cost"] = e["cost_usd"]
    billed = [j for j in jobs.values() if "timeout" in j and "cost" in j]
    assert len(billed) >= 50 and any(j["cost"] > j["recorded"] for j in billed)  # the overshoot is in the record
    assert all(j["cost"] <= L.reservation_usd(j["timeout"], j["price"]) + 1e-9 for j in billed)


def test_submission_is_refused_when_the_cap_would_be_exceeded():
    ledger = [reserve("a", 1200), {"event": "submitted", "key": "a", "job_id": "1"},
              reserve("b", 1200), {"event": "submitted", "key": "b", "job_id": "2"}]
    b = L.books(ledger)
    assert b["reserved_usd"] == pytest.approx(2 * 1202 * L4, abs=1e-3) and b["settled_usd"] == 0
    ok, why = L.may_submit(ledger, BUDGET, L.reservation_usd(1800, L4))  # 32.05 + 24.03 = 56.08 <= 57
    assert ok, why
    ok, why = L.may_submit(ledger, BUDGET, L.reservation_usd(1900, L4))  # 32.05 + 25.36 = 57.41 > 57
    assert not ok and "refused" in why and "57.00" in why
    assert not L.may_submit([], BUDGET, 57.01)[0] and L.may_submit([], BUDGET, 57.0)[0]


def test_closed_jobs_count_at_the_billed_cost_and_open_jobs_at_their_reservation():
    ledger = [reserve("a", 600), {"event": "submitted", "key": "a", "job_id": "1"},
              {"event": "settled", "key": "a", "job_id": "1", "cost_usd": 2.0}]
    b = L.books(ledger)
    assert b["reserved_usd"] == 0 and b["settled_usd"] == 2.0 and not b["open"] and b["committed_usd"] == 2.0
    ledger.append({"event": "provider_cost", "key": "a", "job_id": "1", "cost_usd": 2.6})
    assert L.books(ledger)["settled_usd"] == 2.6  # the provider's bill replaces the estimate
    ledger += [reserve("b", 300), {"event": "submitted", "key": "b", "job_id": "2"}]
    # a job in progress is in the provider's running total and in its own reservation: it is counted once
    ledger.append({"event": "reconcile", "provider_used_usd": 4.1, "ledger_settled_usd": 2.6})
    b = L.books(ledger)
    assert b["committed_usd"] == pytest.approx(2.6 + L.reservation_usd(300, L4), abs=1e-3)
    ledger.append({"event": "reconcile", "provider_used_usd": 9.0, "ledger_settled_usd": 2.6})
    assert L.books(ledger)["committed_usd"] == 9.0  # if the provider's total is ever larger, it counts
    ledger += [reserve("c", 60), {"event": "submit_failed", "key": "c", "error": "x"}]
    assert L.books(ledger)["reserved_usd"] == pytest.approx(L.reservation_usd(300, L4), abs=1e-3)


def test_parallel_jobs_are_limited_and_the_committed_budget_file_is_sane():
    ledger = [e for i in range(4) for e in (reserve(f"k{i}", 10), {"event": "submitted", "key": f"k{i}", "job_id": str(i)})]
    ok, why = L.may_submit(ledger, BUDGET, 0.1)
    assert not ok and "max_parallel" in why
    budget = json.loads((ROOT / "configs/study02/budget.json").read_text())
    assert budget["cap_usd"] == 60.0 and 0 < budget["reserve_usd"] < budget["cap_usd"]
    assert max(budget["flavors"].values()) <= 1.0  # no flavour above 1 USD per hour is approved


def test_job_command_checks_out_one_commit_and_never_carries_a_secret():
    budget = json.loads((ROOT / "configs/study02/budget.json").read_text())
    cmd = L.job_command({"configs": ["configs/study02/e0/x.json"], "stages": ["prepare_v1", "train", "score"]},
                        budget, "abc123")
    text = cmd[-1]
    assert "git checkout --quiet abc123" in text and "requirements-cuda.lock" in text
    assert text.index("s2_prepare_v1.py") < text.index("s2_train.py") < text.index("s2_score.py")
    assert "hf_" not in text and "TOKEN" not in text


def test_a_closed_budget_refuses_and_a_stage_has_its_own_cap_and_ledger():
    old, old_ledger = L.load_budget()
    assert old["closed"] and old_ledger == L.LEDGER
    ok, why = L.may_submit(L.read_ledger(old_ledger), old, 0.01)  # 6.6 USD are left under the old cap
    assert not ok and "closed" in why
    stage, ledger = L.load_budget("e2_stage1")
    assert stage["cap_usd"] == 15.0 and 0 < stage["reserve_usd"] < stage["cap_usd"]
    assert stage["closed"] and not L.may_submit(L.read_ledger(ledger), stage, 0.01)[0]  # Stage 1 has ended
    stage = {**stage, "closed": False}  # the rule as it applied while the stage ran
    assert ledger != old_ledger and ledger.parent.name == "e2"
    # the provider's total before the stage's first job is what the closed ledger adds up to
    assert stage["provider_baseline_usd"] == pytest.approx(L.books(L.read_ledger(old_ledger))["settled_usd"], abs=0.001)
    limit = stage["cap_usd"] - stage["reserve_usd"]
    assert L.may_submit([], stage, limit)[0] and not L.may_submit([], stage, limit + 0.01)[0]
    b = L.books(L.read_ledger(ledger))  # whatever has run so far is inside the cap, by the launcher's own rule
    assert b["committed_usd"] <= limit + 1e-9


def test_stage_one_job_specs_use_the_stage_budget_and_fit_under_its_cap_together():
    stage, _ = L.load_budget("e2_stage1")
    specs = [json.loads(p.read_text()) for p in sorted((ROOT / "configs/study02/jobs/e2").glob("e2s1_*.json"))]
    assert len(specs) >= 4 and len({s["key"] for s in specs}) == len(specs)
    worst = 0.0
    for s in specs:
        assert s["budget"] == "e2_stage1" and s["flavor"] in stage["flavors"] and not s["configs"] and not s["stages"]
        worst += L.reservation_usd(s["timeout_min"], stage["flavors"][s["flavor"]] / 60)
        text = " ".join(L.job_command(s, stage, "0" * 40))
        assert "scripts/s2_e2.py" in text and "s2_score.py" not in text  # a Stage 1 job scores no sealed split
        assert "hf_" not in text and "HF_TOKEN" not in text and "secret" not in text.lower()
    assert worst <= stage["cap_usd"] - stage["reserve_usd"]  # even if every job ran to its time limit


def test_stage_two_is_prepared_with_its_own_cap_and_its_job_specs_fit_under_it():
    stage, ledger = L.load_budget("e2_stage2")
    assert stage["cap_usd"] == 10.0 and ledger.name == "ledger_stage2.jsonl" and set(stage["flavors"]) == {"a10g-small"}
    spent = L.books(L.read_ledger(ledger))
    if stage.get("closed") and not spent["closed"] and not spent["open"]:  # not approved yet: nothing can be submitted
        assert not L.may_submit([], stage, 0.01)[0]
    assert spent["committed_usd"] <= stage["cap_usd"] - stage["reserve_usd"] + 1e-9
    specs = [json.loads(p.read_text()) for p in sorted((ROOT / "configs/study02/jobs/e2").glob("e2s2_*.json"))]
    worst = 0.0
    for s in specs:
        assert s["budget"] == "e2_stage2" and s["flavor"] in stage["flavors"]
        worst += L.reservation_usd(s["timeout_min"], stage["flavors"][s["flavor"]] / 60)
        text = " ".join(L.job_command(s, stage, "0" * 40))
        assert "scripts/s2_e2.py" in text and "s2_score.py" not in text  # the pilot scores no sealed split
    assert len(specs) >= 2 and worst <= stage["cap_usd"] - stage["reserve_usd"]
