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
    assert L.reservation_usd(60, L4) == pytest.approx(61 * L4, abs=1e-4)
    assert L.job_cost_usd(61, L4) == pytest.approx(3 * L4, abs=1e-4)  # started minutes, plus one for start-up
    assert L.job_cost_usd(0, L4) == 0.0
    assert L.job_cost_usd(772, L4) == pytest.approx(0.1867, abs=1e-3)  # the first job: the provider billed 0.186662
    assert L.job_cost_usd(60 * 60, L4) <= L.reservation_usd(60, L4)  # a job cannot cost more than it reserved


def test_submission_is_refused_when_the_cap_would_be_exceeded():
    ledger = [reserve("a", 1200), {"event": "submitted", "key": "a", "job_id": "1"},
              reserve("b", 1200), {"event": "submitted", "key": "b", "job_id": "2"}]
    b = L.books(ledger)
    assert b["reserved_usd"] == pytest.approx(2 * 1201 * L4, abs=1e-3) and b["settled_usd"] == 0
    ok, why = L.may_submit(ledger, BUDGET, L.reservation_usd(1800, L4))  # 32.03 + 24.01 = 56.04 <= 57
    assert ok, why
    ok, why = L.may_submit(ledger, BUDGET, L.reservation_usd(1900, L4))  # 32.03 + 25.35 = 57.38 > 57
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
