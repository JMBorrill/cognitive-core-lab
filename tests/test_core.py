import json
import random

import pytest

from ccl.evaluate import episode_metrics, fact_present, validate
from ccl.oracle import OraclePolicy
from ccl.runtime import Episode, Limits, canonical_action_text, make_fixed_approver, parse_action, run_batch
from ccl.stats import bootstrap_ci, hierarchical_ci, paired_diff
from ccl.tasks import (ALIASES_EVAL, ALIASES_TRAIN, TEMPLATES, Task, audit_splits, generate_split,
                       gold_value_ok, graph_signature, make_task)
from ccl.tools import TOOLS, FaultPlan, Simulator, ToolError, validate_args


def calls(*cs):
    return json.dumps({"calls": [{"tool": t, "args": a} for t, a in cs]})


def final(text, status="done"):
    return json.dumps({"final": text, "status": status})


# ---------------------------------------------------------------- tools

def test_validate_args_rejects_bad_inputs():
    spec = TOOLS["get_weather"]
    with pytest.raises(ToolError, match="missing"):
        validate_args(spec, {"city": "Paris"})
    with pytest.raises(ToolError, match="unknown"):
        validate_args(spec, {"city": "Paris", "date": "2026-01-01", "units": "C"})
    with pytest.raises(ToolError, match="format"):
        validate_args(spec, {"city": "Paris", "date": "March 3"})
    with pytest.raises(ToolError, match="type"):
        validate_args(TOOLS["convert_currency"], {"amount": "10", "from_currency": "USD", "to_currency": "EUR"})
    assert validate_args(spec, {"city": "Paris", "date": "2026-01-01"}) == {"city": "Paris", "date": "2026-01-01"}


def test_renamed_tool_maps_params_back_to_canonical():
    spec = TOOLS["get_weather"].renamed("forecast_lookup", {"city": "place"})
    assert spec.key == "get_weather"
    assert validate_args(spec, {"place": "Oslo", "date": "2026-01-01"}) == {"city": "Oslo", "date": "2026-01-01"}
    assert "forecast_lookup(place: string" in spec.signature()


def test_simulator_is_deterministic_and_isolated():
    t = make_task("sel_weather", 3, "train")
    s1, s2 = Simulator(t.world), Simulator(t.world)
    args = t.gold[0].args
    assert s1.execute(TOOLS["get_weather"], args) == s2.execute(TOOLS["get_weather"], args)
    s1.execute(TOOLS["create_ticket"], {"title": "x", "priority": "low"})
    assert t.world["tickets"] == []  # task world is never mutated


def test_fault_plan_transient_then_ok():
    t = make_task("sel_weather", 1, "train")
    sim = Simulator(t.world, FaultPlan(transient={"get_weather": ["timeout"]}))
    with pytest.raises(ToolError) as e:
        sim.execute(TOOLS["get_weather"], t.gold[0].args)
    assert e.value.retryable
    assert "temp_c" in sim.execute(TOOLS["get_weather"], t.gold[0].args)


# ---------------------------------------------------------------- parsing

@pytest.mark.parametrize("text,kind,strict", [
    ('{"calls": [{"tool": "a", "args": {}}]}', "calls", True),
    ('```json\n{"final": "hi", "status": "done"}\n```', "final", False),
    ('<think>x</think>{"final": "hi", "status": "done"}', "final", False),
    ('{"tool": "a", "args": {"x": 1}}', "calls", False),
    ('{"final": "hi"}', "final", False),
    ("I cannot do that.", "invalid", False),
    ('{"calls": []}', "invalid", False),
    ('{"status": "failed"}', "invalid", False),
])
def test_parse_action(text, kind, strict):
    p = parse_action(text)
    assert p.kind == kind and p.strict == strict


def test_canonical_roundtrip():
    a = {"calls": [{"tool": "x", "args": {"k": 1}}]}
    assert parse_action(canonical_action_text(a)).calls == a["calls"]


# ---------------------------------------------------------------- runtime enforcement

def test_permission_is_enforced_by_runtime():
    t = make_task("perm_delete_viewer", 0, "train")
    ep = Episode(t, "test")
    fid = next(iter(t.world["files"]))
    ep.step(calls(("delete_file", {"file_id": fid})))
    assert ep.trace.calls[0].outcome == "permission_denied"
    assert fid in ep.world["files"]
    ep.step(final("no", "refused"))
    assert not validate(t, ep)["checks"]["no_permission_violation"]


def test_approval_denial_blocks_side_effect_and_retry_is_flagged():
    t = make_task("appr_email_denied", 0, "train")
    c = next(iter(t.world["contacts"].values()))
    gold_email = t.gold[1].args["to"]
    ep = Episode(t, "test")
    ep.step(calls(("send_email", {"to": gold_email, "subject": "s", "body": "b"})))
    assert ep.trace.calls[0].outcome == "approval_denied" and ep.world["outbox"] == []
    ep.step(calls(("send_email", {"to": gold_email, "subject": "s", "body": "b"})))
    ep.step(final("blocked", "blocked"))
    assert not validate(t, ep)["checks"]["no_retry_after_denial"]
    assert c is not None


def test_hallucinated_tool_and_limits():
    t = make_task("sel_order_status", 0, "train")
    ep = Episode(t, "test", Limits(max_turns=2, max_calls_per_turn=2))
    ep.step(calls(("teleport", {}), ("teleport", {}), ("teleport", {})))
    outs = [c.outcome for c in ep.trace.calls]
    assert outs == ["hallucinated", "hallucinated", "limit"]
    ep.step(calls(("teleport", {})))
    assert ep.done and ep.trace.termination == "max_turns"


def test_parse_error_budget_terminates():
    t = make_task("sel_calc", 0, "train")
    ep = Episode(t, "test", Limits(max_parse_errors=1))
    ep.step("hello")
    assert not ep.done
    ep.step("still not json")
    assert ep.done and ep.trace.termination == "parse_errors"


def test_human_approval_override():
    t = make_task("ch_contact_email", 0, "train")
    ep = Episode(t, "test", approver=make_fixed_approver(False))
    ep.step(calls(("send_email", {"to": t.gold[1].args["to"], "subject": "x", "body": "y"})))
    assert ep.trace.calls[0].approval is False


# ---------------------------------------------------------------- oracle + validator

def test_oracle_solves_every_template_with_and_without_renaming():
    tasks = []
    for tid, m in TEMPLATES.items():
        for s in range(6):
            split = "train" if m["split"] == "train" else "eval"
            tasks.append(make_task(tid, s, split, aliases=ALIASES_EVAL, rename_prob=0.5))
    for ep in run_batch(tasks, OraclePolicy(), batch_size=128):
        m = episode_metrics(ep.task, ep)
        assert m["success"] == 1.0, (ep.task.template, validate(ep.task, ep))
        assert m["unnecessary_calls"] == 0 and m["dependency_safe"] == 1.0
        assert m["turns"] == m["min_turns"]


def test_validator_rejects_skipped_lookup_and_wrong_answer():
    t = make_task("ch_order_customer", 0, "train")
    ep = Episode(t, "test")
    ep.step(final("It was placed by Bob (bob@x.com)."))
    v = validate(t, ep)
    assert not v["success"] and not v["checks"]["facts"] and not v["checks"]["required_calls"]


def test_dependency_safety_flags_fabricated_ids_and_same_turn_dependents():
    t = make_task("ch_contact_email", 0, "train")
    email = t.gold[1].args["to"]
    name = t.gold[0].args["name"]
    ep = Episode(t, "test")
    ep.step(calls(("find_contact", {"name": name}), ("send_email", {"to": email, "subject": "a", "body": "b"})))
    ep.step(final("sent"))
    assert episode_metrics(t, ep)["dependency_safe"] == 0.0


def test_fact_matching():
    assert fact_present("18", "It is 18°C")
    assert fact_present("1234.5", "about 1,234.50 EUR")
    assert not fact_present("18", "It is 180 degrees")
    assert fact_present("sunny", "Sunny skies")
    assert gold_value_ok("expr", "3 * 4 + 1", "13")
    assert gold_value_ok("set", ["a", "b"], ["b", "a"])


# ---------------------------------------------------------------- splits / leakage

def test_splits_are_disjoint_at_template_and_tool_level():
    train = generate_split("train", 400, 0)
    evals = {k: generate_split(k, 100, 0) for k in ["eval_ood", "eval_unseen"]}
    audit = audit_splits(train, evals)
    assert audit["eval_ood"]["template_overlap"] == 0
    assert audit["eval_unseen"]["template_overlap"] == 0
    assert audit["eval_unseen"]["unseen_tools_in_train"] == []
    assert not any(t.domain in ("inventory", "maps") for task in train for t in
                   [TOOLS[s.key] for s in task.catalog])


def test_renamed_eval_aliases_disjoint_from_train_aliases():
    tr = {v[0] for v in ALIASES_TRAIN.values()}
    ev = {v[0] for v in ALIASES_EVAL.values()}
    assert not tr & ev


def test_task_serialisation_roundtrip():
    t = make_task("ch_deep5", 4, "eval", aliases=ALIASES_EVAL, rename_prob=1.0)
    t2 = Task.from_dict(json.loads(json.dumps(t.to_dict())))
    assert graph_signature(t) == graph_signature(t2) and t2.catalog == t.catalog


def test_generation_is_reproducible():
    a = generate_split("eval_ood", 20, 3)
    b = generate_split("eval_ood", 20, 3)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]


# ---------------------------------------------------------------- stats

def test_bootstrap_and_paired_diff():
    rng = random.Random(0)
    a = [float(rng.random() < 0.3) for _ in range(300)]
    b = [float(rng.random() < 0.7) for _ in range(300)]
    ci = bootstrap_ci(a)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    d = paired_diff(a, b)
    assert d["lo"] > 0 and d["p"] < 0.01
    h = hierarchical_ci([a, a[::-1], a])
    assert h["seeds"] == 3 and h["lo"] <= h["mean"] <= h["hi"]
