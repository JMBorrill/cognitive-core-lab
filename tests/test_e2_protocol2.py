"""Experiment 2, amendment 1 (protocol 2): the per-part answer check, the shortcut detector, the adversarial
evaluation with complete candidate records, and the acceptance rule that uses the first two."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ccl import adversarial as A
from ccl import loop as L
from ccl import s2
from ccl import verify as V
from ccl.evaluate import validate
from ccl.oracle import OraclePolicy
from ccl.runtime import Episode, run_batch
from ccl.tools import SIMULATORS, ToolError
from ccl.world2 import TOOLS2

ROOT = Path(__file__).resolve().parents[1]
PILOT1 = ROOT / "results/study02/runs/e2/r0_practice"


@pytest.fixture(scope="module")
def practice():
    return s2.load_practice()


@pytest.fixture(scope="module")
def stream(practice):
    return L.Stream(*practice, 0)


@pytest.fixture(scope="module")
def pilot1():
    """Every attempt of the first pilot, by (cycle, stream index, attempt)."""
    out = {}
    for f in sorted((PILOT1 / "attempts").glob("c*_*.jsonl.gz")):
        for x in s2.read_gz(f):
            out[(int(f.name[1]), x["index"], x["attempt"])] = x
    return out


def task(facts, reference):
    return SimpleNamespace(answer_facts=facts, reference_answer=reference)


# ---------------------------------------------------------------- multi-part answers

def test_a_multi_part_answer_must_give_each_fact_for_its_own_part():
    ref = "The grade of the enrollment in course CS-1 is merit. The grade of the enrollment in course CS-2 is merit."
    t = task(["merit", "merit"], ref)
    assert V.answer_parts(ref, t.answer_facts) == [("grade of the enrollment in course CS-1", "merit"),
                                                   ("grade of the enrollment in course CS-2", "merit")]
    assert V.parts_ok(t, ref)
    # the false accept the first pilot's audit found: the word occurs, but only for one part
    assert not V.parts_ok(t, "The grade of the enrollment in course CS-1 is merit. The grade of the enrollment in course CS-2 is fail.")
    assert not V.parts_ok(t, "The grade of the enrollment in course CS-1 is fail. The grade of the enrollment in course CS-2 is merit.")
    assert V.parts_ok(t, "merit and merit") and not V.parts_ok(t, "merit") and not V.parts_ok(t, None)
    d = task(["expired", "active"], "The status of the contract with supplier SU-3144 is expired. The status of the contract with supplier SU-1237 is active.")
    assert V.parts_ok(d, d.reference_answer)
    assert not V.parts_ok(d, "The status of the contract with supplier SU-3144 is active. The status of the contract with supplier SU-1237 is expired.")
    # a correct answer in other words and in the other order (a teacher wrote this one) is still correct
    assert V.parts_ok(d, "The contract with supplier SU-1237 (CT-5109) is active, and the contract with supplier SU-3144 (CT-5500) is expired.")
    assert not V.parts_ok(d, "The contract with supplier SU-1237 (CT-5109) is expired, and the contract with supplier SU-3144 (CT-5500) is active.")
    assert V.parts_ok(d, "expired for SU-3144; active for SU-1237") and not V.parts_ok(d, "active for SU-3144; expired for SU-1237")
    # two questions about one record: no identifier tells the parts apart, their own words do
    p = task(["2026-07-17", "3888.83"], "For purchase order PO-1: The end date of the contract governing that purchase order is "
             "2026-07-17. The amount of the invoice issued against that purchase order is 3888.83.")
    assert V.parts_ok(p, p.reference_answer)
    assert V.parts_ok(p, "The invoice amount is 3,888.83 and the contract ends on 2026-07-17.")
    assert not V.parts_ok(p, "The contract ends on 2026-07-17.")
    n = task(["87.80", "152.71"], "The ticket price of show SH-5190 is 87.80. The ticket price of show SH-1031 is 152.71.")
    assert V.parts_ok(n, "The ticket price of show SH-5190 is 87.8. The ticket price of show SH-1031 is 152.71.")  # numbers by value
    assert not V.parts_ok(n, "The ticket price of show SH-5190 is 152.71. The ticket price of show SH-1031 is 87.80.")
    assert V.parts_ok(task(["x"], "anything"), "no x here? yes: x") and V.parts_ok(task([], ""), None)  # one fact or none: not this check's
    h = task(["Paris", "Paris"], "Both offices are in Paris.")  # a hand-written reference (Study 01): occurrences are counted
    assert not V.parts_ok(h, "Paris.") and V.parts_ok(h, "Paris and Paris")


def test_the_per_part_check_passes_every_oracle_answer_and_rejects_what_the_first_pilot_let_through(pilot1, stream):
    dev = s2.load_split("dev")
    assert all(V.parts_ok(e.task, e.trace.final) for e in run_batch(dev, OraclePolicy(), batch_size=512))
    assert sum(len(t.answer_facts) >= 2 for t in dev) > 80
    x = next(v for k, v in pilot1.items() if k[:2] == (2, 1962) and not v["rejection"])  # the attempt the audit read
    t = stream.get(1962)
    assert t.answer_facts == ["merit", "merit"] and not x["rejection"]  # accepted by the validator as it was
    assert x["trace"]["final"].count("merit") == 1 and not V.parts_ok(t, x["trace"]["final"])
    wrong = [(k, v) for k, v in pilot1.items() if not v["rejection"] and not V.parts_ok(stream.get(k[1]), v["trace"]["final"])]
    assert len(wrong) == 37  # every one of them lacks the reference's sentence for some part
    for k, v in wrong:
        parts = V.answer_parts(stream.get(k[1]).reference_answer, stream.get(k[1]).answer_facts)
        assert any(f"{s} is {f}".lower() not in v["trace"]["final"].lower() for s, f in parts)


# ---------------------------------------------------------------- the shortcut

def call(tool, arg, result=None, error=None):
    return {"tool": tool, "args": {"id": arg}, "outcome": "error" if error else "ok",
            "result": {"error": error} if error else (result if result is not None else {"id": arg})}


LISTED = call("list_doctor_assays", "DR-1", {"assays": [{"assay_id": "AS-1", "instrument_id": "IS-1"},
                                                        {"assay_id": "AS-2", "instrument_id": "IS-2"},
                                                        {"assay_id": "AS-3", "instrument_id": "IS-3"}]})


def test_shortcut_is_a_probe_and_a_hit_on_one_column_whatever_the_tool_or_the_order():
    S = lambda *calls: V.shortcut({"calls": [LISTED, *calls]})  # noqa: E731
    get, nf = (lambda i: call("get_instrument", i)), (lambda i: call("get_instrument", i, error="not_found"))
    under = lambda i, rows: call("list_instrument_assays", i, {"assays": rows})  # noqa: E731
    assert S(nf("IS-1"), nf("IS-2"), get("IS-3"))  # "not found" until one exists
    assert S(under("IS-1", []), under("IS-3", [{"assay_id": "AS-9"}]))  # empty lists until one has records
    assert S(get("IS-3"), nf("IS-1"))  # the right one first, then the others: the probe confirmed it
    assert S(under("IS-1", []), get("IS-3"))  # the probe and the hit need not be the same tool
    assert not S(get("IS-3"))  # the rule was applied
    assert not S(nf("IS-1"))  # nothing was found: a task whose path ends there, or a failure
    assert not S(call("get_instrument", "IS-3", error="timeout"), get("IS-3"))  # a retry
    assert not S(nf("AS-3"), get("IS-3"))  # a mistyped identifier from another column, then the right one
    assert not S(nf("ZZ-9"), get("IS-3"))  # an invented identifier
    assert not S(under("IS-3", []), get("IS-3"))  # the same candidate asked two things
    # the hit failed once itself (a wrong tool first) and other candidates were probed: still a shortcut
    # (review of PR #3: the first implementation subtracted every probed value from the hits)
    assert S(call("find_instrument", "IS-3", error="not_found"), under("IS-3", [{"assay_id": "AS-9"}]), under("IS-1", []), under("IS-2", []))
    assert S(nf("IS-1"), call("find_instrument", "IS-3", error="not_found"), get("IS-3"))
    assert not V.shortcut({"calls": [nf("IS-1"), LISTED, get("IS-3")]})  # a probe needs the list before it
    one = call("list_doctor_assays", "DR-1", {"assays": [{"assay_id": "AS-1", "instrument_id": "IS-1"}]})
    assert not V.shortcut({"calls": [one, nf("IS-9"), get("IS-1")]})  # one candidate: nothing to tell apart
    limit = {**nf("IS-1"), "outcome": "limit"}
    assert not S(limit, get("IS-3"))
    assert V.enumerates({"calls": [LISTED, get("IS-1"), get("IS-3")]}) and not V.shortcut({"calls": [LISTED, get("IS-1"), get("IS-3")]})
    assert not V.enumerates({"calls": [LISTED, get("IS-3"), get("IS-3")]})


def test_the_detector_agrees_with_every_verdict_of_the_first_pilots_audit_and_with_the_two_earlier_forms(pilot1):
    audit = json.loads((PILOT1 / "audit.json").read_text())
    verdicts = json.loads((PILOT1 / "audit_verdicts.json").read_text())
    seen = {"elimination": 0, "none": 0, "unflagged": 0}
    for x in audit["sample"]:
        trace = pilot1[(x["cycle"], x["index"], x["attempt"])]["trace"]
        kind = verdicts.get(f"{x['cycle']}:{x['index']}", {}).get("kind")
        if kind is None:
            assert not V.shortcut(trace)
            seen["unflagged"] += 1
        elif "elimination" in kind:
            assert V.shortcut(trace)
            seen["elimination"] += 1
        elif kind == "none":
            assert not V.shortcut(trace)
            seen["none"] += 1
    assert seen == {"elimination": 53, "none": 26, "unflagged": 120}
    admissible = [x for x in pilot1.values() if not x["rejection"]]
    assert len(admissible) == 2894
    assert [V.shortcut(x["trace"]) for x in admissible] == [L.shortcut_v1(x["trace"]) for x in admissible]
    assert sum(V.shortcut(x["trace"]) for x in admissible) == 128
    assert not any(V.shortcut(e.trace.to_dict()) for e in run_batch(s2.load_split("dev"), OraclePolicy(), batch_size=512))


# ---------------------------------------------------------------- the adversarial evaluation

@pytest.fixture(scope="module")
def twins():
    dev = {t.task_id: t for t in s2.load_split("dev")}
    adv = A.adversarial_dev(version=1)
    return [(t, dev[next(x for x in t.tags if x.startswith("twin_of="))[8:]]) for t in adv]


def wrong_walk(t):
    """Follow the gold path's tools, but from every list take a record other than the one the gold path
    takes. Returns what each call after the first wrong turn came back with: 'record', 'rows', 'empty' or
    'missing'."""
    world = json.loads(json.dumps(t.world))
    world["_log"] = []
    result, seen, off = {}, [], False
    for g in t.gold:
        tool = TOOLS2[g.tool]
        if tool.kind == "action":
            break
        args = dict(g.args)
        for d in g.deps:  # the argument comes from a field of the record a dependency returned
            src = result.get(d)
            if src is None:
                continue
            for k, v in g.args.items():
                f = next((f for f, x in src["gold"].items() if x == v), None)
                if f is not None and src["mine"] is not None:
                    args[k] = src["mine"].get(f, v)
        try:
            out = SIMULATORS[g.tool](world, args)
        except ToolError as e:
            seen.append("missing" if e.code == "not_found" else e.code)
            break
        rows = next((v for v in out.values() if isinstance(v, list)), None)
        gold_out = SIMULATORS[g.tool](json.loads(json.dumps({**t.world, "_log": []})), dict(g.args)) if not off else None
        if off:
            seen.append("record" if rows is None else ("rows" if rows else "empty"))
            if rows == []:
                break
        if rows is None:
            result[g.id] = {"gold": gold_out if gold_out else out, "mine": out}
        else:
            later = {str(v) for h in t.gold if g.id in h.deps for v in h.args.values()}  # what the calls that depend on this list use
            picked = next((r for r in (gold_out or out)[next(k for k, v in (gold_out or out).items() if isinstance(v, list))]
                           if {str(v) for v in r.values()} & later), None) if not off else None
            other = next((r for r in rows if r is not picked and r != picked), None)
            if not off and picked is not None and other is not None:
                off = True
                result[g.id] = {"gold": picked, "mine": other}
            else:
                result[g.id] = {"gold": picked or (rows[0] if rows else {}), "mine": rows[0] if rows else None}
    return seen


def test_twins_are_the_same_task_for_an_agent_that_applies_the_rule(twins):
    assert len(twins) == 350 and len({t.task_id for t, _ in twins}) == 350
    recorded = json.loads((ROOT / "configs/study02/e2/protocol2.json").read_text())["adversarial_dev"]
    assert L.tasks_sha256([t for t, _ in twins]) == recorded["sha256"] and recorded["tasks"] == 350
    assert L.tasks_sha256(A.adversarial_dev(version=1)) == recorded["sha256"]  # built again: the same bytes
    on_twin = run_batch([t for t, _ in twins], OraclePolicy(), batch_size=512)
    on_orig = {e.task.task_id: e for e in run_batch([o for _, o in twins], OraclePolicy(), batch_size=512)}
    for e, (t, o) in zip(on_twin, twins):
        assert validate(t, e)["success"] and V.parts_ok(t, e.trace.final) and not V.shortcut(e.trace.to_dict())
        assert e.trace.messages[1:] == on_orig[o.task_id].trace.messages[1:]  # the same observations, call for call
        assert (t.request, t.gold, t.answer_facts, t.catalog, t.faults) == (o.request, o.gold, o.answer_facts, o.catalog, o.faults)
        for name, table in o.world.items():  # every original record is there, unchanged
            if name != "_log":
                assert all(t.world[name][rid] == rec for rid, rec in table.items())
        assert t.split == "dev_adv" and A.depends_on_selection(o)
    review = L._review()
    assert all(review(t)[0] in ("ok", "skipped") for t, _ in twins)  # every selection still singles out one record


def test_a_wrong_candidate_leads_somewhere_on_a_twin_and_nowhere_on_the_original(twins):
    """The shortcut's signal: on the original worlds a walk from another candidate runs into a missing record
    or an empty list; on the twins it never does, so nothing tells the candidates apart but the rule."""
    told, walked = 0, 0
    for t, o in twins:
        a, b = wrong_walk(t), wrong_walk(o)
        if not b:
            continue  # the selection is the last call before an action or the answer
        walked += 1
        assert a and "missing" not in a and "empty" not in a, (t.task_id, a)
        told += "missing" in b or "empty" in b
    # 300 of 304: in the other four the next call does not depend on the candidate (a lookup by name)
    assert walked > 300 and told >= 0.98 * walked


# ---------------------------------------------------------------- the acceptance rule and the loop under protocol 2

def eliminating(t, attempt=0):
    """The oracle, except that before the first call whose argument comes from a list it asks the same of
    another candidate. Returns (episode, whether such a call was made)."""
    e = Episode(t, "eliminating")
    e.attempt = attempt
    probed = False
    while not e.done:
        act = json.loads(OraclePolicy().act(e))
        if not probed and "calls" in act:
            for c in act["calls"]:
                for k, v in c["args"].items():
                    for rec in e.trace.calls:
                        rows = next((x for x in rec.result.values() if isinstance(x, list)), None) if rec.outcome == "ok" else None
                        for f in ({f for r in rows for f in r} if rows and len(rows) > 1 else ()):
                            col = [r[f] for r in rows]
                            if v in col and len(set(col)) > 1 and not probed:
                                other = next(x for x in col if x != v)
                                e.step(json.dumps({"calls": [{"tool": c["tool"], "args": {**c["args"], k: other}}]}))
                                probed = True
        if not e.done:
            e.step(json.dumps(act))
    return e, probed


def test_protocol_two_rejects_the_shortcut_and_wrong_parts_and_protocol_one_is_unchanged(stream):
    t = next(x for x in stream.first(300) if x.depth >= 3 and "mod=none" in x.tags and A.depends_on_selection(x)
             and x.expected_status == "done")
    e, probed = eliminating(t)
    assert probed and validate(t, e)["success"]
    assert s2.rejection(t, e) == [] and s2.rejection(t, e, protocol=2) == ["shortcut"]  # the rule of the proposal admits it
    row = L.attempt_row(7, e)
    assert row["protocol"] == 2 == L.PROTOCOL and row["rejection"] == ["shortcut"] and row["rejection_v1"] == []
    assert L.choose([row]) is None and not row["well_formed"]
    assert L.replay(t, json.loads(json.dumps(row["trace"])))[0] is None  # nor does it pass a replay under protocol 2
    assert L.replay(t, json.loads(json.dumps(row["trace"])), protocol=1)[1] == "ok"
    clean = Episode(t, "oracle")
    while not clean.done:
        clean.step(OraclePolicy().act(clean))
    assert s2.rejection(t, clean, protocol=2) == [] and L.replay(t, json.loads(json.dumps(clean.trace.to_dict())))[1] == "ok"
    twin_task = next(x for x in stream.first(600) if len(x.answer_facts) == 2 and x.answer_facts[0] != x.answer_facts[1]
                     and "twin" in x.family and x.expected_status == "done")
    w = Episode(twin_task, "swapped")
    while not w.done:
        act = json.loads(OraclePolicy().act(w))
        if "final" in act:
            a, b = twin_task.answer_facts
            act["final"] = act["final"].replace(f" is {a}.", " is \x00.").replace(f" is {b}.", f" is {a}.").replace(" is \x00.", f" is {b}.")
        w.step(json.dumps(act))
    assert validate(twin_task, w)["success"]  # every fact occurs, so the original validator passes it
    assert s2.rejection(twin_task, w) == [] and s2.rejection(twin_task, w, protocol=2) == ["answer_parts"]
    m = s2.metrics_row(twin_task, w, "v2/dev", "x")
    assert m["success"] == 1.0 and m["accepted"] and m["check_parts"] == 0.0 and m["success_parts"] == 0.0 == m["success_clean"]
    m = s2.metrics_row(t, e, "v2/dev", "x")
    assert m["success"] == 1.0 and m["shortcut"] and m["success_parts"] == 1.0 and m["success_clean"] == 0.0


def test_no_shortcut_episode_enters_a_buffer_under_protocol_two_and_the_first_pilots_arm_keeps_protocol_one(practice, tmp_path):
    fams, tools = practice

    def run(tasks, seed):  # every attempt probes another candidate where it can; half of them answer wrongly instead
        out = []
        for t in tasks:
            for a in range(L.ATTEMPTS):
                e, _ = eliminating(t, a) if a % 2 == 0 else (Episode(t, "x"), False)
                if a % 2:
                    e.attempt = a
                    while not e.done:
                        e.step(OraclePolicy().act(e))
                out.append(e)
        return out
    arm = L.Practice(tmp_path / "v2", 0, "practice", fams, tools)
    assert arm.protocol == 2 and arm.state["protocol"] == 2
    rec = arm.cycle(1, run, max_tasks=200)
    rows = [x for f in sorted((tmp_path / "v2/attempts").glob("*.gz")) for x in s2.read_gz(f)]
    cut = [x for x in rows if "shortcut" in x["rejection"]]
    assert len(cut) > 50 and all(x["rejection_v1"] == [] for x in cut) and rec["attempts_rejected_for_shortcut"] == len(cut)
    picks = [L.choose([x for x in rows if x["index"] == i]) for i in rec["accepted"]]
    assert picks and not any(V.shortcut(p["trace"]) for p in picks)  # the clean attempt of the task was taken
    assert rec["tasks_admissible_under_protocol_1"] >= len(rec["accepted"])
    old = L.Practice(tmp_path / "v1", 0, "practice", fams, tools, protocol=1)
    r1 = old.cycle(1, run, max_tasks=200)
    assert [x["rejection"] for x in s2.read_gz(tmp_path / "v1/attempts/c1_000.jsonl.gz")] == \
        [x["rejection_v1"] for x in s2.read_gz(tmp_path / "v2/attempts/c1_000.jsonl.gz")]
    assert r1["attempts_rejected_for_shortcut"] == 0
    assert L.Practice(PILOT1, 0, "practice", fams, tools).protocol == 1  # as it was run; never reinterpreted
    with pytest.raises(AssertionError):
        L.Practice(PILOT1, 0, "practice", fams, tools, protocol=2)
    u = L.Practice(tmp_path / "u", 0, "practice-unverified", fams, tools).cycle(1, run, max_tasks=200)
    urows = [x for x in s2.read_gz(tmp_path / "u/attempts/c1_000.jsonl.gz")]
    assert not any(x["well_formed"] and V.shortcut(x["trace"]) for x in urows) and len(u["accepted"]) > 0


def test_protocol_two_code_reads_no_sealed_split_and_training_sets_of_experiment_1_are_unchanged(practice):
    import re
    for src in ("ccl/verify.py", "ccl/adversarial.py", "scripts/s2_e2_stage2r.py"):
        text = (ROOT / src).read_text()
        if src.startswith("ccl/"):
            assert "sealed" not in text, src
        # the analysis script names sealed splits in a cost table; it opens no file of theirs and no test split
        assert "v2/sealed" not in text and "sealed/" not in text and not re.search(r"test_(comp|depth|tool|domain|id)", text), src
        assert set(re.findall(r"load_split\(([^)]*)\)", text)) <= {'"dev"'}, src
    assert not re.search(r"^\s*(import|from) (torch|transformers|peft)", (ROOT / "ccl/verify.py").read_text(), re.M)
    fams, tools = practice
    cfg = json.loads((ROOT / "configs/study02/e1/e1_s15_D25_r0.json").read_text())
    rows, stats = s2.oracle_rows(s2.e1_train_tasks(fams, tools, 0, 25))
    assert s2.rows_sha256(rows) == cfg["data"]["sha256"] and stats["rejected"] == 0
    # the adversarial set is built from dev families only, and no practice family is among them
    dev = {f["id"] for f in s2.read_gz(ROOT / "data/v2/dev/families.jsonl.gz")}
    assert {t.template for t in A.adversarial_dev(version=1)} <= dev and not dev & {f["id"] for f in fams}


# ---------------------------------------------------------------- the repeated pilot's audit and analysis

def test_the_repeated_pilots_audit_and_analysis_run_on_a_scripted_protocol_two_arm(practice, tmp_path):
    import importlib.util
    import shutil
    import types

    def load(name):
        spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    fams, tools = practice
    runs = tmp_path / "runs"

    def policy(skill, eliminate):
        """Solves tasks of up to `skill` hops as the oracle does (with a probe first, if `eliminate`); answers
        deeper ones at once and wrongly."""
        def one(t, a=0):
            if t.depth > skill:
                e = Episode(t, "scripted")
                e.attempt = a
                e.step(json.dumps({"final": "It is done.", "status": "done"}))
                return e
            if eliminate and a % 2 == 0:
                return eliminating(t, a)[0]
            e = Episode(t, "scripted")
            e.attempt = a
            while not e.done:
                e.step(OraclePolicy().act(e))
            return e
        return one

    def score(d, one):
        d.mkdir(parents=True)
        with open(d / "metrics.jsonl", "w") as f:
            for split, tasks in (("v2/dev", s2.load_split("dev")), ("v2/dev_adv", A.adversarial_dev(version=1))):
                for t in tasks:
                    f.write(json.dumps(s2.metrics_row(t, one(t), split, d.name)) + "\n")
    score(runs / "r0_seed_p2", policy(1, False))
    arm = L.Practice(runs / "r0_practice_p2", 0, "practice", fams, tools)
    for k, skill in ((1, 2), (2, 4)):
        one = policy(skill, True)
        rec = arm.cycle(k, lambda tasks, seed: [one(t, a) for t in tasks for a in range(L.ATTEMPTS)])
        score(runs / f"r0_practice_p2/c{k}", policy(skill, True))
        (runs / f"r0_practice_p2/c{k}/result.json").write_text(json.dumps({
            "rollout_seconds_this_job": 0.3 * rec["attempts"], "attempts_sampled_this_job": rec["attempts"], "seconds_per_attempt": 0.3,
            "steps": 100 + 120 * k, "train_seconds": 3.3 * (100 + 120 * k), "seconds_per_step": 3.3,
            "dev_seconds_this_job": 250.0, "adv_seconds_this_job": 150.0}))
    E = load("s2_e2")
    E.cmd_audit(types.SimpleNamespace(dir=str(runs / "r0_practice_p2"), replicate=0, arm="practice", sample=200))
    au = json.loads((runs / "r0_practice_p2/audit.json").read_text())
    assert au["protocol"] == 2 and au["blind"] == 20 and au["to_read"] == sum(bool(x["flags"]) or bool(x.get("blind")) for x in au["sample"])
    assert not any({"shortcut", "answer_parts"} & set(x["flags"]) for x in au["sample"])  # the rule kept them out
    M = load("s2_e2_stage2r")
    M.RUNS, M.OUT = runs, tmp_path / "out"
    M.OUT.mkdir()
    for f in ("stage1.json", "stage2.json"):
        shutil.copyfile(ROOT / "results/study02/e2" / f, M.OUT / f)
    M.main()
    res = json.loads((M.OUT / "stage2r.json").read_text())
    assert not res["gates"]["audit"] and not res["passed"]  # nothing has been read yet
    verdicts = {f"{x['cycle']}:{x['index']}": {"false_accept": False, "kind": "none", "note": "read"}
                for x in au["sample"] if x["flags"] or x.get("blind")}
    (runs / "r0_practice_p2/audit_verdicts.json").write_text(json.dumps(verdicts))
    M.main()
    res = json.loads((M.OUT / "stage2r.json").read_text())
    c1, c2 = res["cycles"]
    assert res["gates"] == {"yield": True, "audit": True, "cost": True, "dev": True} and res["passed"]
    assert res["buffers_free_of_the_shortcut"] and c1["buffer_shortcut"] == 0 and c1["attempts_using_the_shortcut"] > 0
    assert c1["tasks_admissible_under_protocol_1"] >= c1["accepted"] and c2["yield"] > c1["yield"]
    d = res["dev"]["after cycle 2"]
    assert d["success"]["all"] > d["success_clean"]["all"] and d["shortcut_successes"] > 0  # the scripted policy probes on dev
    assert res["dev"]["seed adapter"]["success"]["all"] == res["dev"]["seed adapter"]["success_clean"]["all"]
    a2 = res["adversarial"]["after cycle 2"]["all"]
    assert a2["tasks"] == 350 and res["adversarial"]["after cycle 2"]["complete"] and 0 < a2["adversarial_success_parts"] <= 1
    text = (M.OUT / "stage2r.md").read_text()
    assert "Pilot passed" in text and "adversarial twins" in text and "`dev`, three ways" in text
    k = next(iter(verdicts))
    verdicts[k] = {"false_accept": True, "kind": "(iii) a third form", "note": "x"}  # enough false accepts fail the gate
    for q in list(verdicts)[:12]:
        verdicts[q] = {"false_accept": True, "kind": "(iii) a third form", "note": "x"}
    (runs / "r0_practice_p2/audit_verdicts.json").write_text(json.dumps(verdicts))
    M.main()
    res = json.loads((M.OUT / "stage2r.json").read_text())
    assert res["audit"]["false_accept_rate"] > 0.02 and not res["gates"]["audit"] and "Pilot did not pass" in (M.OUT / "stage2r.md").read_text()
