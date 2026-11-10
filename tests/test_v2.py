"""Study 02 environment: schema, family identity, split isolation, determinism and solvability."""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from ccl import compose as C
from ccl import s2
from ccl.data import rejection_reasons
from ccl.evaluate import episode_metrics, validate
from ccl.oracle import OraclePolicy
from ccl.runtime import Episode, run_batch
from ccl.tasks import TEMPLATES
from ccl.tools import MUTATING, TOOLS
from ccl.world2 import BACKUP_OF, ENTITIES, TOOLS2

ROOT = Path(__file__).resolve().parents[1]
V1_PREFIXES = {"C", "CU", "ORD", "F", "EV", "MSG", "RF", "TK", "BK", "RS", "FL", "SKU"}


@pytest.fixture(scope="module")
def space():
    return C.build_space()


def sample(fams, n):
    return fams[:: max(1, len(fams) // n)]


def edges_of(task):
    by = {g.id: g for g in task.gold}
    return {f"{by[d].tool}>{g.tool}" for g in task.gold for d in g.deps}


# ---------------------------------------------------------------- schema

def test_schema_shares_nothing_with_study01():
    assert not set(TOOLS2) & set(TOOLS)
    assert not {e.prefix for e in ENTITIES.values()} & V1_PREFIXES
    assert len({e.prefix for e in ENTITIES.values()}) == len(ENTITIES)
    assert not {e.domain for e in ENTITIES.values()} & {t.domain for t in TOOLS.values()}
    assert {k for k, t in TOOLS2.items() if t.kind == "action"} <= MUTATING
    assert set(TEMPLATES)  # v1 templates are untouched by importing the v2 tool table


def test_phrases_are_right_branching_and_values_unambiguous():
    for e in ENTITIES.values():
        for r in e.refs:
            assert len(set(r.phrases)) == 3 and all(p.endswith("{x}") and p.count("{x}") == 1 for p in r.phrases)
        for a in e.actions:
            assert len(set(a.phrases)) == 3 and all(p.endswith("{x}.") for p in a.phrases)
            for p in a.phrases:
                assert set(re.findall(r"{(\w+)}", p)) == {"x"} | {n for n, _ in a.params}
        for a in e.attrs:
            if a.kind[0] == "enum":  # a fact check must not be satisfied by a different value
                vals = a.kind[1]
                assert all(len(v) >= 3 and not any(v != u and v in u for u in vals) for v in vals), (e.name, a.name)


# ---------------------------------------------------------------- splits

def test_every_family_is_in_exactly_one_split_and_the_space_is_large_enough(space):
    fams = space["families"]
    ids = [f["id"] for fs in fams.values() for f in fs]
    sigs = [f["signature"] for fs in fams.values() for f in fs]
    assert len(ids) == len(set(ids)) == len(set(sigs))
    assert len(fams["practice"]) >= 5000
    for split, need in (("test_comp", 600), ("test_depth", 200), ("test_tool", 200), ("test_domain", 200)):
        assert len(fams[split]) >= need


def test_split_definitions_hold_for_every_family(space):
    part, fams = space["partition"], space["families"]
    practice_tools, practice_edges = set(part["practice_tools"]), set(part["practice_edges"])
    dom = lambda f: {TOOLS2[t].domain for t in f["tools"]}  # noqa: E731
    for f in fams["practice"]:
        assert set(f["tools"]) <= practice_tools and set(f["edges"]) <= practice_edges and 1 <= f["hops"] <= 4
    for f in fams["test_comp"]:
        assert set(f["tools"]) <= practice_tools and set(f["edges"]) & set(part["test_edges"])
        assert not set(f["edges"]) & set(part["dev_edges"]) and 2 <= f["hops"] <= 4
    for f in fams["test_depth"]:
        assert set(f["tools"]) <= practice_tools and set(f["edges"]) <= practice_edges and f["hops"] in (5, 6)
    for f in fams["test_tool"]:
        assert set(f["tools"]) & set(part["test_tools"]) and not set(f["tools"]) & set(part["dev_tools"])
        assert not set(f["edges"]) & (set(part["test_edges"]) | set(part["dev_edges"]))
    for f in fams["test_domain"]:
        assert dom(f) <= set(part["test_domains"])
    for s in ("dev_comp", "dev_tool", "dev_domain"):  # dev holds out its own edges, tools and domain
        for f in fams[s]:
            assert not set(f["tools"]) & set(part["test_tools"]) and not set(f["edges"]) & set(part["test_edges"])
            assert not dom(f) & set(part["test_domains"])
    for f in fams["dev_comp"]:
        assert set(f["edges"]) & set(part["dev_edges"]) and set(f["tools"]) <= practice_tools


def test_held_out_edges_join_tools_that_are_practised_elsewhere(space):
    part = space["partition"]
    src = Counter(e.split(">")[0] for e in part["practice_edges"])
    dst = Counter(e.split(">")[1] for e in part["practice_edges"])
    for e in part["test_edges"] + part["dev_edges"]:
        s, t = e.split(">")
        assert src[s] >= 1 and dst[t] >= 1, e
    used = {t for f in space["families"]["practice"] for t in f["tools"]}
    used |= {BACKUP_OF[t] for t in used if t in BACKUP_OF}
    assert used == set(part["practice_tools"])  # every practice tool is needed by some practice family


# ---------------------------------------------------------------- family identity

def test_tasks_of_a_family_share_its_graph_and_families_differ_in_graph_or_behaviour(space):
    fams = sample(space["families"]["practice"], 150) + sample(space["families"]["test_comp"], 80)
    for f in fams:
        sigs = {C.family_signature(C.make_task(f, i, "train", "x", space["partition"]["practice_tools"]))
                for i in range(4)}
        assert sigs == {f["signature"]}
    # distinct families never differ only by wording, values or the order of independent calls
    sig = Counter(f["signature"] for fs in space["families"].values() for f in fs)
    assert max(sig.values()) == 1
    retry = next(f for f in space["families"]["practice"] if f["mod"] == "retry" and f["hops"] == 1)
    down = next(f for f in space["families"]["practice"] if f["mod"] == "fallback" and f["hops"] == 1)
    assert "retry:" in retry["signature"] and "down:" in down["signature"]


def test_one_fault_variant_per_base_graph(space):
    for mod in ("retry", "fallback"):
        per_graph = Counter(g for f in space["families"]["practice"] if f["mod"] == mod for g in f["graphs"])
        assert max(per_graph.values()) == 1


# ---------------------------------------------------------------- generation

def test_generation_is_deterministic_and_pools_are_disjoint(space):
    tools = space["partition"]["practice_tools"]
    for f in sample(space["families"]["practice"], 120):
        a, b = C.make_task(f, 3, "train", "practice", tools), C.make_task(f, 3, "train", "practice", tools)
        assert a.to_dict() == b.to_dict()
        ev = C.make_task(f, 3, "eval", "test_id", tools)
        ids = lambda t: {k for tab in t.world.values() if isinstance(tab, dict) for k in tab}  # noqa: E731
        assert not ids(a) & ids(ev)
        assert all(1000 <= int(i.split("-")[1]) < 6000 for i in ids(a))
        assert all(6000 <= int(i.split("-")[1]) < 10000 for i in ids(ev))


def test_oracle_solves_every_sampled_task_and_the_world_is_consistent(space):
    part = space["partition"]
    for split, fams in space["families"].items():
        pool = sorted(TOOLS2) if split != "practice" else part["practice_tools"]
        tasks = [C.make_task(f, i, "eval", split, pool) for f in sample(fams, 120) for i in (0, 1)]
        for e in run_batch(tasks, OraclePolicy(), batch_size=512):
            assert validate(e.task, e)["success"], (split, e.task.request)
            assert not s2.rejection(e.task, e)
            assert s2.dependency_safe(e.task, e), e.task.request  # every ID the oracle uses was observed
            m = episode_metrics(e.task, e)
            assert m["unnecessary_calls"] == 0
            if len({g.tool for g in e.task.gold}) == len(e.task.gold):
                assert not rejection_reasons(e.task, e)  # Study 01's stricter filter agrees when tools are unique


def test_acceptance_rule_uses_only_outcome_and_runtime_observations(space):
    f = next(f for f in space["families"]["practice"] if f["mod"] == "none" and f["hops"] == 2
             and f["shape"] == "chain")
    task = C.make_task(f, 0, "train", "practice", space["partition"]["practice_tools"])
    good = Episode(task, "t")
    while not good.done:
        good.step(OraclePolicy().act(good))
    assert s2.rejection(task, good) == []
    bad = Episode(task, "t")
    bad.step(json.dumps({"calls": [{"tool": "no_such_tool", "args": {}}]}))
    while not bad.done:
        bad.step(OraclePolicy().act(bad))
    assert validate(task, bad)["success"] and s2.rejection(task, bad) == ["unknown_tool"]
    lazy = Episode(task, "t")
    lazy.step(json.dumps({"final": "Done.", "status": "done"}))
    assert "not_success" in s2.rejection(task, lazy)


# ---------------------------------------------------------------- Experiment 1 subsets

def test_e1_subsets_are_nested_balanced_and_drawn_from_practice_only():
    fams, tools = s2.load_practice()
    by_id = {f["id"]: f for f in fams}
    sealed = {f["id"] for f in s2.read_gz(ROOT / "data/v2/sealed/families.jsonl.gz") if f["eval_split"] != "test_id"}
    part = json.loads((ROOT / "data/v2/sealed/partition.json").read_text())
    orders = [s2.e1_order(fams, r) for r in (0, 1)]
    assert [f["id"] for f in orders[0][:1600]] != [f["id"] for f in orders[1][:1600]]
    for order in orders:
        assert len({f["id"] for f in order}) == len(order) and not {f["id"] for f in order} & sealed
        mix = {}
        for d in (25, 100, 400, 1600):
            sub = order[:d]
            assert all(f["id"] in by_id for f in sub)
            mix[d] = Counter(f["hops"] for f in sub)
            assert max(mix[d].values()) / d <= 0.30 and min(mix[d].values()) / d >= 0.15
        for d in (100, 400):  # the mix of depth is the same at every level
            assert all(abs(mix[d][h] / d - mix[1600][h] / 1600) < 0.02 for h in (1, 2, 3, 4))
    tasks = s2.e1_train_tasks(fams, tools, 0, 100)
    assert len(tasks) == 3200 and len({t.template for t in tasks}) == 100
    assert not {s.key for t in tasks for s in t.catalog} - set(tools)
    assert not {e for t in tasks for e in edges_of(t)} & (set(part["test_edges"]) | set(part["dev_edges"]))
    own = s2.e1_own_tasks(fams, tools, 0, 100)
    assert len(own) == 400 and {t.template for t in own} == {t.template for t in tasks}
    assert not {t.request for t in own} & {t.request for t in tasks}


# ---------------------------------------------------------------- committed files and isolation

def test_committed_v2_files_rebuild_byte_for_byte(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts/build_v2.py"), "--out", str(tmp_path / "v2")],
                   cwd=ROOT, check=True, capture_output=True)
    committed = json.loads((ROOT / "data/v2/manifest.json").read_text())
    rebuilt = json.loads((tmp_path / "v2/manifest.json").read_text())
    assert rebuilt["files"] == committed["files"] and rebuilt["census"] == committed["census"]


def test_no_held_out_structure_reaches_training_inputs():
    fams, tools = s2.load_practice()
    part = json.loads((ROOT / "data/v2/sealed/partition.json").read_text())
    sealed = s2.read_gz(ROOT / "data/v2/sealed/families.jsonl.gz")
    dev = s2.read_gz(ROOT / "data/v2/dev/families.jsonl.gz")
    held_tools = set(part["test_tools"]) | set(part["dev_tools"])
    held_edges = set(part["test_edges"]) | set(part["dev_edges"])
    held_domains = set(part["test_domains"]) | set(part["dev_domains"])
    assert not set(tools) & held_tools and not {TOOLS2[t].domain for t in tools} & held_domains
    practice_ids = {f["id"] for f in fams}
    assert not practice_ids & {f["id"] for f in dev}
    assert not practice_ids & {f["id"] for f in sealed if f["eval_split"] != "test_id"}
    assert {f["id"] for f in sealed if f["eval_split"] == "test_id"} <= practice_ids
    for f in fams:
        assert not set(f["tools"]) & held_tools and not set(f["edges"]) & held_edges
    for split in ("test_comp", "test_depth", "test_tool", "test_domain", "test_id", "dev"):
        tasks = s2.load_split(split)
        assert all(6000 <= int(k.split("-")[1]) for t in tasks for tab in t.world.values()
                   if isinstance(tab, dict) for k in tab)
    # training code cannot name the sealed directory or load a test split
    for src in ("scripts/s2_train.py", "ccl/hf.py", "ccl/sft.py"):
        text = (ROOT / src).read_text()
        assert "sealed" not in text, src
        assert set(re.findall(r"load_split\(([^)]*)\)", text)) <= {'"dev"'}, src
