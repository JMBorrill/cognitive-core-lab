"""Study 02, Experiment 2, Stage 0: the rollout engine, the practice loop and their isolation, on CPU."""
import json
import random
import re
from pathlib import Path

import pytest

from ccl import loop as L
from ccl import s2
from ccl.oracle import OraclePolicy
from ccl.runtime import Episode

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- engine (needs torch; tiny random model)

class Scripted:
    """A stand-in for an episode: the 'environment' answers with observations of varying length, sometimes
    rewrites what the model said (as stripping and re-tokenising can), and ends after a set number of turns."""

    def __init__(self, i: int, turns: int, attempt: int = 0):
        self.task = type("T", (), {"task_id": f"s{i}"})()
        self.i, self.turns, self.attempt, self.turn, self.done = i, turns, attempt, 0, False
        self.messages = [{"role": "system", "content": f"You are a tool user. Case {i}." + " Be exact." * (i % 4)},
                         {"role": "user", "content": f"Look up item {i} please."}]
        self.said = []

    def step(self, text, latency_s=0.0, gen_tokens=0):
        self.turn += 1
        self.said.append(text)
        kept = text[:-2] + "~" if (self.i + self.turn) % 3 == 0 and len(text) > 3 else text  # not what was generated
        self.messages.append({"role": "assistant", "content": kept})
        if self.turn >= self.turns:
            self.done = True
        else:
            self.messages.append({"role": "user", "content": "Observations:\n" + json.dumps(
                {"tool": "get_item", "result": {"n": list(range((self.i * 7 + self.turn * 3) % 9))}})})


@pytest.fixture(scope="module")
def tiny_policy(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from test_hf import episode, tiny as tiny_fixture  # the tiny model of the backend tests
    from ccl import hf
    d = tiny_fixture.__wrapped__(tmp_path_factory)
    out = tmp_path_factory.mktemp("e2run")
    hf.train({"model": d["path"], "train_files": [d["data"]], "steps": 4, "batch_size": 2, "grad_accum": 1,
              "lr": 5e-3, "warmup": 1, "seed": 1, "lora": {"rank": 4, "alpha": 8, "layers": None},
              "out_dir": str(out)}, fresh=True)
    return d["path"], out


def reference(model, tok, eps, max_tokens):
    """The same episodes advanced by the Experiment 1 policy, which re-reads every conversation in full."""
    from ccl import hf
    pol = hf.HFPolicy(model, tok, "ref", max_tokens=max_tokens)
    live = list(eps)
    while live:
        for e, (text, n) in zip(live, pol.generate([e.messages for e in live])):
            e.step(text, gen_tokens=n)
        live = [e for e in live if not e.done]
    return eps


def test_cached_decoding_equals_full_re_encoding_token_for_token(tiny_policy):
    from ccl import engine, hf
    path, adapter = tiny_policy
    model, tok = hf.load_policy_model(path, adapter, "cpu")
    make = lambda: [Scripted(i, turns=1 + i % 4) for i in range(9)]  # noqa: E731  mixed lengths, different ends
    for max_tokens in (12, 40):  # 40 runs past some end-of-turn tokens
        ref = reference(model, tok, make(), max_tokens)
        eng = engine.Engine(model, tok, "new", max_tokens=max_tokens)
        got = eng._run_group(make())
        alone = [eng._run_group([e])[0].said for e in make()]  # a row does not depend on its group
        assert [e.said for e in got] == [e.said for e in ref]
        assert [e.messages for e in got] == [e.messages for e in ref]
        assert eng.stats["refed_tokens"] > 0  # some cached tokens were replaced by the canonical ones
        assert alone == [e.said for e in ref]
    # the cache pays: far fewer tokens are read than by re-encoding every turn
    full = sum(len(hf.render_prompt_ids(tok, e.messages[:2 * t + 2])) for e in ref for t in range(e.turns))
    assert eng.stats["fed_tokens"] < 0.75 * 2 * full


def test_grouped_attention_equals_the_stock_function_and_is_what_decoding_uses(tiny_policy):
    import torch
    from transformers.integrations.sdpa_attention import sdpa_attention_forward as stock
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as fns
    from ccl import engine, hf
    torch.manual_seed(0)
    module = torch.nn.Module()
    module.num_key_value_groups = 6
    module.eval()
    b, kv, d, keys = 5, 2, 16, 37
    q, k, v = torch.randn(b, 12, 1, d), torch.randn(b, kv, keys, d), torch.randn(b, kv, keys, d)
    mask = torch.rand(b, 1, 1, keys) > 0.3  # padding anywhere in a row, as the engine has it
    mask[..., -1] = True
    seen = []
    mine = engine.grouped_attention(lambda *a, **kw: seen.append(1) or stock(*a, **kw))
    out, _ = mine(module, q, k, v, mask, scaling=d ** -0.5)
    ref, _ = stock(module, q, k, v, mask, scaling=d ** -0.5)
    assert not seen and out.shape == ref.shape and torch.allclose(out, ref, atol=1e-5)
    for args in ((q, k, v, None), (torch.randn(b, 12, 3, d), k, v, torch.ones(b, 1, 3, keys, dtype=torch.bool))):
        a, _ = mine(module, *args, scaling=d ** -0.5)  # no mask, or several query positions: the stock function
        assert torch.equal(a, stock(module, *args, scaling=d ** -0.5)[0])
    assert len(seen) == 2
    # the engine switches it on for its own forward passes only, and the outputs are those of the stock path
    path, adapter = tiny_policy
    model, tok = hf.load_policy_model(path, adapter, "cpu")
    make = lambda: [Scripted(i, turns=1 + i % 3) for i in range(6)]  # noqa: E731
    on = engine.Engine(model, tok, "on", max_tokens=20, grouped=True)
    off = engine.Engine(model, tok, "off", max_tokens=20)
    assert not off.grouped  # off unless asked for: it has not passed a calibration on a GPU
    before = fns["sdpa"]
    with engine._Grouped():
        assert fns["sdpa"] is not before
        a = on._run_group(make())
    assert fns["sdpa"] is before
    assert [e.said for e in a] == [e.said for e in off._run_group(make())]


def test_bfloat16_weights_load_and_sampling_is_seeded_per_task_attempt_and_turn(tiny_policy):
    import torch
    from ccl import engine
    path, adapter = tiny_policy
    model, tok = engine.load_policy_model(path, adapter, "cpu", dtype="float32")
    assert engine.load_policy_model(path, adapter, "cpu")[0].dtype == torch.bfloat16  # R1: cast once, after merging
    eng = engine.Engine(model, tok, "s", max_tokens=16, temperature=0.8, seed="k")
    a = eng._run_group([Scripted(i, 2, attempt=a) for i in range(4) for a in range(3)])
    b = eng._run_group([Scripted(i, 2, attempt=a) for i in (3, 1) for a in (2, 0)])  # other group, other order
    said = {(e.i, e.attempt): e.said for e in a}
    assert all(said[(e.i, e.attempt)] == e.said for e in b)
    assert len({tuple(said[(0, a)]) for a in range(3)}) > 1  # attempts of one task differ
    eng.seed = "other"
    assert [e.said for e in eng._run_group([Scripted(i, 2) for i in range(4)])] != [said[(i, 0)] for i in range(4)]
    assert engine.uniforms("k", "t", 0, 1, 5).tolist() != engine.uniforms("k", "t", 1, 1, 5).tolist()
    greedy = engine.Engine(model, tok, "g", max_tokens=16)
    assert [e.said for e in greedy._run_group([Scripted(0, 2)])] == [e.said for e in greedy._run_group([Scripted(0, 2)])]


def test_engine_runs_real_tasks_in_token_sized_groups_and_survives_out_of_memory(tiny_policy):
    import torch
    from ccl import engine, hf
    path, adapter = tiny_policy
    model, tok = hf.load_policy_model(path, adapter, "cpu")
    fams, tools = s2.load_practice()
    tasks = [t for t in L.Stream(fams, tools, 0).first(40) if t.depth == 1][:5]
    from ccl.runtime import render_system
    longest = max(len(hf.render_prompt_ids(tok, [{"role": "system", "content": render_system(t)},
                                                 {"role": "user", "content": t.request}])) for t in tasks)
    eng = engine.Engine(model, tok, "tiny", max_tokens=8, token_budget=3 * longest, turn_reserve=0)
    seen = []
    eps = eng.run(tasks, attempts=range(2), stop_on_error=True, on_done=seen.append)
    assert [(e.task.task_id, e.attempt) for e in eps] == [(t.task_id, a) for t in tasks for a in range(2)]
    assert len(seen) == 10 and eng.stats["groups"] >= 3  # the token budget, not a row count, sized the groups
    # R4: a random model's first turn is a parse error, the attempt stops there and can never be accepted
    assert all(e.trace.termination == "stopped" and len(e.trace.turns) == 1 for e in eps)
    assert all("not_success" in s2.rejection(e.task, e) and not L.well_formed(e.task, e) for e in eps)
    slow = engine.Engine(model, tok, "tiny", max_tokens=8)  # without R4 the same attempts run on
    assert all(len(e.trace.turns) == 3 for e in slow.run(tasks[:2]))
    real, sizes = eng._run_group, []

    def limited(group, stop):
        sizes.append(len(group))
        if len(group) > 1:
            raise torch.OutOfMemoryError("simulated")
        return real(group, stop)
    eng._run_group = limited
    again = eng.run(tasks[:2], attempts=range(2), stop_on_error=True)
    assert [e.trace.turns[0]["raw"] for e in again] == [e.trace.turns[0]["raw"] for e in eps[:4]]
    assert eng.stats["oom_splits"] >= 2 and max(sizes) > 1


def test_gradient_checkpointing_only_for_long_batches_gives_the_same_adapter(tiny_policy, tmp_path):
    from safetensors.torch import load_file
    import torch
    from ccl import hf
    path, _ = tiny_policy
    data = str(Path(path) / "train.jsonl")
    cfg = {"model": path, "train_files": [data], "steps": 5, "batch_size": 2, "grad_accum": 2, "lr": 5e-3,
           "warmup": 1, "seed": 3, "lora": {"rank": 4, "alpha": 8, "layers": None}}
    hf.train({**cfg, "out_dir": str(tmp_path / "off")}, fresh=True)
    hf.train({**cfg, "grad_checkpoint": "auto", "grad_checkpoint_tokens": 430, "out_dir": str(tmp_path / "auto")},
             fresh=True)
    a, b = (load_file(str(tmp_path / n / "adapters.safetensors")) for n in ("off", "auto"))
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)


# ---------------------------------------------------------------- loop (no tensor library)

@pytest.fixture(scope="module")
def practice():
    return s2.load_practice()


def scripted_run(skill: float):
    """A policy for loop tests: the oracle, which on a seeded share of attempts answers too early, calls
    a tool that does not exist, or makes an extra call. Deterministic per task, attempt and seed."""
    def run(tasks, seed):
        eps = []
        for t in tasks:
            for a in range(L.ATTEMPTS):
                rng = random.Random(f"{seed}|{t.task_id}|{a}")
                x = rng.random()
                e = Episode(t, "scripted")
                e.attempt = a
                if x > skill and x < skill + 0.1:
                    e.step(json.dumps({"calls": [{"tool": "no_such_tool", "args": {}}]}))
                elif x > skill:
                    e.step(json.dumps({"final": "It is done.", "status": "done"}))
                elif rng.random() < 0.3 and t.gold:  # a correct episode with one call repeated
                    e.step(OraclePolicy().act(e))
                    if not e.done:
                        e.step(e.trace.turns[0]["raw"])
                while not e.done:
                    e.step(OraclePolicy().act(e))
                eps.append(e)
        return eps
    return run


def test_seed_set_and_stream_are_practice_only_balanced_and_reproducible(practice):
    fams, tools = practice
    seed = L.seed_tasks(fams, tools, 0)
    assert len(seed) == 400 and len({t.template for t in seed}) == 50 and {t.depth for t in seed} == {1, 2}
    rows, stats = s2.oracle_rows(seed)
    assert stats == {"episodes": 400, "rejected": 0, "rows": len(rows)}
    a, b = L.Stream(fams, tools, 0), L.Stream(fams, tools, 0)
    first = a.first(L.PASS_TASKS + 40)  # into the second pass
    assert L.tasks_sha256(first) == L.tasks_sha256(b.first(L.PASS_TASKS + 40))
    assert L.tasks_sha256(first[:200]) != L.tasks_sha256(L.Stream(fams, tools, 1).first(200))
    assert not a.dropped  # every task passed the review's selection check
    ids = {f["id"] for f in fams}
    assert {t.template for t in first} <= ids and not {s.key for t in first for s in t.catalog} - set(tools)
    assert not {t.request for t in first} & {t.request for t in seed}  # new instances, not the seed's
    assert len({t.task_id for t in first}) == len(first)
    from collections import Counter
    for part in (first[:480], first[:1600]):  # every prefix has the mix of an Experiment 1 training subset
        mix = Counter(t.depth for t in part)
        assert all(0.15 <= mix[h] / len(part) <= 0.31 for h in (1, 2, 3, 4))
    d = L.diversity(a.family[:1600], first[:1600], L.seed_families(fams, 0))
    assert d["families"] == 1600 and d["outside_reference"]["deeper"] > 0.5 and 0 < d["mean_selections"] < 3


def test_acceptance_keeps_the_shortest_passing_attempt_and_replay_must_reproduce_it(practice):
    fams, tools = practice
    task = next(t for t in L.Stream(fams, tools, 0).first(60) if t.depth == 2 and "mod=none" in t.tags)
    eps = []
    for a, extra in enumerate((True, False, False)):
        e = Episode(task, "x")
        e.attempt = a
        e.step(OraclePolicy().act(e))
        if extra:
            e.step(e.trace.turns[0]["raw"])  # repeats its first call: still a success, one call longer
        while not e.done:
            e.step(OraclePolicy().act(e))
        eps.append(e)
    lazy = Episode(task, "x")
    lazy.attempt = 3
    lazy.step(json.dumps({"final": "All done.", "status": "done"}))
    rows = [L.attempt_row(7, e) for e in eps + [lazy]]
    assert [bool(r["rejection"]) for r in rows] == [False, False, False, True]
    assert L.choose(rows)["attempt"] == 1 and rows[0]["n_calls"] > rows[1]["n_calls"] and not rows[0]["strict"]
    assert L.choose(rows[3:]) is None and L.choose(rows[3:], verified=False)["attempt"] == 3  # well-formed, wrong
    assert {L.choose(rows, verified=False, key=str(k))["attempt"] for k in range(40)} == {0, 1, 2, 3}
    trace = json.loads(json.dumps(rows[1]["trace"]))  # as read back from a file
    ep, why = L.replay(task, trace)
    assert why == "ok" and not s2.rejection(task, ep)
    forged = json.loads(json.dumps(rows[3]["trace"]))
    assert L.replay(task, forged)[0] is None and L.replay(task, forged, verified=False)[1] == "ok"
    tampered = json.loads(json.dumps(trace))
    tampered["messages"][3]["content"] = tampered["messages"][3]["content"].replace("result", "resu1t")
    assert "observations differ" in L.replay(task, tampered)[1]  # what the model saw is not what the simulator says
    short = json.loads(json.dumps(trace))
    short["turns"] = short["turns"][:-1]
    assert L.replay(task, short)[0] is None


def test_practice_cycle_is_deterministic_resumable_and_stops_at_its_targets(practice, tmp_path):
    fams, tools = practice
    run = scripted_run(0.45)
    a = L.Practice(tmp_path / "a", 0, "practice", fams, tools)
    ra = a.cycle(1, run)
    assert len(ra["accepted"]) == L.CYCLE_ACCEPT and ra["offered"] == list(range(len(ra["offered"])))
    assert ra["offered"][-1] == ra["accepted"][-1]  # nothing after the last accepted task counts as offered
    assert 0.2 < ra["yield"] < 0.95 and ra["accepted_strict"] < ra["accepted_success"] == L.CYCLE_ACCEPT
    assert (tmp_path / "a/buffer/c1.jsonl.gz").exists() and len(a.state["cycles"]) == 1
    # an interrupted job: only the first chunk of attempts was written; the resumed cycle gives the same buffer
    b = L.Practice(tmp_path / "b", 0, "practice", fams, tools)
    calls = []

    def dies(tasks, seed):
        calls.append(len(tasks))
        if len(calls) == 2:
            raise KeyboardInterrupt
        return run(tasks, seed)
    with pytest.raises(KeyboardInterrupt):
        b.cycle(1, dies)
    assert not (tmp_path / "b/state.json").exists() and len(list((tmp_path / "b/attempts").glob("*.gz"))) == 1
    fresh = []
    rb = L.Practice(tmp_path / "b", 0, "practice", fams, tools).cycle(1, lambda t, s: fresh.append(len(t)) or run(t, s))
    assert rb["rows_sha256"] == ra["rows_sha256"] and rb["accepted"] == ra["accepted"]
    assert sum(fresh) + calls[0] >= len(ra["offered"]) > sum(fresh)  # the first chunk was read, not sampled again
    # a probe on 400 tasks stores no cycle, and the full cycle continues from its chunks
    c = L.Practice(tmp_path / "c", 0, "practice", fams, tools)
    probe = c.cycle(1, run, max_tasks=400)
    assert len(probe["offered"]) == 400 and not probe["complete"] and not c.state["cycles"]
    assert c.cycle(1, run)["rows_sha256"] == ra["rows_sha256"]
    # the next cycle goes on where this one stopped, with other samples
    r2 = a.cycle(2, run)
    assert r2["offered"][0] == ra["offered"][-1] + 1 and not set(r2["offered"]) & set(ra["offered"])
    with pytest.raises(AssertionError):
        a.cycle(2, run)
    # a weak policy runs into the task cap and carries on with what it has
    weak = L.Practice(tmp_path / "w", 0, "practice", fams, tools).cycle(1, scripted_run(0.02))
    assert len(weak["offered"]) == L.CYCLE_TASK_CAP and len(weak["accepted"]) < L.CYCLE_ACCEPT and weak["complete"]


def test_unverified_arm_keeps_well_formed_failures_and_curriculum_follows_measured_weakness(practice, tmp_path):
    fams, tools = practice
    run = scripted_run(0.45)
    u = L.Practice(tmp_path / "u", 0, "practice-unverified", fams, tools).cycle(1, run)
    assert len(u["accepted"]) == L.CYCLE_ACCEPT and u["accepted_success"] < L.CYCLE_ACCEPT
    v = L.Practice(tmp_path / "v", 0, "practice", fams, tools).cycle(1, run)
    assert len(u["offered"]) < len(v["offered"])  # without the verifier almost every task yields an episode
    bucket = L.buckets(fams)
    names = sorted(set(bucket.values()))
    from collections import Counter
    size = Counter(bucket.values())
    assert set(bucket) == {f["id"] for f in fams} and 15 <= len(names) <= 60
    group = {f["id"]: (L.shape_class(f), L.mod_class(f)) for f in fams}
    for n in names:  # a bucket under 30 families had no neighbour of its shape and modifier class to merge into
        kinds = {group[i] for i, b in bucket.items() if b == n}
        assert len(kinds) == 1 and (size[n] >= L.MIN_BUCKET or sum(1 for m in names if m.split("|")[1:] == n.split("|")[1:]) == 1)
    w = L.weights({"x": (5, 10), "y": (0, 40), "z": (40, 40)}, ["x", "y", "z", "new"])
    assert w["x"] > w["new"] - 1e-9 and w["x"] > w["y"] and w["x"] > w["z"] and min(w.values()) > 0.05
    assert w["new"] == pytest.approx(0.30)
    # cycle 1 has no history, so the curriculum arm's weights are flat; from cycle 2 they follow outcomes
    c = L.Practice(tmp_path / "c", 0, "practice+curriculum", fams, tools)

    def by_hops(tasks, seed):  # solves shallow tasks always, deep ones half the time, four-hop ones never
        out = []
        for e in scripted_run(1.0)(tasks, seed):
            p = {1: 1.0, 2: 1.0, 3: 0.5, 4: 0.0}[e.task.depth]
            if random.Random(f"{seed}|{e.task.task_id}").random() >= p:
                e = Episode(e.task, "scripted")
                e.attempt = len(out) % L.ATTEMPTS
                e.step(json.dumps({"final": "It is done.", "status": "done"}))
            out.append(e)
        return out
    c1 = c.cycle(1, by_hops)
    assert len(set(c1["offered"])) == len(c1["offered"]) and len(c1["accepted"]) == L.CYCLE_ACCEPT
    c2 = c.cycle(2, by_hops)
    assert not set(c2["offered"]) & set(c1["offered"])
    hops = lambda idx: Counter(c.stream.get(i).depth for i in idx)  # noqa: E731
    share3 = lambda idx: hops(idx)[3] / len(idx)  # noqa: E731
    assert share3(c2["offered"]) > share3(c1["offered"]) + 0.05  # towards what it solves sometimes
    again = L.Practice(tmp_path / "c2", 0, "practice+curriculum", fams, tools)
    again.cycle(1, by_hops)
    assert again.cycle(2, by_hops)["offered"] == c2["offered"]  # the draw is seeded


# ---------------------------------------------------------------- isolation

def test_no_sealed_input_is_reachable_from_the_loop_or_its_job_script():
    for src in ("ccl/loop.py", "ccl/engine.py", "scripts/s2_e2.py"):
        text = (ROOT / src).read_text()
        assert "sealed" not in text and "test_" not in text.replace("test_e2", ""), src
        assert set(re.findall(r"load_split\(([^)]*)\)", text)) <= {'"dev"'}, src
    for src in ("ccl/loop.py",):  # the loop itself needs no tensor library
        assert not re.search(r"^\s*(import|from) (torch|transformers|peft)", (ROOT / src).read_text(), re.M)
    fams, tools = s2.load_practice()
    sealed = {f["id"] for f in s2.read_gz(ROOT / "data/v2/sealed/families.jsonl.gz") if f["eval_split"] != "test_id"}
    dev = {f["id"] for f in s2.read_gz(ROOT / "data/v2/dev/families.jsonl.gz")}
    part = json.loads((ROOT / "data/v2/sealed/partition.json").read_text())
    held_edges = set(part["test_edges"]) | set(part["dev_edges"])
    st = L.Stream(fams, tools, 0)
    st.first(2 * L.PASS_TASKS)
    used = {f["id"] for f in st.family} | {f["id"] for f in L.seed_families(fams, 0)}
    assert not used & (sealed | dev)
    assert not {e for f in st.family for e in f["edges"]} & held_edges
