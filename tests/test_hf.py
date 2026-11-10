"""PyTorch backend tests on CPU with a tiny randomly initialised model (no weights or network needed)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from ccl import hf  # noqa: E402
from ccl.sft import batch_schedule, completion_ids, episode_sequence, pack_rows, render_prompt_ids  # noqa: E402

CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def episode(i: int, turns: int = 2) -> list[dict]:
    """SFT rows (one per assistant turn) of a small made-up episode."""
    msgs = [{"role": "system", "content": f"You are a tool user. Case {i}."},
            {"role": "user", "content": f"Look up item {i} please."}]
    rows = []
    for t in range(turns):
        done = t == turns - 1
        comp = json.dumps({"final": f"item {i} ok", "status": "done"} if done
                          else {"calls": [{"tool": "get_item", "args": {"item_id": f"IT-{1000 + i + t}"}}]})
        rows.append({"messages": [dict(m) for m in msgs], "completion": comp,
                     "meta": {"task_id": f"t{i}", "source": "oracle", "turn": t + 1, "family": "x", "template": "x"}})
        msgs += [{"role": "assistant", "content": comp},
                 {"role": "user", "content": f'Observations:\n{{"tool": "get_item", "result": {{"n": {i + t}}}}}'}]
    return rows


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A 3-block Qwen2 model with a character-level ChatML tokenizer, saved like a Hub snapshot."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    d = tmp_path_factory.mktemp("tiny")
    # byte-level alphabet, as in the real Qwen tokenizer: a space is "\u0120" and a newline "\u010a"
    chars = [chr(c) for c in range(33, 127)] + ["\u0120", "\u010a"]
    vocab = {"<unk>": 0, "<|endoftext|>": 1, "<|im_start|>": 2, "<|im_end|>": 3}
    vocab.update({c: i + 4 for i, c in enumerate(chars)})
    core = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="<unk>", eos_token="<|im_end|>",
                                  pad_token="<|endoftext|>",
                                  additional_special_tokens=["<|im_start|>", "<|im_end|>", "<|endoftext|>"])
    tok.chat_template = CHATML
    tok.save_pretrained(d)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      eos_token_id=3, pad_token_id=1, bos_token_id=None)
    Qwen2ForCausalLM(cfg).save_pretrained(d)
    data = d / "train.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for i in range(12) for r in episode(i, 2 + i % 2)))
    return {"path": str(d), "data": str(data)}


def test_prompt_tokenisation_round_trips_and_packing_is_prefix_consistent(tiny):
    _, tok = hf.load_base(tiny["path"], "cpu")
    rows = episode(1, 3)
    ids = render_prompt_ids(tok, rows[0]["messages"])
    assert isinstance(ids, list) and tok.decode(ids).endswith("<|im_start|>assistant\n")
    seq, mask = episode_sequence(tok, rows[-1])
    assert len(seq) == len(mask)
    # the supervised tokens are exactly the assistant completions with their end-of-turn marker
    spans, cur = [], []
    for t, m in zip(seq, mask):
        if m:
            cur.append(t)
        elif cur:
            spans.append(cur)
            cur = []
    spans += [cur] if cur else []
    assert spans == [completion_ids(tok, r["completion"]) for r in rows]
    assert [tok.decode(s) for s in spans] == [r["completion"] + "<|im_end|>" for r in rows]


def test_loss_covers_assistant_tokens_only_and_packed_equals_per_turn(tiny):
    model, tok = hf.load_base(tiny["path"], "cpu")
    rows = episode(2, 3)
    packed, dropped = pack_rows(rows, tok, 4096)
    assert dropped == 0 and len(packed) == 1
    b, m = hf.collate(packed, 4096, "cpu")
    loss, n = hf.loss_fn(model, b, m)
    total, count = 0.0, 0
    for r in rows:  # one teacher-forced example per assistant turn
        p, c = render_prompt_ids(tok, r["messages"]), completion_ids(tok, r["completion"])
        bb, mm = hf.collate([(p + c, [0] * len(p) + [1] * len(c))], 4096, "cpu")
        l, k = hf.loss_fn(model, bb, mm)
        total, count = total + l.item() * k, count + k
    assert n == count == sum(len(completion_ids(tok, r["completion"])) for r in rows)
    assert loss.item() == pytest.approx(total / count, rel=1e-5)
    # the loss does not depend on what the prompt positions would have predicted
    logits = model(input_ids=b[:, :-1]).logits[0]
    manual = torch.nn.functional.cross_entropy(logits[m[0, 1:]], b[0, 1:][m[0, 1:]])
    assert loss.item() == pytest.approx(manual.item(), rel=1e-5)


def test_padding_in_a_batch_does_not_change_the_loss(tiny):
    model, tok = hf.load_base(tiny["path"], "cpu")
    ex = pack_rows(episode(3, 2) + episode(4, 3), tok, 4096)[0]
    both, n = hf.loss_fn(model, *hf.collate(ex, 4096, "cpu"))
    parts = [hf.loss_fn(model, *hf.collate([e], 4096, "cpu")) for e in ex]
    assert both.item() == pytest.approx(sum(l.item() * k for l, k in parts) / n, rel=1e-5)


def test_adam_matches_reference_updates_with_and_without_bias_correction():
    rng = np.random.default_rng(0)
    w0, grads = rng.normal(size=5), rng.normal(size=(6, 5))
    for bc in (False, True):
        p = torch.nn.Parameter(torch.tensor(w0))
        opt = hf.Adam([p], bias_correction=bc)
        w, m, v = w0.copy(), np.zeros(5), np.zeros(5)
        for t, g in enumerate(grads, 1):
            p.grad = torch.tensor(g)
            opt.step(lr=1e-2)
            m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
            if bc:
                w = w - 1e-2 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            else:  # MLX: p - lr * m / (sqrt(v) + eps)
                w = w - 1e-2 * m / (np.sqrt(v) + 1e-8)
            assert np.allclose(p.detach().numpy(), w, atol=1e-12)
    # with bias correction it is PyTorch's own AdamW
    a, b = torch.nn.Parameter(torch.tensor(w0)), torch.nn.Parameter(torch.tensor(w0))
    ours, ref = hf.Adam([a], bias_correction=True), torch.optim.AdamW([b], lr=1e-2, weight_decay=0.0)
    for g in grads:
        a.grad, b.grad = torch.tensor(g), torch.tensor(g)
        ours.step(lr=1e-2)
        ref.step()
    assert torch.allclose(a, b, atol=1e-12)


def test_learning_rate_schedule_is_the_study01_schedule():
    lr, warm, total = 1e-5, 20, 250
    assert hf.lr_at(0, lr, warm, total) == pytest.approx(1e-7)
    assert hf.lr_at(warm, lr, warm, total) == pytest.approx(lr)
    assert hf.lr_at(total, lr, warm, total) == pytest.approx(lr * 0.1)
    for k in (25, 100, 249):  # mlx.optimizers.cosine_decay(lr, total - warm, 0.1 * lr) after the warm-up
        want = 0.1 * lr + 0.9 * lr * 0.5 * (1 + math.cos(math.pi * (k - warm) / (total - warm)))
        assert hf.lr_at(k, lr, warm, total) == pytest.approx(want)
    assert all(hf.lr_at(k, lr, warm, total) <= lr for k in range(total + 1))


def test_study01_lora_layout_has_the_reported_number_of_parameters():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=64, hidden_size=896, intermediate_size=4864, num_hidden_layers=24,
                      num_attention_heads=14, num_key_value_heads=2)  # Qwen2.5-0.5B blocks, small vocabulary
    model = hf.add_lora(Qwen2ForCausalLM(cfg), {"rank": 16, "alpha": 320, "dropout": 0.0, "layers": list(range(8, 24))}, 0)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 5_865_472
    assert all("lora_" in n for n in names) and {int(n.split(".")[2]) for n in names} == set(range(8, 24))
    layer = model.model.layers[8].self_attn.q_proj
    assert layer.scaling["default"] == pytest.approx(20.0)  # alpha / rank = MLX-LM's scale
    assert float(layer.lora_B["default"].weight.detach().abs().sum()) == 0.0  # the update starts at zero
    full = hf.add_lora(Qwen2ForCausalLM(cfg), {"rank": 16, "alpha": 32, "dropout": 0.0, "layers": None}, 0)
    assert sum(p.numel() for p in full.parameters() if p.requires_grad) == 5_865_472 * 24 // 16


def test_training_is_deterministic_and_resumes_exactly(tiny, tmp_path):
    from safetensors.torch import load_file
    cfg = {"model": tiny["path"], "train_files": [tiny["data"]], "val_file": tiny["data"], "steps": 6,
           "batch_size": 2, "grad_accum": 2, "lr": 5e-3, "warmup": 2, "save_every": 100, "eval_every": 3, "seed": 5,
           "lora": {"rank": 4, "alpha": 8, "layers": None}}
    full = hf.train({**cfg, "out_dir": str(tmp_path / "full")}, fresh=True)
    again = hf.train({**cfg, "out_dir": str(tmp_path / "again")}, fresh=True)
    assert full["adapter_sha256"] == again["adapter_sha256"]
    assert hf.train({**cfg, "out_dir": str(tmp_path / "res"), "stop_after": 3}, fresh=True) == {"interrupted_at": 3}
    resumed = hf.train({**cfg, "out_dir": str(tmp_path / "res")})
    assert resumed["resumed_from"] == 3
    a, b = load_file(str(tmp_path / "full/adapters.safetensors")), load_file(str(tmp_path / "res/adapters.safetensors"))
    assert set(a) == set(b) and all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)
    log = [json.loads(l) for l in open(tmp_path / "full/train_log.jsonl")]
    assert len(log) == 6 and log[-1]["loss"] < log[0]["loss"] and "val_loss" in log[2] and log[0]["grad_norm"] > 0
    assert (tmp_path / "full/DONE").exists() and full["trainable_params"] > 0
    # a finished run is not trained again
    assert hf.train({**cfg, "out_dir": str(tmp_path / "full")})["adapter_sha256"] == full["adapter_sha256"]
    other = hf.train({**cfg, "seed": 6, "out_dir": str(tmp_path / "other")}, fresh=True)
    assert other["adapter_sha256"] != full["adapter_sha256"]


def test_batch_schedule_matches_the_study01_schedule():
    a, b = batch_schedule(10, 4, 6, seed=3), batch_schedule(10, 4, 6, seed=3)
    assert all((x == y).all() for x, y in zip(a, b))
    assert sorted(np.concatenate(a)[:10].tolist()) == list(range(10))


def test_batched_greedy_generation_equals_unbatched_and_ignores_model_sampling_defaults(tiny, tmp_path):
    cfg = {"model": tiny["path"], "train_files": [tiny["data"]], "steps": 4, "batch_size": 2, "grad_accum": 1,
           "lr": 5e-3, "warmup": 1, "seed": 1, "lora": {"rank": 4, "alpha": 8, "layers": None},
           "out_dir": str(tmp_path / "run")}
    hf.train(cfg, fresh=True)
    model, tok = hf.load_policy_model(tiny["path"], tmp_path / "run", "cpu")
    base, _ = hf.load_policy_model(tiny["path"], None, "cpu")
    # the adapter is merged into the weights for inference; that is the same function as the unmerged model
    raw, _ = hf.load_base(tiny["path"], "cpu")
    raw = hf.add_lora(raw, json.loads((tmp_path / "run/adapter_config.json").read_text())["lora"], 0)
    hf.load_adapter(raw, tmp_path / "run/adapters.safetensors")
    ids = torch.tensor([render_prompt_ids(tok, episode(0, 2)[1]["messages"])])
    with torch.no_grad():
        assert torch.allclose(raw.eval()(input_ids=ids).logits, model(input_ids=ids).logits, atol=1e-5)
        assert not torch.allclose(base(input_ids=ids).logits, model(input_ids=ids).logits, atol=1e-3)
    assert hf.merge_lora(raw) > 0 and hf.merge_lora(raw) == 0
    convs = [episode(i, 2)[t]["messages"] for i in range(5) for t in range(2)]  # prompts of different lengths
    pol = hf.HFPolicy(model, tok, "tiny", max_tokens=12)
    together = pol.generate(convs)
    alone = [pol.generate([c])[0] for c in convs]
    assert together == alone and all(n <= 12 for _, n in together)
    assert together != hf.HFPolicy(base, tok, "base", max_tokens=12).generate(convs)  # the adapter is in use
    model.generation_config.do_sample = True
    model.generation_config.repetition_penalty = 1.8
    model.generation_config.temperature = 1.5
    assert pol.generate(convs) == together
    assert pol.stats["calls"] == 2 + len(convs) and pol.stats["prompt_tokens"] > 0


MODEL = Path(__file__).resolve().parents[1] / "models/Qwen2.5-0.5B-Instruct"
VARIANT = Path(__file__).resolve().parents[1] / "data/v1/sft/variant_C.jsonl"


@pytest.mark.slow
@pytest.mark.skipif(not MODEL.exists() or not VARIANT.exists(), reason="base weights or rebuilt v1 data not present")
def test_first_training_step_matches_the_study01_log():
    """With the real tokenizer and weights, the port sees the same tokens in the same batches as the MLX
    trainer did, and computes the same loss before any update (the LoRA update starts at zero)."""
    from ccl.sft import load_examples
    root = MODEL.parents[1]
    model, tok = hf.load_base(str(MODEL), "cpu")
    assert model.generation_config.do_sample and model.generation_config.repetition_penalty > 1  # hence the override
    exs, dropped = load_examples(str(VARIANT), tok, 2048)
    log = [json.loads(l) for l in open(root / "runs/C_s0/train_log.jsonl")]
    micro = batch_schedule(len(exs), 4, 500, 0)
    assert dropped == 0 and len(exs) == 1000
    for s in range(250):
        idx = [i for b in micro[2 * s:2 * s + 2] for i in b]
        assert sum(len(exs[i][0]) for i in idx) == log[s]["seq_tokens"]
        assert sum(sum(exs[i][1]) for i in idx) == log[s]["supervised_tokens"]
    tot = n = 0
    with torch.no_grad():
        for b in micro[:2]:
            l, k = hf.loss_fn(model, *hf.collate([exs[i] for i in b], 2048, "cpu"))
            tot, n = tot + l.item() * k, n + k
    assert tot / n == pytest.approx(log[0]["loss"], abs=5e-3)
    lora = hf.add_lora(model, {"rank": 16, "alpha": 320, "dropout": 0.0, "layers": list(range(8, 24))}, 0)
    assert sum(p.numel() for p in lora.parameters() if p.requires_grad) == 5_865_472
