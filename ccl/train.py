"""Resumable LoRA fine-tuning on MLX.

Uses mlx-lm's LoRA layers and masked cross-entropy, with a small custom loop so
that (a) batch order is a deterministic function of the seed, and (b) a run can
be resumed from its latest checkpoint (adapter + optimizer state + step).

Usage: python -m ccl.train configs/train/<name>.json [--fresh]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .models import completion_ids, render_prompt_ids

DEFAULTS = {
    "model": "models/Qwen3.5-0.8B",
    "train_files": [],  # list of {"path": ..., "weight": int} or str
    "val_file": None,
    "stages": None,  # optional sequential curriculum: [{"train_files": [...], "steps": N}, ...]
    "out_dir": "runs/debug",
    "seed": 0,
    "steps": 250,
    "batch_size": 4,
    "grad_accum": 2,
    "lr": 1e-4,
    "warmup": 30,
    "weight_decay": 0.0,
    "max_seq_len": 2048,
    "lora": {"rank": 16, "scale": 20.0, "dropout": 0.0},
    "num_layers": -1,
    "grad_checkpoint": True,
    "save_every": 100,
    "eval_every": 100,
    "val_batches": 20,
    "memory_limit_gb": 11.0,
}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _episode_sequence(tokenizer, ex: dict) -> tuple[list[int], list[int]] | None:
    """Pack all assistant turns of one episode into a single sequence with a loss mask.

    Equivalent to one example per turn under teacher forcing, but the shared
    prefix is processed once. Returns None if the chat template does not render
    history prefix-consistently (caller then falls back to per-turn examples)."""
    msgs = ex["messages"] + [{"role": "assistant", "content": ex["completion"]}]
    seq, mask = [], []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        p = render_prompt_ids(tokenizer, msgs[:i])
        if p[: len(seq)] != seq:
            return None
        mask += [0] * (len(p) - len(seq))
        seq = list(p)
        c = completion_ids(tokenizer, m["content"])
        seq += c
        mask += [1] * len(c)
    return seq, mask


def load_examples(path: str, tokenizer, max_len: int) -> tuple[list[tuple[list[int], list[int]]], int]:
    """Return [(tokens, loss_mask)] (one per episode) and the number dropped for exceeding max_len.

    SFT files hold one row per assistant turn; only the last turn of each episode is
    needed because its `messages` contain the full canonical history."""
    rows = [json.loads(l) for l in open(path)]
    last = {}
    for r in rows:
        key = (r["meta"]["task_id"], r["meta"]["source"], r["meta"].get("sample", 0))
        if key not in last or r["meta"]["turn"] > last[key]["meta"]["turn"]:
            last[key] = r
    out, dropped = [], 0
    for r in last.values():
        packed = _episode_sequence(tokenizer, r)
        seqs = [packed] if packed else []
        if packed is None:  # template not prefix-consistent: per-turn fallback
            for q in rows:
                if q["meta"]["task_id"] == r["meta"]["task_id"] and q["meta"]["source"] == r["meta"]["source"]:
                    p = render_prompt_ids(tokenizer, q["messages"])
                    c = completion_ids(tokenizer, q["completion"])
                    seqs.append((p + c, [0] * len(p) + [1] * len(c)))
        for t, m in seqs:
            if len(t) > max_len:
                dropped += 1
            else:
                out.append((t, m))
    return out, dropped


def batch_schedule(n_examples: int, batch_size: int, steps: int, seed: int) -> list[np.ndarray]:
    """Deterministic list of index batches covering `steps` micro-batches (epoch-wise reshuffle)."""
    rng = np.random.RandomState(seed)
    out, perm = [], np.array([], dtype=int)
    while len(out) < steps:
        if len(perm) < batch_size:
            perm = np.concatenate([perm, rng.permutation(n_examples)])
        out.append(perm[:batch_size])
        perm = perm[batch_size:]
    return out


def collate(examples, max_len: int):
    lengths = [min(len(t), max_len) for t, _ in examples]
    width = 32 * ((max(lengths) + 31) // 32) + 1
    toks = np.zeros((len(examples), width), np.int32)
    mask = np.zeros((len(examples), width), np.float32)
    for i, (t, m) in enumerate(examples):
        toks[i, : lengths[i]] = t[: lengths[i]]
        mask[i, : lengths[i]] = m[: lengths[i]]
    return mx.array(toks), mx.array(mask)


def loss_fn(model, toks, mask):
    """Mean cross-entropy over supervised (assistant) tokens."""
    inputs, targets, m = toks[:, :-1], toks[:, 1:], mask[:, 1:]
    logits = model(inputs)
    ce = nn.losses.cross_entropy(logits, targets) * m
    ntoks = m.sum()
    return ce.astype(mx.float32).sum() / ntoks, ntoks


def _latest_ckpt(out: Path) -> Path | None:
    cks = sorted(out.glob("checkpoints/step_*"), key=lambda p: int(p.name.split("_")[1]))
    return cks[-1] if cks else None


def train(cfg: dict, fresh: bool = False) -> dict:
    from mlx_lm import load
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers

    cfg = {**DEFAULTS, **cfg}
    out = Path(cfg["out_dir"])
    if fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        print(f"[train] {out} already complete; skipping")
        return json.loads((out / "summary.json").read_text())

    mx.random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if cfg.get("memory_limit_gb"):
        mx.set_memory_limit(int(cfg["memory_limit_gb"] * 1e9))

    model, tok = load(cfg["model"])
    model.freeze()
    n_layers = len(model.layers) if cfg["num_layers"] < 0 else cfg["num_layers"]
    linear_to_lora_layers(model, n_layers, cfg["lora"])
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    if cfg["grad_checkpoint"]:
        grad_checkpoint(model.layers[0])

    # ---- data (stages allow sequential curricula; a single stage otherwise)
    stages = cfg["stages"] or [{"train_files": cfg["train_files"], "steps": cfg["steps"]}]
    total_steps = sum(s["steps"] for s in stages)
    data_meta, plan = [], []  # plan: list of (examples, micro-batch indices) per optimizer step
    for si, st in enumerate(stages):
        exs = []
        for f in st["train_files"]:
            f = {"path": f, "weight": 1} if isinstance(f, str) else f
            e, dropped = load_examples(f["path"], tok, cfg["max_seq_len"])
            exs += e * f.get("weight", 1)
            data_meta.append({"stage": si, "path": f["path"], "sha256": file_sha256(f["path"]),
                              "examples": len(e), "dropped_too_long": dropped, "weight": f.get("weight", 1)})
        micro = batch_schedule(len(exs), cfg["batch_size"], st["steps"] * cfg["grad_accum"], cfg["seed"] + 1000 * si)
        for s in range(st["steps"]):
            plan.append((exs, micro[s * cfg["grad_accum"]:(s + 1) * cfg["grad_accum"]]))
    val = load_examples(cfg["val_file"], tok, cfg["max_seq_len"])[0] if cfg["val_file"] else []

    sched = optim.join_schedules(
        [optim.linear_schedule(1e-7, cfg["lr"], cfg["warmup"]),
         optim.cosine_decay(cfg["lr"], max(1, total_steps - cfg["warmup"]), cfg["lr"] * 0.1)],
        [cfg["warmup"]])
    opt = optim.AdamW(learning_rate=sched, weight_decay=cfg["weight_decay"])

    start = 0
    ck = None if fresh else _latest_ckpt(out)
    if ck is not None:
        model.load_weights(str(ck / "adapters.safetensors"), strict=False)
        opt.init(model.trainable_parameters())
        state = tree_unflatten(list(mx.load(str(ck / "optimizer.safetensors")).items()))
        opt.state = state
        start = json.loads((ck / "state.json").read_text())["step"]
        print(f"[train] resumed from {ck} at step {start}")

    (out / "config.json").write_text(json.dumps({**cfg, "total_steps": total_steps,
                                                 "trainable_params": int(n_train), "data": data_meta}, indent=2))
    adapter_cfg = {"fine_tune_type": "lora", "num_layers": n_layers, "lora_parameters": cfg["lora"],
                   "base_model": cfg["model"]}
    (out / "adapter_config.json").write_text(json.dumps(adapter_cfg, indent=2))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def evaluate_val() -> float:
        if not val:
            return float("nan")
        model.eval()
        tot, n = 0.0, 0
        for i in range(0, min(len(val), cfg["val_batches"] * cfg["batch_size"]), cfg["batch_size"]):
            b, l = collate(val[i:i + cfg["batch_size"]], cfg["max_seq_len"])
            lv, nt = loss_fn(model, b, l)
            tot += lv.item() * nt.item()
            n += nt.item()
        model.train()
        return tot / max(1, n)

    log = open(out / "train_log.jsonl", "a")
    model.train()
    t_start, tok_count = time.perf_counter(), 0
    for step in range(start, total_steps):
        exs, micro = plan[step]
        grads, lsum, nsum, seq_tokens = None, 0.0, 0, 0
        for idx in micro:
            seq_tokens += sum(len(exs[i][0]) for i in idx)
            b, l = collate([exs[i] for i in idx], cfg["max_seq_len"])
            (lv, nt), g = loss_and_grad(model, b, l)
            grads = g if grads is None else tree_map_add(grads, g)
            mx.eval(lv, nt, grads)
            lsum += lv.item() * nt.item()
            nsum += nt.item()
        grads = tree_map_scale(grads, 1.0 / len(micro))
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.state)
        tok_count += seq_tokens
        s = step + 1
        rec = {"step": s, "loss": lsum / max(1, nsum), "lr": float(sched(s)), "supervised_tokens": nsum,
               "seq_tokens": seq_tokens, "tok_per_s": tok_count / (time.perf_counter() - t_start),
               "peak_mem_gb": mx.get_peak_memory() / 1e9}
        if s % cfg["eval_every"] == 0 or s == total_steps:
            rec["val_loss"] = evaluate_val()
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if s % 10 == 0 or "val_loss" in rec:
            print(f"[train] step {s}/{total_steps} loss {rec['loss']:.4f} "
                  f"{'val ' + format(rec['val_loss'], '.4f') + ' ' if 'val_loss' in rec else ''}"
                  f"tok/s {rec['tok_per_s']:.0f} mem {rec['peak_mem_gb']:.2f}GB", flush=True)
        if s % cfg["save_every"] == 0 or s == total_steps or s == cfg.get("stop_after"):
            save_checkpoint(out, model, opt, s)
        mx.clear_cache()
        if s == cfg.get("stop_after"):  # test hook: simulate an interruption after a checkpoint
            return {"interrupted_at": s}

    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(out / "adapters.safetensors"), weights)
    summary = {"out_dir": str(out), "steps": total_steps, "trainable_params": int(n_train),
               "train_time_s": time.perf_counter() - t_start, "resumed_from": start,
               "final_val_loss": evaluate_val(), "peak_mem_gb": mx.get_peak_memory() / 1e9,
               "adapter_sha256": file_sha256(str(out / "adapters.safetensors")), "data": data_meta}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "DONE").write_text("ok\n")
    return summary


def tree_map_add(a, b):
    from mlx.utils import tree_map
    return tree_map(lambda x, y: x + y, a, b)


def tree_map_scale(a, k):
    from mlx.utils import tree_map
    return tree_map(lambda x: x * k, a)


def save_checkpoint(out: Path, model, opt, step: int, keep: int = 2) -> None:
    ck = out / "checkpoints" / f"step_{step:05d}"
    tmp = ck.with_suffix(".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(tmp / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    mx.save_safetensors(str(tmp / "optimizer.safetensors"), dict(tree_flatten(opt.state)))
    (tmp / "state.json").write_text(json.dumps({"step": step}))
    if ck.exists():
        shutil.rmtree(ck)
    tmp.rename(ck)  # atomic publish: a crash never leaves a half-written checkpoint
    for old in sorted(out.glob("checkpoints/step_*"), key=lambda p: int(p.name.split("_")[1]))[:-keep]:
        shutil.rmtree(old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="override key=json_value")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    print(json.dumps(train(cfg, fresh=a.fresh), indent=2))


if __name__ == "__main__":
    main()
