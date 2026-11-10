"""PyTorch / PEFT backend for Study 02: a batched policy and a resumable LoRA trainer.

The trainer mirrors `ccl.train` (MLX) step for step so that Study 01 can be reproduced:
episode packing and the batch schedule come from `ccl.sft`, the loss is the mean
cross-entropy over assistant tokens of each micro-batch, gradients are averaged over
micro-batches and clipped by global norm, the schedule is linear warm-up from 1e-7 then
cosine decay to 10% of the peak, and checkpoints (adapter + optimizer state + step) are
published atomically.

Two things differ between the frameworks and are explicit settings here:

* `lora.alpha`: MLX-LM multiplies the low-rank update by `scale`; PEFT multiplies it by
  `alpha / rank`. Study 01's scale 20 at rank 16 is `alpha = 320`.
* `bias_correction`: MLX's Adam omits bias correction by default, PyTorch's applies it.
  The Study 01 reproduction uses `false`; Study 02's conventional settings use `true`.

Usage: python -m ccl.hf <config.json> [--fresh]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .sft import batch_schedule, file_sha256, load_examples, render_prompt_ids

QWEN_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

DEFAULTS = {
    "model": "models/Qwen2.5-0.5B-Instruct",
    "train_files": [],  # list of {"path": ..., "weight": int} or str
    "val_file": None,
    "stages": None,  # optional sequential curriculum: [{"train_files": [...], "steps": N}, ...]
    "out_dir": "runs/debug",
    "seed": 0,
    "steps": 250,
    "batch_size": 4,
    "grad_accum": 2,
    "lr": 1e-4,
    "warmup": 20,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "bias_correction": True,
    "max_seq_len": 2048,
    # layers: None = every block, otherwise the list of block indices that receive adapters
    "lora": {"rank": 16, "alpha": 32, "dropout": 0.0, "layers": None, "target_modules": QWEN_LINEAR},
    "amp": "auto",  # bf16 autocast on CUDA, full precision on CPU
    "grad_checkpoint": False,  # recompute block activations in the backward pass: same gradients, less memory
    "save_every": 100,
    "eval_every": 100,
    "val_batches": 20,
}


# ------------------------------------------------------------------ model

def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_base(path: str, device: str | None = None, dtype: str = "float32"):
    """Load tokenizer and base model. Students are held in float32 (autocast provides bf16 on CUDA);
    a large reference model can be loaded directly in "bfloat16"."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype))
    model.to(device)
    model.requires_grad_(False)
    return model, tok


def add_lora(model, lora: dict, seed: int):
    """Inject LoRA layers in place (the model keeps its class) and leave only them trainable."""
    from peft import LoraConfig, inject_adapter_in_model
    torch.manual_seed(seed)
    cfg = LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora.get("dropout", 0.0),
                     target_modules=list(lora.get("target_modules") or QWEN_LINEAR),
                     layers_to_transform=lora.get("layers"), bias="none")
    model = inject_adapter_in_model(cfg, model)
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)
    return model


def adapter_state(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().to("cpu", torch.float32).contiguous()
            for n, p in model.named_parameters() if p.requires_grad}


def load_adapter(model, path: str | Path) -> None:
    from safetensors.torch import load_file
    state = load_file(str(path))
    params = dict(model.named_parameters())
    missing = sorted(set(state) - set(params))
    if missing or set(state) != {n for n, p in params.items() if p.requires_grad}:
        raise ValueError(f"adapter does not match the model's LoRA layers ({len(missing)} unknown tensors)")
    with torch.no_grad():
        for n, v in state.items():
            params[n].copy_(v.to(params[n].device, params[n].dtype))


def load_policy_model(model_path: str, adapter_dir: str | Path | None, device: str | None = None,
                      dtype: str = "float32"):
    """Base model, plus the adapter saved by `train` when `adapter_dir` is given."""
    model, tok = load_base(model_path, device, dtype)
    if adapter_dir:
        ad = Path(adapter_dir)
        acfg = json.loads((ad / "adapter_config.json").read_text())
        model = add_lora(model, acfg["lora"], seed=0)
        load_adapter(model, ad / "adapters.safetensors")
        merge_lora(model)
    model.eval()
    return model, tok


def merge_lora(model) -> int:
    """Fold every low-rank update into its base weight (W + scale * B A). The model computes the same
    function with one matrix product per layer instead of three, which makes scoring cheaper."""
    from peft.tuners.lora import LoraLayer
    n = 0
    for m in model.modules():
        if isinstance(m, LoraLayer) and not m.merged:
            m.merge()
            n += 1
    return n


# ------------------------------------------------------------------ optimisation

class Adam(torch.optim.Optimizer):
    """Adam with decoupled weight decay and optional bias correction.

    `bias_correction=False` is MLX's default update, `p -= lr * m / (sqrt(v) + eps)`;
    `True` is the PyTorch AdamW update. The learning rate is passed to `step`."""

    def __init__(self, params, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, bias_correction=True):
        super().__init__(params, dict(betas=betas, eps=eps, weight_decay=weight_decay,
                                      bias_correction=bias_correction))
        self.t = 0

    @torch.no_grad()
    def step(self, lr: float) -> None:
        self.t += 1
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["m"], st["v"] = torch.zeros_like(p), torch.zeros_like(p)
                if g["weight_decay"]:
                    p.mul_(1 - lr * g["weight_decay"])
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                if g["bias_correction"]:
                    denom = (st["v"] / (1 - b2 ** self.t)).sqrt_().add_(g["eps"])
                    p.addcdiv_(st["m"], denom, value=-lr / (1 - b1 ** self.t))
                else:
                    p.addcdiv_(st["m"], st["v"].sqrt().add_(g["eps"]), value=-lr)

    def state_dict(self):
        return {**super().state_dict(), "t": self.t}

    def load_state_dict(self, sd):
        sd = dict(sd)
        self.t = sd.pop("t")
        super().load_state_dict(sd)


def lr_at(k: int, lr: float, warmup: int, total: int) -> float:
    """Learning rate of the k-th update (0-based): linear 1e-7 -> lr over `warmup` updates,
    then cosine decay to 0.1 * lr over the remaining ones (the MLX schedule of Study 01)."""
    if k < warmup:
        return 1e-7 + (lr - 1e-7) * k / warmup
    decay = max(1, total - warmup)
    end = lr * 0.1
    return end + (lr - end) * 0.5 * (1 + math.cos(math.pi * min(k - warmup, decay) / decay))


def collate(examples, max_len: int, device):
    lengths = [min(len(t), max_len) for t, _ in examples]
    width = max(lengths)
    toks = np.zeros((len(examples), width), np.int64)
    mask = np.zeros((len(examples), width), bool)
    for i, (t, m) in enumerate(examples):
        toks[i, : lengths[i]] = t[: lengths[i]]
        mask[i, : lengths[i]] = m[: lengths[i]]
    return torch.from_numpy(toks).to(device), torch.from_numpy(mask).to(device)


def _autocast(model, amp):
    dev = next(model.parameters()).device.type
    on = dev == "cuda" and amp in ("auto", "bf16")
    return torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=on)


def loss_fn(model, toks, mask, amp="auto"):
    """Mean cross-entropy over supervised (assistant) tokens. The output layer is applied only
    at supervised positions, which is the same loss at a fraction of the memory."""
    inputs, targets, m = toks[:, :-1], toks[:, 1:], mask[:, 1:]
    with _autocast(model, amp):
        hidden = model.base_model(input_ids=inputs).last_hidden_state
        logits = model.get_output_embeddings()(hidden[m])
    ntoks = int(m.sum())
    return F.cross_entropy(logits.float(), targets[m], reduction="sum") / ntoks, ntoks


def _latest_ckpt(out: Path) -> Path | None:
    cks = sorted(out.glob("checkpoints/step_*"), key=lambda p: int(p.name.split("_")[1]))
    # state.json is written and mirrored last, so a checkpoint without it was cut off and is ignored
    cks = [c for c in cks if c.suffix != ".tmp" and (c / "state.json").exists()]
    return cks[-1] if cks else None


def save_checkpoint(out: Path, model, opt, step: int, keep: int = 2) -> None:
    from safetensors.torch import save_file
    ck = out / "checkpoints" / f"step_{step:05d}"
    tmp = ck.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_file(adapter_state(model), str(tmp / "adapters.safetensors"))
    torch.save(opt.state_dict(), tmp / "optimizer.pt")
    (tmp / "state.json").write_text(json.dumps({"step": step}))
    if ck.exists():
        shutil.rmtree(ck)
    tmp.rename(ck)  # atomic publish: a crash never leaves a half-written checkpoint
    done = sorted((p for p in out.glob("checkpoints/step_*") if p.suffix != ".tmp"),
                  key=lambda p: int(p.name.split("_")[1]))
    for old in done[:-keep]:
        shutil.rmtree(old)


def train(cfg: dict, fresh: bool = False, model_tok=None, on_checkpoint=None) -> dict:
    """Train one adapter. `model_tok` lets a caller pass an already loaded (base model, tokenizer);
    `on_checkpoint` is called after every checkpoint (used to mirror the run to durable storage)."""
    from safetensors.torch import save_file
    cfg = {**DEFAULTS, **cfg}
    cfg["lora"] = {**DEFAULTS["lora"], **cfg["lora"]}
    out = Path(cfg["out_dir"])
    if fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        print(f"[train] {out} already complete; skipping", flush=True)
        return json.loads((out / "summary.json").read_text())

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model, tok = model_tok or load_base(cfg["model"])
    device = next(model.parameters()).device
    model = add_lora(model, cfg["lora"], cfg["seed"])
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    if cfg["grad_checkpoint"]:
        model.gradient_checkpointing_enable()

    # ---- data (stages allow sequential curricula; a single stage otherwise)
    stages = cfg["stages"] or [{"train_files": cfg["train_files"], "steps": cfg["steps"]}]
    total_steps = sum(s["steps"] for s in stages)
    data_meta, plan = [], []  # plan: (examples, micro-batch indices) per optimizer step
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

    opt = Adam(params, weight_decay=cfg["weight_decay"], bias_correction=cfg["bias_correction"])
    start = 0
    ck = None if fresh else _latest_ckpt(out)
    if ck is not None:
        load_adapter(model, ck / "adapters.safetensors")
        opt.load_state_dict(torch.load(ck / "optimizer.pt", map_location=device))
        start = json.loads((ck / "state.json").read_text())["step"]
        print(f"[train] resumed from {ck} at step {start}", flush=True)

    (out / "config.json").write_text(json.dumps({**cfg, "total_steps": total_steps,
                                                 "trainable_params": int(n_train), "data": data_meta}, indent=2))
    (out / "adapter_config.json").write_text(json.dumps(
        {"fine_tune_type": "lora", "format": "ccl.hf", "lora": cfg["lora"], "base_model": cfg["model"]}, indent=2))

    def evaluate_val() -> float:
        if not val:
            return float("nan")
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, min(len(val), cfg["val_batches"] * cfg["batch_size"]), cfg["batch_size"]):
                b, l = collate(val[i:i + cfg["batch_size"]], cfg["max_seq_len"], device)
                lv, nt = loss_fn(model, b, l, cfg["amp"])
                tot += lv.item() * nt
                n += nt
        model.train()
        return tot / max(1, n)

    def peak_mem() -> float:
        return torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    log = open(out / "train_log.jsonl", "a")
    model.train()
    t_start, tok_count = time.perf_counter(), 0
    for step in range(start, total_steps):
        exs, micro = plan[step]
        lsum, nsum, seq_tokens = 0.0, 0, 0
        for idx in micro:
            seq_tokens += sum(len(exs[i][0]) for i in idx)
            b, l = collate([exs[i] for i in idx], cfg["max_seq_len"], device)
            lv, nt = loss_fn(model, b, l, cfg["amp"])
            (lv / len(micro)).backward()  # gradients are averaged over micro-batches
            lsum += lv.item() * nt
            nsum += nt
        if cfg.get("max_grad_norm"):
            gnorm = torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]).item()  # pre-clip norm
        else:
            gnorm = math.sqrt(sum(float((p.grad.float() ** 2).sum()) for p in params if p.grad is not None))
        opt.step(lr_at(step, cfg["lr"], cfg["warmup"], total_steps))
        opt.zero_grad(set_to_none=True)
        tok_count += seq_tokens
        s = step + 1
        rec = {"step": s, "loss": lsum / max(1, nsum), "lr": lr_at(s, cfg["lr"], cfg["warmup"], total_steps),
               "grad_norm": gnorm, "supervised_tokens": nsum, "seq_tokens": seq_tokens,
               "tok_per_s": tok_count / (time.perf_counter() - t_start), "peak_mem_gb": peak_mem()}
        if s % cfg["eval_every"] == 0 or s == total_steps:
            rec["val_loss"] = evaluate_val()
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if s % 25 == 0 or "val_loss" in rec:
            print(f"[train] step {s}/{total_steps} loss {rec['loss']:.4f} "
                  f"{'val ' + format(rec['val_loss'], '.4f') + ' ' if 'val_loss' in rec else ''}"
                  f"gnorm {gnorm:.2f} tok/s {rec['tok_per_s']:.0f}", flush=True)
        if s % cfg["save_every"] == 0 or s == total_steps or s == cfg.get("stop_after"):
            save_checkpoint(out, model, opt, s)
            if on_checkpoint:
                on_checkpoint()
        if s == cfg.get("stop_after"):  # test hook: simulate an interruption after a checkpoint
            return {"interrupted_at": s}

    save_file(adapter_state(model), str(out / "adapters.safetensors"))
    summary = {"out_dir": str(out), "steps": total_steps, "trainable_params": int(n_train),
               "train_time_s": time.perf_counter() - t_start, "resumed_from": start,
               "final_val_loss": evaluate_val(), "peak_mem_gb": peak_mem(),
               "adapter_sha256": file_sha256(str(out / "adapters.safetensors")), "data": data_meta}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "DONE").write_text("ok\n")
    return summary


# ------------------------------------------------------------------ inference

class HFPolicy:
    """Batched generation behind the `Policy` protocol of `ccl.runtime`.

    Greedy by default. The model's own generation defaults (Qwen ships sampling and a
    repetition penalty) are overridden so that decoding matches Study 01."""

    def __init__(self, model, tokenizer, name: str, max_tokens: int = 256, temperature: float = 0.0,
                 seed: int = 0, amp: str = "auto"):
        self.model, self.tokenizer, self.name = model, tokenizer, name
        self.max_tokens, self.temperature, self.amp = max_tokens, temperature, amp
        self.device = next(model.parameters()).device
        tok = tokenizer
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ends = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                tok.convert_tokens_to_ids("<|endoftext|>")}
        self.eos_ids = sorted(e for e in ends if isinstance(e, int) and e >= 0 and e != tok.unk_token_id)
        self.stats = {"prompt_tokens": 0, "gen_tokens": 0, "gen_time_s": 0.0, "calls": 0}
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        torch.manual_seed(seed)

    def generate(self, batch: list[list[dict]], episodes=None) -> list[tuple[str, int]]:
        """Generate for a batch; if the GPU runs out of memory the batch is halved and retried."""
        try:
            return self._generate(batch)
        except torch.OutOfMemoryError:
            if len(batch) == 1:
                raise
            torch.cuda.empty_cache()
            self.stats["oom_splits"] = self.stats.get("oom_splits", 0) + 1
            half = len(batch) // 2
            return self.generate(batch[:half]) + self.generate(batch[half:])

    @torch.inference_mode()
    def _generate(self, batch: list[list[dict]]) -> list[tuple[str, int]]:
        prompts = [render_prompt_ids(self.tokenizer, m) for m in batch]
        width = max(map(len, prompts))
        ids = torch.full((len(prompts), width), self.pad_id, dtype=torch.long)
        att = torch.zeros((len(prompts), width), dtype=torch.long)
        for i, p in enumerate(prompts):  # left padding, so that generation continues every prompt
            ids[i, width - len(p):] = torch.tensor(p)
            att[i, width - len(p):] = 1
        ids, att = ids.to(self.device), att.to(self.device)
        t0 = time.perf_counter()
        with _autocast(self.model, self.amp):
            new = self._sample(ids, att) if self.temperature > 0 else self._greedy(ids, att)
        dt = time.perf_counter() - t0
        res = []
        for row in new.tolist():
            n = next((j for j, t in enumerate(row) if t in self.eos_ids), len(row))
            res.append((self.tokenizer.decode(row[:n], skip_special_tokens=True).strip(), n))
        self.stats["prompt_tokens"] += sum(map(len, prompts))
        self.stats["gen_tokens"] += sum(n for _, n in res)
        self.stats["gen_time_s"] += dt
        self.stats["calls"] += 1
        return res

    def _greedy(self, ids, att):
        """Greedy decoding with a key-value cache: the same computation as `generate(do_sample=False)`
        without its per-token bookkeeping. The output layer is applied to the last position only."""
        body, head = self.model.base_model, self.model.get_output_embeddings()
        eos = torch.tensor(self.eos_ids, device=ids.device)
        pos = (att.cumsum(-1) - 1).masked_fill(att == 0, 1)  # positions count real tokens, as in generate
        nxt_pos = att.sum(-1, keepdim=True)
        out = body(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=True)
        done = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
        new = []
        for step in range(self.max_tokens):
            tok = head(out.last_hidden_state[:, -1]).argmax(-1)
            tok = torch.where(done, torch.full_like(tok, self.pad_id), tok)
            new.append(tok)
            done |= torch.isin(tok, eos)
            if bool(done.all()) or step == self.max_tokens - 1:
                break
            att = torch.cat([att, att.new_ones(att.shape[0], 1)], dim=1)
            out = body(input_ids=tok[:, None], attention_mask=att, position_ids=nxt_pos + step,
                       past_key_values=out.past_key_values, use_cache=True)
        return torch.stack(new, dim=1)

    def _sample(self, ids, att):
        from transformers import GenerationConfig
        gc = GenerationConfig(max_new_tokens=self.max_tokens, do_sample=True, num_beams=1, repetition_penalty=1.0,
                              temperature=self.temperature, top_p=0.95, top_k=0, pad_token_id=self.pad_id,
                              eos_token_id=self.eos_ids)
        return self.model.generate(input_ids=ids, attention_mask=att, generation_config=gc)[:, ids.shape[1]:]

    def reference_generate(self, batch: list[list[dict]]) -> list[tuple[str, int]]:
        """Greedy decoding through the library's own `generate` (used by the tests as the reference)."""
        from transformers import GenerationConfig
        prompts = [render_prompt_ids(self.tokenizer, m) for m in batch]
        width = max(map(len, prompts))
        ids = torch.full((len(prompts), width), self.pad_id, dtype=torch.long)
        att = torch.zeros((len(prompts), width), dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, width - len(p):] = torch.tensor(p)
            att[i, width - len(p):] = 1
        gc = GenerationConfig(max_new_tokens=self.max_tokens, do_sample=False, num_beams=1, repetition_penalty=1.0,
                              pad_token_id=self.pad_id, eos_token_id=self.eos_ids)
        with torch.inference_mode(), _autocast(self.model, self.amp):
            out = self.model.generate(input_ids=ids.to(self.device), attention_mask=att.to(self.device),
                                      generation_config=gc)
        res = []
        for row in out[:, width:].tolist():
            n = next((j for j, t in enumerate(row) if t in self.eos_ids), len(row))
            res.append((self.tokenizer.decode(row[:n], skip_special_tokens=True).strip(), n))
        return res


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
