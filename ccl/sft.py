"""Framework-free supervised fine-tuning helpers.

Prompt tokenisation, episode packing, the deterministic batch schedule and file hashing are
shared by the MLX trainer of Study 01 (`ccl.train`, `ccl.models`) and the PyTorch trainer of
Study 02 (`ccl.hf`). They were moved here unchanged so that they can be imported on a machine
without MLX; `ccl.train` and `ccl.models` re-export them under their old names.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

CHAT_KW = {"enable_thinking": False}  # all models are run in non-thinking mode


def render_prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    """Single source of truth for prompt tokenisation (training and inference)."""
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **CHAT_KW)
    # Some tokenizer versions return an encoding object instead of a plain list.
    return ids if isinstance(ids, list) else list(ids["input_ids"])


def completion_ids(tokenizer, text: str) -> list[int]:
    """Tokens of an assistant completion including the end-of-turn marker."""
    end = "<|im_end|>"
    return tokenizer.encode(text + end, add_special_tokens=False)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def episode_sequence(tokenizer, ex: dict) -> tuple[list[int], list[int]] | None:
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


def pack_rows(rows: list[dict], tokenizer, max_len: int) -> tuple[list[tuple[list[int], list[int]]], int]:
    """SFT rows (one per assistant turn) -> [(tokens, loss_mask)] per episode, and the number dropped
    for exceeding max_len. Only the last turn of each episode is needed because its `messages`
    contain the full canonical history."""
    last = {}
    for r in rows:
        key = (r["meta"]["task_id"], r["meta"]["source"], r["meta"].get("sample", 0))
        if key not in last or r["meta"]["turn"] > last[key]["meta"]["turn"]:
            last[key] = r
    out, dropped = [], 0
    for r in last.values():
        packed = episode_sequence(tokenizer, r)
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


def load_examples(path: str, tokenizer, max_len: int) -> tuple[list[tuple[list[int], list[int]]], int]:
    """Read an SFT file and pack it; see `pack_rows`."""
    return pack_rows([json.loads(l) for l in open(path)], tokenizer, max_len)


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
