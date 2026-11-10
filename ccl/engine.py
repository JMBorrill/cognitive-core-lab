"""Rollout engine for Study 02, Experiment 2 (plan §4, R1-R4).

`ccl.hf.HFPolicy` re-reads a whole conversation at every turn and keeps student weights in float32 under
autocast. That was the cost of Experiment 1. This engine changes how the same computation is carried out:

  R1  inference weights are held in bfloat16 after the adapter is merged (`load_policy_model`);
  R2  an episode's key-value cache is kept between turns, so a turn reads only what is new;
  R3  a group of episodes is sized by tokens, not by a fixed count;
  R4  a practice attempt may be stopped at its first parse error, unknown tool or invalid argument,
      because the acceptance rule can no longer admit it;
  R6  while one token per row is decoded, attention is computed per key-value head (`grouped_attention`).
      The student has 12 query heads on 2 key-value heads, and with an attention mask present the library
      copies the whole cache out to 12 heads in every layer at every step, which was three quarters of the
      first Stage 1 calibration's time. (R5, a vLLM fallback, was not needed.)

How R2 stays exact. Each turn the canonical prompt of every episode is rendered as before
(`ccl.sft.render_prompt_ids`) and compared with the tokens that episode already has in the cache. The
common prefix is kept; any cached token beyond it is switched off in the attention mask; the rest of the
canonical prompt is fed as new tokens at their canonical positions. The model therefore attends to exactly
the token sequence that full re-encoding would give it, whatever was generated (text is stripped and
re-tokenised between turns, and that need not round-trip). Rows of a group differ in length, so new tokens
are left-padded and the padding is masked.

Sampling is seeded per task, attempt and turn: the uniform numbers that pick each token come from a
generator keyed on those three and the engine's seed. An attempt does not depend on which other episodes
share its group or on how a job was resumed, up to floating-point rounding of the logits.
"""

from __future__ import annotations

import gc
import hashlib
import time

import numpy as np
import torch

from .runtime import Episode, Limits, render_system
from .sft import render_prompt_ids

DISQUALIFYING = ("hallucinated", "invalid_argument")


def load_policy_model(model_path: str, adapter_dir=None, device: str | None = None, dtype: str = "bfloat16"):
    """R1: the model `ccl.hf.load_policy_model` returns (adapter merged in float32), then cast once."""
    from . import hf
    model, tok = hf.load_policy_model(model_path, adapter_dir, device, "float32" if adapter_dir else dtype)
    model.to(getattr(torch, dtype))
    return model.eval(), tok


def grouped_attention(stock):
    """An attention function for single-token decoding with grouped-query attention and a mask.

    The G query heads that share a key-value head are folded into the query dimension, so keys and values are
    read once as they lie in the cache and never expanded to all heads. The same sums as the stock function,
    in another order; every other case (no mask, several query positions, no grouping) goes to `stock`."""
    def attend(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        g = getattr(module, "num_key_value_groups", 1)
        if g == 1 or query.shape[2] != 1 or attention_mask is None or attention_mask.dtype != torch.bool \
                or module.training or query.shape[1] != g * key.shape[1]:
            return stock(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        b, h, _, d = query.shape
        q = query.reshape(b, h // g, g, d)  # head j * g + i is query i of key-value head j (as in repeat_kv)
        scores = torch.matmul(q, key.transpose(-1, -2)) * (scaling if scaling is not None else d ** -0.5)
        scores = scores.float().masked_fill(~attention_mask, float("-inf"))  # the mask is (rows, 1, 1, keys)
        out = torch.matmul(torch.softmax(scores, dim=-1).to(value.dtype), value)
        return out.reshape(b, h, 1, d).transpose(1, 2).contiguous(), None
    return attend


class _Grouped:
    """While active, the library's `sdpa` attention is `grouped_attention`; restored on exit."""

    def __enter__(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as fns
        self.fns, self.stock = fns, fns["sdpa"]
        fns["sdpa"] = grouped_attention(self.stock)  # a local override of this mapping object only

    def __exit__(self, *exc):
        del self.fns["sdpa"]
        assert self.fns["sdpa"] is self.stock


def disqualified(ep: Episode) -> bool:
    """True once an attempt holds a parse error, an unknown tool or an invalid argument (the three
    runtime observations of the acceptance rule, `ccl.s2.rejection`)."""
    t = ep.trace
    return any(x["kind"] == "invalid" for x in t.turns) or any(c.outcome in DISQUALIFYING for c in t.calls)


def uniforms(seed, task_id: str, attempt: int, turn: int, n: int) -> np.ndarray:
    key = hashlib.sha256(f"{seed}|{task_id}|{attempt}|{turn}".encode()).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "big")).random(n)


class Engine:
    """Runs episodes to completion in groups that share one batched key-value cache.

    `run` is the driver (it replaces `ccl.runtime.run_batch` for a model policy); `ccl.s2.evaluate` uses it
    when a policy has one. Greedy when `temperature` is 0."""

    def __init__(self, model, tokenizer, name: str, max_tokens: int = 256, temperature: float = 0.0,
                 top_p: float = 0.95, seed=0, token_budget: int = 160_000, max_rows: int = 128,
                 turn_reserve: int = 600, grouped: bool = True):
        self.model, self.tokenizer, self.name, self.grouped = model, tokenizer, name, grouped
        self.max_tokens, self.temperature, self.top_p, self.seed = max_tokens, temperature, top_p, seed
        self.token_budget, self.max_rows, self.turn_reserve = token_budget, max_rows, turn_reserve
        self.device = next(model.parameters()).device
        tok = tokenizer
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ends = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                tok.convert_tokens_to_ids("<|endoftext|>")}
        self.eos_ids = sorted(e for e in ends if isinstance(e, int) and e >= 0 and e != tok.unk_token_id)
        self.stats = {"episodes": 0, "groups": 0, "turn_batches": 0, "decode_steps": 0, "fed_tokens": 0,
                      "refed_tokens": 0, "gen_tokens": 0, "gen_time_s": 0.0, "oom_splits": 0, "stopped": 0}

    # -------------------------------------------------------------- driver

    def run(self, tasks, on_done=None, fewshot: bool = False, attempts=(0,), stop_on_error: bool = False,
            limits: Limits = Limits()) -> list[Episode]:
        """Every task once per attempt index; returns the episodes in that order (the attempts of a task
        adjacent), each carrying `.attempt`. Groups are formed from the longest prompt down, so that the rows
        of a group are of similar length. `on_done` is called per episode when its group has finished."""
        jobs = [(t, a) for t in tasks for a in attempts]
        size = {}
        for t in tasks:
            if id(t) not in size:
                msgs = [{"role": "system", "content": render_system(t, fewshot)}, {"role": "user", "content": t.request}]
                size[id(t)] = len(render_prompt_ids(self.tokenizer, msgs)) + self.turn_reserve
        order = sorted(range(len(jobs)), key=lambda j: -size[id(jobs[j][0])])  # stable: ties keep their order
        out: list = [None] * len(jobs)
        i = 0
        while i < len(order):
            widest = size[id(jobs[order[i]][0])]  # the first row of a group is its longest
            n = max(1, min(self.max_rows, self.token_budget // widest, len(order) - i))
            group = order[i:i + n]
            for j, ep in zip(group, self._run_safely([jobs[j] for j in group], fewshot, stop_on_error, limits)):
                out[j] = ep
                if on_done:
                    on_done(ep)
            i += n
        return out

    def _run_safely(self, jobs, fewshot, stop_on_error, limits) -> list[Episode]:
        """One group; if the GPU runs out of memory it is halved and both halves start again from their first
        turn. The retry happens outside the handler, where the failed call's memory can be released."""
        try:
            eps = []
            for task, attempt in jobs:
                e = Episode(task, self.name, limits, fewshot=fewshot)
                e.attempt = attempt
                eps.append(e)
            if not self.grouped:
                return self._run_group(eps, stop_on_error)
            with _Grouped():
                return self._run_group(eps, stop_on_error)
        except torch.OutOfMemoryError:
            if len(jobs) == 1:
                raise
        gc.collect()
        torch.cuda.empty_cache()
        self.stats["oom_splits"] += 1
        half = len(jobs) // 2
        return self._run_safely(jobs[:half], fewshot, stop_on_error, limits) + \
            self._run_safely(jobs[half:], fewshot, stop_on_error, limits)

    # -------------------------------------------------------------- one group

    @torch.inference_mode()
    def _run_group(self, eps: list, stop_on_error: bool = False) -> list:
        """Advance fresh episodes to their end. An episode is anything with `messages`, `step`, `done`,
        `turn`, `attempt` and `task.task_id` (`ccl.runtime.Episode`; the tests also use scripted ones)."""
        dev, tok = self.device, self.tokenizer
        body, head = self.model.base_model, self.model.get_output_embeddings()
        eos = torch.tensor(self.eos_ids, device=dev)
        hist = [[] for _ in eps]  # per episode: the tokens it has switched on in the cache, in order
        cols = [[] for _ in eps]  # and the cache column of each
        live = list(range(len(eps)))  # cache row r belongs to episode live[r]
        att = torch.zeros((len(eps), 0), dtype=torch.bool, device=dev)
        cache = None
        while live:
            t0 = time.perf_counter()
            # ---- what each row must read this turn: the canonical prompt beyond what it has cached
            new = []
            for r, i in enumerate(live):
                p, h = render_prompt_ids(tok, eps[i].messages), hist[i]
                k, stop = 0, min(len(h), len(p) - 1)
                while k < stop and h[k] == p[k]:
                    k += 1
                if k < len(h):  # the cache holds tokens the canonical prompt does not: switch them off
                    att[r, torch.tensor(cols[i][k:], device=dev)] = False
                    self.stats["refed_tokens"] += len(h) - k
                    del h[k:], cols[i][k:]
                new.append(p[k:])
            width, base = max(map(len, new)), att.shape[1]
            ids = torch.full((len(live), width), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(live), width), dtype=torch.bool)
            pos = torch.zeros((len(live), width), dtype=torch.long)
            for r, (i, x) in enumerate(zip(live, new)):
                n, start = len(x), len(hist[i])
                ids[r, width - n:] = torch.tensor(x)
                mask[r, width - n:] = True
                pos[r, width - n:] = torch.arange(start, start + n)
                cols[i] += range(base + width - n, base + width)
                hist[i] += x
            att = torch.cat([att, mask.to(dev)], dim=1)
            out = body(input_ids=ids.to(dev), attention_mask=att, position_ids=pos.to(dev),
                       past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            self.stats["fed_tokens"] += sum(map(len, new))

            # ---- decode: one token per row and step; a finished row is fed masked padding
            nxt = torch.tensor([len(hist[i]) for i in live], device=dev)
            u = None
            if self.temperature > 0:
                u = torch.tensor(np.stack([uniforms(self.seed, eps[i].task.task_id, eps[i].attempt, eps[i].turn,
                                                    self.max_tokens) for i in live]), device=dev)
            done = torch.zeros(len(live), dtype=torch.bool, device=dev)
            toks, fed = [], []
            for step in range(self.max_tokens):
                t = self._pick(head(out.last_hidden_state[:, -1]), None if u is None else u[:, step])
                t = torch.where(done, torch.full_like(t, self.pad_id), t)
                toks.append(t)
                done = done | torch.isin(t, eos)
                if bool(done.all()) or step == self.max_tokens - 1:
                    break
                fed.append(~done)
                att = torch.cat([att, fed[-1][:, None]], dim=1)
                out = body(input_ids=t[:, None], attention_mask=att, position_ids=(nxt + step)[:, None],
                           past_key_values=cache, use_cache=True)
                cache = out.past_key_values
            toks = torch.stack(toks, dim=1).tolist()
            fed = torch.stack(fed, dim=1).tolist() if fed else [[] for _ in live]
            dt = (time.perf_counter() - t0) / len(live)
            first = att.shape[1] - len(fed[0])
            keep = []
            for r, i in enumerate(live):
                row = toks[r]
                n = next((j for j, x in enumerate(row) if x in self.eos_ids), len(row))
                for s, on in enumerate(fed[r]):
                    if on:
                        hist[i].append(row[s])
                        cols[i].append(first + s)
                e = eps[i]
                e.step(tok.decode(row[:n], skip_special_tokens=True).strip(), latency_s=dt, gen_tokens=n)
                self.stats["gen_tokens"] += n
                if stop_on_error and not e.done and disqualified(e):
                    e._finish("stopped")
                    self.stats["stopped"] += 1
                if not e.done:
                    keep.append(r)
            self.stats["turn_batches"] += 1
            self.stats["decode_steps"] += len(toks[0])
            self.stats["gen_time_s"] += time.perf_counter() - t0
            if len(keep) < len(live):  # drop finished rows from the cache
                live = [live[r] for r in keep]
                if live:
                    idx = torch.tensor(keep, device=dev)
                    cache.batch_select_indices(idx)
                    att = att[idx]
        self.stats["episodes"] += len(eps)
        self.stats["groups"] += 1
        return eps

    def _pick(self, logits, u):
        """Next token per row: the arg-max, or a draw from the top-p nucleus at `temperature` by inverting
        the cumulative distribution at the row's uniform number."""
        if u is None:
            return logits.argmax(-1)
        p, order = torch.softmax(logits.float() / self.temperature, dim=-1).sort(dim=-1, descending=True)
        p = p * ((p.cumsum(-1) - p) < self.top_p)  # the smallest set of tokens whose mass reaches top_p
        c = (p / p.sum(-1, keepdim=True)).cumsum(-1)
        k = torch.minimum((c < u[:, None].to(c.dtype)).sum(-1), (p > 0).sum(-1) - 1)  # never outside the nucleus
        return order.gather(-1, k[:, None]).squeeze(-1)
