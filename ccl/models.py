"""Local MLX model backend (student, LoRA variants and the local teacher)."""

from __future__ import annotations

import time

import mlx.core as mx

from .sft import CHAT_KW, completion_ids, render_prompt_ids  # noqa: F401  (moved; re-exported)


class MLXPolicy:
    def __init__(self, model_path: str, adapter_path: str | None = None, name: str | None = None,
                 max_tokens: int = 256, temperature: float = 0.0, seed: int = 0, prefill_batch_size: int = 16):
        from mlx_lm import load
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        self.name = name or (adapter_path or model_path)
        self.max_tokens, self.temperature, self.seed = max_tokens, temperature, seed
        self.prefill_batch_size = prefill_batch_size
        self.stats = {"prompt_tokens": 0, "gen_tokens": 0, "gen_time_s": 0.0, "calls": 0}
        mx.random.seed(seed)

    def _sampler(self):
        if self.temperature <= 0:
            return None
        from mlx_lm.sample_utils import make_sampler
        return make_sampler(temp=self.temperature, top_p=0.95)

    def generate(self, batch: list[list[dict]], episodes=None) -> list[tuple[str, int]]:
        from mlx_lm.generate import batch_generate
        prompts = [render_prompt_ids(self.tokenizer, m) for m in batch]
        t0 = time.perf_counter()
        resp = batch_generate(self.model, self.tokenizer, prompts, max_tokens=self.max_tokens,
                              return_token_ids=True, sampler=self._sampler(),
                              prefill_batch_size=self.prefill_batch_size)
        dt = time.perf_counter() - t0
        self.stats["prompt_tokens"] += sum(map(len, prompts))
        self.stats["gen_tokens"] += sum(map(len, resp.token_ids))
        self.stats["gen_time_s"] += dt
        self.stats["calls"] += 1
        return [(t.strip(), len(ids)) for t, ids in zip(resp.texts, resp.token_ids)]
