# Model, teacher and framework selection

Checked 2026-11-08 against the Hugging Face Hub API (licence tags, gating, revisions) and by
running each candidate locally. Hardware: Apple M4 (4P+6E CPU, 10-core GPU), 16 GB unified memory, macOS 27.

## Student (foundation checkpoint) — **Qwen/Qwen2.5-0.5B-Instruct**

| candidate | params | licence | gated | MLX-LM support | measured on this machine |
|---|---|---|---|---|---|
| **Qwen/Qwen2.5-0.5B-Instruct** @ `7ae557604adf` | 0.49B | Apache-2.0 | no | `qwen2` (mature) | LoRA r=16, batch 4: 10 steps in 48 s, peak 3.8 GB |
| Qwen/Qwen3.5-0.8B @ `2fc06364715b` | 0.87B (incl. vision tower) | Apache-2.0 | no | `qwen3_5` (hybrid Gated-DeltaNet + attention) | LoRA r=16, **batch 1**: 10 steps in 123 s, peak 7.4 GB; batch 4 was killed by the OS (memory) |
| Qwen/Qwen2.5-1.5B-Instruct @ `989aa7980e4c` | 1.5B | Apache-2.0 | no | `qwen2` | not run (documented fallback) |
| Qwen/Qwen3-0.6B @ `c1899de289a0` | 0.6B | Apache-2.0 | no | `qwen3` | not run |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 1.7B | Apache-2.0 | no | `llama` | not run |
| meta-llama/Llama-3.2-1B-Instruct | 1.2B | Llama 3.2 Community | **manual gate** | `llama` | excluded (gating + licence terms) |
| google/gemma-3-1b-it | 1.0B | Gemma terms | **manual gate** | `gemma3_text` | excluded (gating + licence terms) |

**Why Qwen2.5-0.5B-Instruct.** It has a permissive licence, no gating, a mature MLX implementation, and
roughly **10× the per-example LoRA training throughput** of Qwen3.5-0.8B on this machine. The Qwen3.5
hybrid linear-attention layers have no fused backward kernel in MLX-LM 0.32.2, and its 248k-token
vocabulary inflates logit memory. The experiment needs ~15 training runs plus repeated evaluation on
16 GB, so throughput is the binding constraint. Zero-shot, both candidates were near 0% on a 32-task
probe (Qwen3.5-0.8B 0/32, Qwen2.5-0.5B 1/32), so neither had a head start worth the cost.

**Documented fallback:** Qwen/Qwen2.5-1.5B-Instruct (same tokenizer, chat template and action format; ~3×
the compute). Qwen3.5-0.8B remains the preferred modern checkpoint once MLX-LM gains a fused
Gated-DeltaNet backward pass or the experiment moves to a GPU.

## Teacher — **Qwen/Qwen3.5-4B** (MLX 4-bit conversion `mlx-community/Qwen3.5-4B-MLX-4bit` @ `32f3e8ecf654`)

- No hosted-API credentials were available in this environment (no `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`),
  so the teacher is an open-weight model run locally. Apache-2.0 (upstream `Qwen/Qwen3.5-4B` @ `851bf6e806ef`);
  using its outputs to train another model is permitted by the licence.
- Run in non-thinking mode (`enable_thinking=False`) with the **same system prompt and catalogue rendering**
  as the student.
- Verified to be stronger before use: 75% task success on a 48-task training-pool probe vs ~0–3% for the student.
- Cost: $0 API; local wall-clock only (see `data/v1/teacher/manifest.json`).
- The 4-bit download was not pinned at download time; the revision above is the Hub HEAD recorded on
  2026-11-08, when the files were fetched.

## Frameworks

| component | choice | version |
|---|---|---|
| tensor runtime | MLX | 0.32.4 |
| LM loading, LoRA layers, batched generation | MLX-LM | 0.32.2 |
| training loop | custom (`ccl/train.py`) on MLX-LM LoRA layers | — |
| everything else | Python 3.11 stdlib + NumPy 2.4.6 | — |

Full pinned environment: `requirements.lock`. A custom loop was used instead of `mlx_lm.lora`
because the stock trainer cannot resume optimizer state mid-run, and the experiment needs
deterministic, seed-controlled batch order and multi-stage (sequential) curricula.

## Published tool-use benchmarks considered

BFCL (Berkeley Function-Calling Leaderboard), τ-bench, ToolBench/StableToolBench, NexusRaven and
API-Bank all inform the metric design: tool selection, argument accuracy, multi-turn state,
refusal and policy adherence. They were **not used for training or evaluation**. Their data is either
too large for this budget (ToolBench), depends on live or LLM-simulated users (τ-bench), or would
mix licence regimes. Instead, every task here is generated and validated by deterministic code, so
contamination is controlled by construction (see `data/v1/leakage_audit.json`). Results on this suite
are **not comparable** to published leaderboard numbers.
