# Study 02 implementation plan

*Draft for review, 2026-11-10. Companion to [PROPOSAL.md](PROPOSAL.md). Prices, package versions and
model availability were gathered on 2026-11-10 from vendor documentation repositories on GitHub, PyPI
and the SkyPilot price catalogue. Provider websites and the Hugging Face Hub were unreachable from
this session, so every such fact is re-checked at the work package that first depends on it.*

## 1. Principles

- Study 01 stays reproducible. Its code paths, data and results are not edited, apart from one
  behaviour-preserving move of framework-free helpers (WP1). `tests/test_study01_frozen.py` rebuilds
  the v1 datasets and compares hashes on every test run.
- New code goes beside the old: five modules and four scripts.
- Everything that does not need a GPU is tested on CPU.
- Every GPU job is one resumable command that ends by itself.

## 2. What is reused and what is new

| file | status | content |
|---|---|---|
| `ccl/tools.py`, `tasks.py`, `runtime.py`, `oracle.py`, `evaluate.py`, `data.py` | reused | simulators, task model, episode loop, oracle, validator, SFT rows |
| `ccl/stats.py` | extended | bootstrap that also resamples families |
| `ccl/train.py`, `ccl/models.py` | kept for Study 01 | MLX trainer and policy |
| `ccl/sft.py` | new (moved) | tokenisation and episode packing, batch schedule, file hashing; today these sit in the MLX modules and cannot be imported on Linux |
| `ccl/hf.py` | new | PyTorch policy (batched generation) and PEFT LoRA trainer with the Study 01 semantics |
| `ccl/world2.py` | new | schema, derived tool specifications, generic simulators |
| `ccl/compose.py` | new | family sampler: schema path → `Task` with request, catalogue, world and gold graph |
| `ccl/loop.py` | new (Exp. 2) | practice cycle, acceptance rule, curriculum sampler |
| `scripts/build_v2.py` | new | practice space, `dev`, sealed test splits, census, leakage audit → `data/v2` |
| `scripts/s2_run.py` | new | the one entry point a GPU job runs: train, evaluate, practise or generate teacher data from a JSON config |
| `scripts/s2_launch.py` | new | submits jobs, enforces the budget, records the ledger |
| `scripts/s2_analyze.py` | new | tables and intervals for Study 02 → `results/study02` |

The existing `Policy` protocol (`generate(batch) -> [(text, n_tokens)]`) is the seam: the PyTorch
student, the open-weight teacher and the oracle all sit behind it, as the MLX models do today.

Two small constraints from the reused code: v2 identifiers follow the `XX-1234` pattern that the
dependency-safety check recognises, and v2 tools are registered separately from the v1 tool table so
that v1 task generation, which samples distractors from that table, is unaffected.

## 3. Training and inference stack

| component | choice | version seen 2026-11-10 |
|---|---|---|
| framework | PyTorch | 2.15.0 |
| models, tokenisers | `transformers` | 5.22.2 |
| adapters | PEFT (LoRA) | 0.21.2 |
| student inference | `transformers` batched `generate` | — |
| teacher inference and large practice runs | vLLM, added in WP5 only if rollouts dominate cost | 0.33.0 |

Pinned in `requirements-cuda.lock`, created in WP1.

**LoRA settings.** Study 01's configuration in PEFT terms is rank 16, `lora_alpha` 320, the seven
linear modules of the last 16 blocks: 5,865,472 trainable parameters, which the port must reproduce
exactly. The Study 02 default is rank 16, `lora_alpha` 32, all blocks, learning rate 1e-4, two passes
over the data, unless Experiment 0 rejects it
([plan](../../experiments/study02_e0_bridge/PLAN.md)).

**Reproducibility of evaluation.** Batched GPU decoding is not bit-reproducible. Experiment 0 scores
one adapter twice and reports the share of episodes that flip; evaluation uses fixed batch
composition, and hardware and library versions are recorded per run.

## 4. Models

| role | model | licence | note |
|---|---|---|---|
| student | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf` | Apache-2.0 | same weights as Study 01 |
| scale check | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4c` | Apache-2.0 | same tokenizer and template |
| second family | `LFM2.5-1.2B-Instruct` | LFM Open License v1.0, research use permitted, not OSI | different lab and architecture; if an OSI licence is required, `SmolLM2-1.7B-Instruct` (Apache-2.0) |
| teacher | the largest Apache-2.0 Qwen instruct model that fits the GPU, from Qwen3.5-9B upward | to confirm on the Hub | chosen by verified acceptance rate on 200 practice tasks |

Qwen3-0.6B and Qwen3-1.7B are the natural students for a study that starts fresh. They are not the
primary choice here because the gap being closed was measured on Qwen2.5-0.5B.

## 5. Cloud execution path

**Provider: Hugging Face Jobs.** A job runs one command in a container on one GPU and stops when the
command exits. It suits a workflow driven from a session without a GPU: it is plain HTTPS with one
token, has a per-job timeout, bills per minute against prepaid credits, and mounts Hub storage for
checkpoints. Modal offers the same model and is the fallback. RunPod is cheaper per hour but needs
pod lifecycle scripting and SSH, which this session's HTTPS-only proxy probably cannot provide.

| flavour | price per hour | use |
|---|---|---|
| L4 24 GB | $0.80 | default for 0.5B and 1.5B students |
| A10G 24 GB | $1.00–1.50 | alternative if faster per dollar |
| L40S 48 GB | $1.80 | teacher generation |
| A100 80 GB | $2.50 | a 27B-class teacher, if chosen |

**One job.**

```
hf jobs run --flavor l4x1 --timeout 90m --secrets HF_TOKEN \
    -v hf://buckets/<account>/ccl-study02:/out  <pytorch-cuda-image> \
    bash -c "git clone https://github.com/JMBorrill/cognitive-core-lab && cd cognitive-core-lab \
             && git checkout <commit> && pip install -r requirements-cuda.lock \
             && python scripts/s2_run.py configs/study02/<experiment>/<run>.json --out /out"
```

The exact flags are confirmed against the current CLI in WP3. The same `s2_run.py` command runs on any
CUDA machine, so changing provider changes only the wrapper.

**Checkpointing and resumption.** As in Study 01: adapter, optimizer state and step are written
atomically every 100 steps, evaluation appends per-episode rows, and a practice cycle records its
stage. Outputs live on the mounted bucket, so a job that times out or is pre-empted resumes when the
same command is submitted again. A CPU test interrupts a training run and checks that the resumed
adapter matches the uninterrupted one.

**Persistent artifacts.**

| artifact | where | in git |
|---|---|---|
| run config, `run.json` (commit, config hash, data hashes, GPU, seconds), `metrics.jsonl`, `train_log.jsonl` | bucket, then copied into `results/study02/runs/<run>/` | yes |
| final adapters | private Hub model repository | SHA-256 only |
| checkpoints, per-episode traces | bucket | no |
| `data/v2` tasks and manifest | repository | yes |

**Cost controls.**

1. Prepaid credits are the hard limit at account level. Load only the approved amount.
2. Every job has a timeout of twice its estimate.
3. `configs/study02/budget.json` holds a cap per experiment. `s2_launch.py` adds the estimate of a new
   job to the ledger total and refuses to submit beyond the cap.
4. One job at a time by default.
5. Each job appends its GPU-seconds and cost to `results/study02/ledger.jsonl`.
6. No experiment is queued without approval of its plan and cap.

**What this needs from you.** A Hugging Face account with credits; `huggingface.co` allowed in this
environment's network settings; a Hugging Face token stored as an environment secret. I will request
the network change and the credential through the environment tools when you approve the next stage.

## 6. Budget

Planning assumptions: LoRA training at 7,000 tokens per second and evaluation of 3,520 episodes in
about 10 minutes on an L4. Both are extrapolated from published figures for other models and must be
treated as ±2×. Experiment 0 begins with a ten-minute calibration that replaces them.

| experiment | runs | GPU-hours with 1.5× contingency | estimate | cap requested |
|---|---|---|---|---|
| 0 bridge | 6 reproductions, 6 settings, calibration | 5 | $4 | $60 for 0 and 1 together |
| 1 diversity | 26 training runs, baselines | 28 | $23 | |
| 2 signal × curriculum | about 40 runs, teacher generation, second family | 120 | $110 | $200 |
| 3 retention | about 40 runs | 60 | $50 | $100 |
| 4 teacher dependence | about 40 runs | 90 | $70 | $150 |
| **total** | | **about 300** | **about $260** | **$510** |

Storage fits in the 100 GB of private storage a free account includes. Whether Jobs needs a PRO
subscription (about $9 a month) was not confirmed.

Optional and only with Anthropic's authorisation: a Fable 5.2 teacher arm of about 3,700 episodes.
At $8 and $40 per million input and output tokens, with roughly 4,500 input and 600 output tokens per
episode (thinking is always on and billed), that is about $220, or $110 through the Batches API run
one turn per batch.

## 7. Experiment tracking

The repository stays the record: one directory per run with JSONL metrics, and tables regenerated by
script, as in Study 01. Additions:

- `results/study02/ledger.jsonl`: one line per job with run, commit, GPU-seconds and cost.
- `results/study02/test_access.jsonl`: one line each time a sealed split is scored.
- Optional: Trackio as a live mirror of training curves for long jobs. It is not a source of record.

## 8. Work packages

Each package ends with tests that run on CPU and, where marked, a GPU acceptance check.

| | package | delivers | accepted when |
|---|---|---|---|
| WP0 | this proposal | documents, audit script, v1 reproducibility guard | reviewed |
| WP1 | PyTorch backend | `ccl/sft.py`, `ccl/hf.py`, `requirements-cuda.lock` | on CPU with a tiny randomly initialised model: loss covers assistant tokens only; packed episodes equal per-turn examples; batched generation equals unbatched; interrupted and resumed training match. Study 01 tests and hash guard still pass |
| WP2 | v2 environment | `ccl/world2.py`, `ccl/compose.py`, `scripts/build_v2.py`, `data/v2` | oracle solves 100% of generated tasks; degenerate policies fail; ≥ 5,000 practice families; no test family, held-out edge or held-out tool in any training file; rebuild is byte-identical; phrase templates read naturally in a manual sample of 100 |
| WP3 | cloud path and Experiment 0 | `scripts/s2_run.py`, `scripts/s2_launch.py`, budget and ledger | launcher refuses over-budget jobs in a dry run; a smoke job writes to the bucket and resumes; Experiment 0 meets its pass criteria |
| WP4 | Experiment 1 | quality gate run, frozen test set, 26 runs, `scripts/s2_analyze.py`, report | pre-registered analysis produced; stop/go decision recorded |
| WP5 | Experiment 2 | `ccl/loop.py`, teacher generation, pilot, full run | pilot meets its criteria; analysis produced |
| WP6 | Experiment 3 | sequential stages, replay, router, merge | analysis produced |
| WP7 | Experiment 4 | seed-size runs, teacher on demand | analysis produced |

WP1 and WP2 are independent and need no GPU or credentials. WP3 is the first step that spends money.

## 9. Order of work after approval

1. WP1 and WP2 in this kind of session, with CPU tests.
2. Environment access and credits; WP3 smoke job; Experiment 0.
3. Review of Experiment 0 against its plan; freeze `data/v2` and the Experiment 1 plan.
4. Experiment 1; stop/go review.
5. Pre-register Experiment 2 with the measured variance and costs; pilot; full run.
