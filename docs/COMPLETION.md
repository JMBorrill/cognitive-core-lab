# Completion matrix

Status as of 2026-11-10. **Done** = implemented and verified from artifacts. **Partial** = done with a
stated gap. **Not done** = not attempted, with the reason and what it would take.

| # | Requirement | Status | Evidence / gap |
|---|---|---|---|
| 1 | Research and model selection (0.5B–1.5B, fallback, revisions, licences) | Done | [MODEL_SELECTION.md](MODEL_SELECTION.md). Student `Qwen2.5-0.5B-Instruct` @ `7ae557604adf` (Apache-2.0), chosen after Qwen3.5-0.8B ran out of memory at batch 4. Fallback `Qwen2.5-1.5B-Instruct` documented but not exercised. Teacher weights were not revision-pinned at download (recorded HEAD instead). |
| 2 | Infrastructure: simulators, typed schemas, traces, DAGs, seeds, configs, tests | Done | `ccl/`; 28 unit tests + slow resume-equivalence test pass. Simulator crash on malformed arrays found during teacher generation and fixed, with a regression test. |
| 3 | Dataset: distractors, failures, permissions, changing catalogues, leakage prevention | Partial | `data/v1` reproduces byte-for-byte; template- and graph-level leakage audit in `data/v1/leakage_audit.json`. Gaps: eval_id/eval_renamed have no permission tasks (removed by the verbatim-overlap filter); the `un_timezone` template is flawed (REPORT §7.3). "Changing catalogues" is covered by renamed catalogues and varying distractors, not by tools changing mid-episode. |
| 4 | Distillation with recorded teacher identity, settings, cost, rejection rates | Done | `data/v1/teacher/manifest.json`, `pass{1,2}.jsonl.gz` (every episode with rejection reasons). Local 4B teacher (no hosted API available); cost $0; 1,011 / 1,600 tasks accepted. |
| 5 | Training: budget-matched, adapters, logs, hashes, resumable | Done | 16 final runs (`runs/*`: B, C, Dmix, Dseq × 3 seeds; 4 ablations × 1 seed), each 250 steps × 8 episodes, with logs, configs, adapter SHA-256. Adapters (22 MB each) are not committed. Two diverged configurations preserved in `archive/`. |
| 6 | Evaluation: all listed metrics, multiple seeds, CIs | Done | `results/tables.md`, `results/summary.json`: success, tool selection, arguments, dependency safety, recovery (fault breakdown), hallucinated tools, unnecessary calls, latency, throughput, memory. Hierarchical bootstrap CIs, paired permutation tests, Holm correction. |
| 7 | Ablations: data quantity, catalogue size, depth, unseen tools, combination strategy | Partial | All five reported. Data quantity and catalogue size are single-seed; depth is a breakdown (depth 4–5 strata have n=15), not a controlled ablation. |
| 8 | Agent runtime: limits, timeouts, permissions, human approval | Done | `ccl/runtime.py`; `scripts/agent.py` asks a human at the terminal before consequential calls. The live web mode uses scripted approval (per task, or always approve/deny), not a human prompt. |
| 9 | Next.js showcase: replay + local live mode, clearly labelled | Partial | `web/` builds; all pages return 200; replay shows recorded runs labelled REPLAY; live mode verified end to end through `/api/run` → `scripts/serve.py` with trained adapters. Visually reviewed in a browser, with layout fixes applied; not deployed. |
| 10 | Report, case study, CV summary, completion matrix | Done | [REPORT.md](REPORT.md), [CASE_STUDY.md](CASE_STUDY.md), [CV_SUMMARY.md](CV_SUMMARY.md), this file. |
| — | Repository `JMBorrill/cognitive-core-lab` | Done | Public at https://github.com/JMBorrill/cognitive-core-lab (code, data manifests, results, logs; no weights). |
| — | Adapters published | Not done | Not committed (size). Could be attached to a GitHub release (~350 MB for 16 runs). |

## Remaining work and how to do it

- **Second seeds for the ablations** (C_matched, C_25, C_50, C_full): add seeds to `scripts/run_queue.py`;
  about 50 min per run on the M4 (8 runs ≈ 7 h).
- **Fix `un_timezone`** (require only `get_timezone`, or make the request unanswerable from world
  knowledge) and re-evaluate eval_unseen as a v2 split: about 15 min per configuration run.
- **Stronger teacher** (hosted API): `scripts/distill.py` would need an API backend. Measured on the
  local teacher's evaluation run, an episode uses ~1,350 prompt and ~100 output tokens (Qwen
  tokenizer), so the same 2,261 episodes would be about 3.1M input and 0.2M output tokens (multiply by
  the provider's per-token prices; tokenizers differ).
- **Adapters**: attach the 16 final adapters (~350 MB) to a GitHub release.
- **Showcase deployment**: replay mode is static and could be hosted (e.g. Vercel); live mode needs the
  local inference server and stays local.
