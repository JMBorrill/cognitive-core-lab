# Experiment 2, amendment 2: preparation of Stage 3

*Opened 2026-11-11 on the owner's instruction of that date ("Proceed with Stage 3 preparation only. Do not
launch the main experiment or score any sealed split."). It amends the [plan](PLAN.md) and
[amendment 1](AMENDMENT_1.md). The sections are recorded in the order their work is done, each before the
work it governs; the commit history dates them. Section 3 was recorded first, before the one paid job of
this preparation. Nothing here starts Stage 3 training or scores a sealed split.*

## 3. Teacher re-qualification under protocol 2 (recorded before its job)

**Why.** Under protocol 2 no teacher qualifies on the Stage 1 data (amendment 1, §7): `Qwen/Qwen3-32B` has
67.5% overall and 30% at four hops, `Qwen/Qwen3-14B` 55.5% and 18%. That reading is of attempts collected
under protocol 1, where a task whose greedy attempt succeeded by the shortcut got no retry. The owner
approved a re-qualification with a hard cap of 2 USD and unchanged thresholds.

**Unchanged.** The thresholds (accepted on at least 60% of the 200 tasks and on at least 40% of the four-hop
ones); the 200 tasks (the first 50 of each number of hops in replicate 0's stream); the few-shot prompt; one
greedy attempt and one retry sampled at temperature 0.7, top-p 0.95, seed `e2|teacher`; the engine settings
of the Stage 1 qualification (R1 to R4, 16 rows, token budget 28,000; R6 is not used); the code path
(`scripts/s2_e2.py teacher`). The licence requirement of the plan.

**What protocol 2 changes, by construction.** Acceptance is `ccl.s2.rejection(..., protocol=2)`: an attempt
that reaches the right answer by the shortcut, or gives a fact for the wrong part, is rejected, and the task
then gets its retry like any other rejected task. The result file records `protocol: 2`.

**Candidate.** `Qwen/Qwen2.5-32B-Instruct` at revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`
(Apache-2.0, not gated, dense, 32.8 billion parameters, `Qwen2ForCausalLM`: the architecture family of the
student, which is the one the engine was calibrated on). Reasons, all known before the job:

- Re-running `Qwen3-32B` is not expected to qualify. Of its 50 greedy four-hop attempts, 13 are accepted
  under protocol 2, 13 succeed by the shortcut and 24 fail; its four-hop retries were accepted in 2 of 24
  cases. With retries for the 13 as well, about 16 of 50 is the expectation, and 20 are needed.
- The newest Apache-2.0 candidates of similar size (`Qwen/Qwen3.8-27B`, `Qwen3.6-27B`, `Qwen3.5-27B`) mix
  attention layers with recurrent layers (`layer_types`: `full_attention`, `linear_attention`). The engine
  keeps a batch's cache between turns and switches off, by attention mask, positions that belong to no row's
  prompt. A recurrent state cannot be masked after the fact, so the engine would compute wrong continuations
  for these models, and a correct rollout path for them is new work that cannot be validated inside 2 USD.
  They are not tested here. This is listed as an open option for the owner.
- Mixture-of-experts candidates were excluded in Stage 1 for their cost on this engine (plan, deviation 3).

**Rule for a second candidate.** After the first job settles, a second candidate
(`Qwen/Qwen2.5-Coder-32B-Instruct`, Apache-2.0, same architecture) is tested only if the first did not
qualify and what is left of the cap covers the worst case of a job whose time limit is at least 1.25 times
the first job's measured duration. Otherwise it is not run.

**Budget.** `configs/study02/budget_e2_teacher2.json`: cap 2.00 USD, own ledger
(`results/study02/e2/ledger_teacher2.jsonl`), provider baseline 56.192762 USD. One job on `a100-large`
(2.50 USD per hour) with a 40-minute limit: worst case 1.75 USD. The Stage 1 run of a model of the same size
took 20 minutes (0.88 USD).

**Reading the result.** Qualifies: both thresholds met on a complete run (both passes finished). Reported
beside it: acceptance by hops, the count of attempts rejected for the shortcut and for a wrong part, and the
acceptance the same attempts would have under protocol 1. Every accepted attempt is replayed on a fresh
simulator in a separate process (`scripts/s2_e2.py replay`). **If no candidate qualifies**, H2(b) is
reported as not tested, as the plan provides (§3), Stage 4 is not proposed, and the thresholds stay as
they are. The accepted demonstrations of a qualifying teacher are not used for anything in this preparation.
