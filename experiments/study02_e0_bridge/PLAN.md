# Study 02, Experiment 0: bridge to the PyTorch stack (plan written 2026-11-10, before any run)

Status: draft for review. It becomes binding when the first job is submitted; later changes are logged
under "Deviations".

## Purpose

1. Show that the PyTorch/PEFT trainer and evaluator reproduce Study 01.
2. Check whether the Study 01 generalisation gap survives conventional LoRA settings.
3. Fix the training settings and the GPU type for the rest of Study 02.

Data: `data/v1` only. Student: `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf`.

## Steps

**A. Calibration (about 10 minutes per GPU type).** On L4 and on one faster flavour: tokens per second
for 50 training steps of variant C, and episodes per second for `eval_id`. Choose the flavour with the
lower cost per training token. Replace the budget estimates in the implementation plan with the
measurements.

**B. Reproduction.** Variants B and C, seeds 0–2, with the Study 01 settings in PEFT form: rank 16,
`lora_alpha` 320, the seven linear modules of blocks 8–23 (5,865,472 trainable parameters), lr 1e-5,
20 warm-up steps, cosine decay to 1e-6, gradient clipping 1.0, 250 steps × 8 episodes, loss on
assistant tokens. Evaluate on the four main v1 splits with greedy decoding.

Pass: for each of B and C, the three-seed mean is within 5 points of the Study 01 value on `eval_id`
(B 86.0, C 89.7) and on `eval_ood` (B 30.3, C 30.3), and every run passes the Study 01 stability rule.

**C. Conventional settings.** Variant C, seed 0, all 24 blocks, rank 16, `lora_alpha` 32, learning rate
in {5e-5, 1e-4, 2e-4}, 250 and 500 steps: six runs. For each, executable success on the 150-task
`val_pool` is computed.

Rule: the Study 02 default is lr 1e-4 with two passes over the data. It is replaced by the setting with
the highest `val_pool` success only if the default is unstable or more than 3 points below that best
setting. `eval_ood` is reported for all six runs and plays no part in the choice, because the v1
splits are an evaluation anchor in Study 02.

**D. Repeatability.** One adapter from step B is evaluated twice. Report the share of episodes whose
success differs.

## Reading the result

| outcome | meaning | action |
|---|---|---|
| B passes | the stack is trusted | proceed to Experiment 1 |
| B fails | the port differs from Study 01 | find the difference before anything else runs |
| best `eval_ood` in step C ≥ 60% | the Study 01 gap was largely an artefact of its training settings | report it, re-baseline Study 01's conclusions, proceed |
| step D flip rate > 2% | evaluation noise is material | fix batch composition or precision before Experiment 1 |
| measured cost per run > 2× the estimate | budget is wrong | re-plan before Experiment 1 |

## Budget

About 5 GPU-hours. Shares the $60 cap with Experiment 1.

## Clarifications recorded before the first job (2026-11-10)

Approved for implementation on 2026-11-10 with a hard cap of 60 USD for Experiments 0 and 1 together. The points
below fill in details the plan left open. They were written after the code was tested on CPU and before any GPU
job was submitted, so no result influenced them.

1. **Optimizer.** MLX's `AdamW`, which Study 01 used, applies no bias correction by default
   (`p -= lr * m / (sqrt(v) + eps)`); PyTorch's does. Over 250 steps the uncorrected update is 2 to 6 times
   larger at the same nominal learning rate. Step B therefore uses the uncorrected update
   (`bias_correction: false`), which is what "the Study 01 settings" were. Step C uses the standard
   bias-corrected AdamW, since that is the conventional choice.
2. **Step C keeps everything else from step B**: 20 warm-up steps, cosine decay to 10% of the peak, gradient
   clipping at 1.0, 8 episodes per step as 4 x 2, maximum length 2,048, dropout 0.
3. **Precision.** Base weights in float32 with bfloat16 autocast on the GPU, for training and for generation.
4. **Decoding.** Greedy, 256 new tokens at most, 64 episodes in lock-step. The model's own generation defaults
   (sampling and a repetition penalty of 1.05) are switched off, as they were under MLX.
5. **Step A flavours.** `l4x1` (0.80 USD/h) and `a10g-small` (1.00 USD/h). Each calibration run trains variant C
   for 50 steps and scores `eval_id`.
6. **Stability rule.** The rule of `scripts/stability_log.py`: the 5-step moving average of the loss after
   step 20 never exceeds 1.5 times its value at step 10, and the median pre-clip gradient norm of the last 20
   steps is below 10 times the median of the first 10.
7. **Step C selection metric.** Executable success on the 150 tasks of `data/v1/tasks/val_pool.jsonl`, greedy.
   A setting is "unstable" if it fails the rule in point 6.
8. **Step D.** `e0_repro_B_s0` is scored a second time in a separate job.
9. **Cost of a run** is compared with the estimate of the implementation plan (5 GPU-hours and 4 USD for the
   whole of Experiment 0).

## Deviations

1. **Flavours used.** The reproduction (step B) was submitted on `l4x1`, the flavour that step A's rule selects
   (0.022 against 0.027 USD per million training tokens). Both jobs waited twelve minutes for hardware without
   starting, were cancelled at no cost, and were rerun on `a10g-small`. Steps C and D also ran on `a10g-small`.
   The GPU type changes cost and speed, not results.
2. **Flavour for Experiment 1.** Step A's rule compares cost per training token. Scoring, not training, turned
   out to dominate the cost of a run, and the A10G scores 1.7 times as many episodes per second for 1.25 times
   the price. Experiment 1 therefore uses `a10g-small`, chosen on cost per complete run.
3. **Adapter merged for scoring.** From the step C jobs onwards the low-rank update is folded into the base
   weights before scoring (the same function, tested to 1e-5 on logits). The two calibration runs scored the
   unmerged adapter.
4. **Order of scoring.** The step B and D jobs scored tasks grouped by depth and prompt length; the step C jobs
   scored them in file order. Batch composition affects numerical rounding only; step D found 0 of 720 episodes
   differing between two jobs with the same order.

## Outcome (2026-11-10)

All criteria are met: see `results/study02/e0/tables.md` and `decision.json`. The Study 02 default is kept
(rank 16, alpha 32, all blocks, bias-corrected AdamW, lr 1e-4, two passes). Experiment 0 used 1.9 GPU-hours
and about 2.1 USD including two calibration jobs, against an estimate of 5 GPU-hours and 4 USD.
