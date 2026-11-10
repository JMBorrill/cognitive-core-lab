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

## Deviations

None yet.
