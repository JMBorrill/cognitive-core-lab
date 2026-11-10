# LR pilot (2026-11-09), written before running

Trigger: v1 runs at lr 1e-4 without clipping diverged (archive/v1_lr1e-4_noclip).
All candidates now use global grad-norm clipping at 1.0.

Data: Dmix (B ∪ C), so the choice is not tuned to one source. Seed 0. Identical 250-step schedule
(warmup 20, cosine), stopped after step 60 via `stop_after`; val loss on oracle_val at step 50.

Candidates: lr 1e-5, 3e-5, 1e-4.

Rule: a run is *unstable* if any training loss after step 10 exceeds 2x the mean loss of steps 1-10.
Pick the stable candidate with the lowest step-50 val loss. The chosen lr (with clipping) is then used
for every configuration; nothing is tuned per configuration.

## Outcome

| lr | mean loss steps 1-10 | max loss after step 10 | unstable | val loss @50 | max grad norm |
|---|---|---|---|---|---|
| 1e-5 | 0.324 | 0.492 | no | 0.0773 | 1.6e5 |
| 3e-5 | 0.318 | 13.906 | yes | 9.34 | 3.7e15 |
| 1e-4 | 0.327 | 18.073 | yes | 13.04 | 1.2e15 |

Chosen: **lr 1e-5** with clipping 1.0 (the only stable candidate; also the MLX-LM default for scale-20 LoRA).
Caveat: even at 1e-5 the pre-clip gradient norm grew from ~7 (step 30) to ~1.6e5 (step 52) while the loss
stayed low, so full-length stability is checked on every run (loss curves are reported per run).
