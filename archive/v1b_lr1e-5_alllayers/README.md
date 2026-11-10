# Failed experiment: lr 1e-5 with clipping, LoRA on all 24 layers (preserved)

Chosen by the 60-step lr pilot (`lr_pilot_PLAN.md`, `experiments/lr_pilot`). The first full run, B_s0, was
stopped after 43 steps on 2026-11-09: the pre-clip gradient norm rose from ~1e3 to ~2.8e7 and the training
loss trended upward. This led to the gradient diagnosis and stability study in `experiments/grad_diag`.
No evaluation was run; nothing here is used in any reported comparison.
