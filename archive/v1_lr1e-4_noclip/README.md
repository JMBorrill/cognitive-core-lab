# Failed experiment: v1 training at lr 1e-4 without gradient clipping (preserved)

All configurations were trained with lr 1e-4 (20-step linear warmup, cosine decay), LoRA rank 16 /
scale 20, AdamW, and **no gradient clipping**. In 10 of 11 runs the training loss rose from ~0.2 to
6–14 as warmup reached the peak learning rate (steps ~15–30) and never fully recovered; the resulting
adapters emit malformed output or finalise without calling tools. Only C_s2 recovered (val loss 0.035).

Diagnosis: lr 1e-4 is 10x the MLX-LM default for scale-20 LoRA, and without clipping single large
updates destabilise the adapter. The queue was stopped on 2026-11-09 after 23 steps; results here are
**not** used in any reported comparison, except as this negative finding.

`runs/*/train_log.jsonl` hold the loss curves; `results/*/metrics.jsonl` the evaluation of each
diverged adapter; `configs/` the exact training configs; `queue.jsonl` the queue log.
