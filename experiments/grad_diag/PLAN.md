# LoRA stability study (2026-11-09), written before running

Finding so far (init_grads*.log): at initialisation the gradient is dominated by `v_proj.lora_b` in the
earliest LoRA layers (layer 0 when all 24 layers are adapted) and varies ~20x between batches. With LoRA
scale 20, lr 1e-5 still drove the pre-clip gradient norm from ~1e3 to ~1e7 within 40 steps on B
(archive: runs/B_s0 at lr 1e-5 below) and to ~1e5 on Dmix (lr_pilot).

Note: single-batch loss at init already ranges 0.32-0.80, so the lr-pilot rule (raw loss > 2x mean of
steps 1-10) is noise-sensitive. This study uses a smoothed criterion.

Candidates, all on B (worst case observed), seed 0, the 250-step schedule stopped after 100 steps, clip 1.0:
  (a) num_layers 16 (last 16; MLX-LM default), scale 20, lr 1e-5
  (b) num_layers 24 (all),                 scale 2,  lr 1e-4   (alpha = 2r convention)

Stable iff (i) 5-step moving-average loss after step 20 never exceeds 1.5x the moving average at step 10,
and (ii) the median pre-clip grad norm over steps 81-100 is < 10x the median over steps 1-10.
Pick the stable candidate with the lowest val loss at step 100 (oracle_val; tie-break: lower step-50 val).
If neither is stable, stop and report.
