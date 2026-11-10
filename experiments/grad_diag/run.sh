#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
P=.venv/bin/python
$P -m ccl.train configs/train/B.json --fresh --set out_dir='"experiments/grad_diag/a_last16_s20_lr1e-5"' num_layers=16 lr=1e-5 stop_after=100 seed=0
$P -m ccl.train configs/train/B.json --fresh --set out_dir='"experiments/grad_diag/b_all_s2_lr1e-4"' num_layers=24 'lora={"rank":16,"scale":2.0,"dropout":0.0}' lr=1e-4 stop_after=100 seed=0
echo "[study] done"
