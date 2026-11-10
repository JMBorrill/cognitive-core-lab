#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for lr in 1e-5 3e-5 1e-4; do
  .venv/bin/python -m ccl.train configs/train/Dmix.json --fresh --set out_dir="\"experiments/lr_pilot/lr$lr\"" lr=$lr stop_after=60 seed=0
done
echo "[pilot] done"
