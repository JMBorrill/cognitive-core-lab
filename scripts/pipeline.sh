#!/usr/bin/env bash
# Single GPU lane: teacher generation, then every training/evaluation run, then analysis.
# Each stage is resumable; re-running this script continues where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY scripts/distill.py --n 1600 --batch 16
$PY scripts/make_variants.py
$PY scripts/run_queue.py
$PY scripts/stability_log.py
$PY scripts/analyze.py > /dev/null
$PY scripts/export_web.py
echo "[pipeline] complete"
