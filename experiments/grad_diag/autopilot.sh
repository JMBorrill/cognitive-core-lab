#!/usr/bin/env bash
# Unattended: wait for the stability study, apply the pre-registered rule, then run the full pipeline.
set -uo pipefail
cd "$(dirname "$0")/../.."
P=.venv/bin/python
D=experiments/grad_diag
until grep -qE "\[study\] done|Traceback" $D/study.log; do sleep 60; done
$P $D/judge.py $D/a_last16_s20_lr1e-5 $D/b_all_s2_lr1e-4 | tee $D/verdict.txt
CHOICE=$($P - <<'PY'
import re
best = None
for line in open("experiments/grad_diag/verdict.txt"):
    if "stable True" not in line:
        continue
    name = line.split(":")[0].rsplit("/", 1)[-1]
    vals = dict((int(k), float(v)) for k, v in re.findall(r"(\d+): ([\d.]+)", line.split("val ")[1]))
    key = (vals.get(100, 9e9), vals.get(50, 9e9))
    if best is None or key < best[0]:
        best = (key, name)
print(best[1] if best else "none")
PY
)
echo "[autopilot] choice: $CHOICE $(date)" | tee -a $D/verdict.txt
case "$CHOICE" in
  a_last16_s20_lr1e-5) echo '{"num_layers": 16, "lr": 1e-5, "lora": {"rank": 16, "scale": 20.0, "dropout": 0.0}}' > configs/train_overrides.json ;;
  b_all_s2_lr1e-4)     echo '{"num_layers": 24, "lr": 1e-4, "lora": {"rank": 16, "scale": 2.0, "dropout": 0.0}}' > configs/train_overrides.json ;;
  *) echo "[autopilot] no stable candidate; NOT starting the queue" | tee -a $D/verdict.txt; exit 0 ;;
esac
echo "[pipeline] autopilot start $(date) with $(cat configs/train_overrides.json)" >> logs/pipeline.log
bash scripts/pipeline.sh >> logs/pipeline.log 2>&1
echo "[autopilot] pipeline exited rc=$? $(date)" | tee -a $D/verdict.txt
