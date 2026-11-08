"""Run the full experiment queue in priority order. Every step is idempotent/resumable.

    python scripts/run_queue.py            # run everything not yet done
    python scripts/run_queue.py --dry-run  # list steps and their status
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
CATALOG = ["eval_catalog_0", "eval_catalog_5", "eval_catalog_12"]
MAIN = ["eval_id", "eval_ood", "eval_unseen", "eval_renamed"]


def n_lines(p: Path) -> int:
    return sum(1 for _ in open(p)) if p.exists() else 0


def task_count(splits) -> int:
    return sum(n_lines(Path(f"data/v1/tasks/{s}.jsonl")) for s in splits)


def train_step(cfg: str, seed: int):
    name = f"{cfg}_s{seed}"
    return {"name": f"train {name}", "done": lambda: Path(f"runs/{name}/DONE").exists(),
            "cmd": [PY, "-m", "ccl.train", f"configs/train/{cfg}.json", "--set", f'out_dir="runs/{name}"',
                    f"seed={seed}"]}


def eval_step(name: str, splits, extra=()):
    need = task_count(splits)
    return {"name": f"eval {name}", "done": lambda: n_lines(Path(f"results/runs/{name}/metrics.jsonl")) >= need,
            "cmd": [PY, "scripts/evaluate.py", "--name", name, "--splits", *splits, *extra]}


def steps():
    s = [eval_step("A_base", MAIN + CATALOG), eval_step("A_fewshot", MAIN + CATALOG, ["--fewshot"])]
    for cfg in ["B", "C", "Dmix", "Dseq"]:
        s += [train_step(cfg, 0), eval_step(f"{cfg}_s0", MAIN + CATALOG, ["--adapter", f"runs/{cfg}_s0"])]
    for seed in (1, 2):
        for cfg in ["B", "C", "Dmix"]:
            s += [train_step(cfg, seed), eval_step(f"{cfg}_s{seed}", MAIN, ["--adapter", f"runs/{cfg}_s{seed}"])]
    for cfg in ["C_matched", "C_25", "C_50", "C_full"]:
        s += [train_step(cfg, 0), eval_step(f"{cfg}_s0", MAIN, ["--adapter", f"runs/{cfg}_s0"])]
    s += [eval_step("teacher", MAIN, ["--model", "models/Qwen3.5-4B-MLX-4bit", "--batch", "24"])]
    for seed in (1, 2):
        s += [train_step("Dseq", seed), eval_step(f"Dseq_s{seed}", MAIN, ["--adapter", f"runs/Dseq_s{seed}"])]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="regex; run only steps whose name matches")
    a = ap.parse_args()
    log = Path("logs/queue.jsonl")
    log.parent.mkdir(exist_ok=True)
    import re
    for st in steps():
        if a.only and not re.search(a.only, st["name"]):
            continue
        if st["done"]():
            print(f"[queue] done  {st['name']}")
            continue
        if a.dry_run:
            print(f"[queue] TODO  {st['name']}: {' '.join(st['cmd'])}")
            continue
        print(f"[queue] start {st['name']}", flush=True)
        t0 = time.time()
        safe = st["name"].replace(" ", "_")
        with open(f"logs/{safe}.log", "a") as out:
            rc = subprocess.run(st["cmd"], stdout=out, stderr=subprocess.STDOUT).returncode
        rec = {"step": st["name"], "rc": rc, "seconds": round(time.time() - t0), "ended": time.strftime("%F %T")}
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[queue] {'ok' if rc == 0 else 'FAILED'} {st['name']} ({rec['seconds']}s)", flush=True)
        if rc != 0 and st["name"].startswith("train"):
            print("[queue] training failed; dependent eval will be skipped", flush=True)


if __name__ == "__main__":
    main()
