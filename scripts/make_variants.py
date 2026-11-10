"""Build the training-data variants and their train configs once teacher data exists.

Budget matching: every variant trains for the same number of optimizer steps with
the same batch (8 episodes/step), and every single-source variant has exactly
N_EP = 1,000 episodes. The oracle variants depend only on the task pool, so they
can be built (`--oracle-only`) before teacher generation finishes.

  B          1,000 random teacher-accepted episodes              (N_EP)
  C          oracle episodes on 1,000 random tasks offered to the teacher (N_EP)
  Dmix       B ∪ C shuffled together                             (2·N_EP)
  Dseq       B for half the steps, then C for the other half
  C_matched  oracle on exactly B's accepted tasks (isolates label source from task coverage)
  C_25/C_50  25% / 50% of C (data quantity)
  C_full     oracle on all 3,000 training tasks (data quantity, upward)
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccl.data import read_jsonl, write_jsonl  # noqa: E402

DATA = Path("data/v1")
STEPS = 250
N_EP = 1000
OFFERED = 1600  # tasks offered to the teacher: the first 1,600 of the training pool
BASE = {"model": "models/Qwen2.5-0.5B-Instruct", "val_file": str(DATA / "sft/oracle_val.jsonl"),
        "steps": STEPS, "batch_size": 4, "grad_accum": 2, "lr": 1e-5, "max_grad_norm": 1.0, "warmup": 20,
        "lora": {"rank": 16, "scale": 20.0, "dropout": 0.0}, "save_every": 50, "eval_every": 50}
# LoRA stability settings chosen by the pre-registered study in experiments/grad_diag (if present)
_OVR = ROOT / "configs/train_overrides.json" if "ROOT" in globals() else Path("configs/train_overrides.json")
if _OVR.exists():
    BASE.update(json.loads(_OVR.read_text()))


def episodes(rows):
    by = {}
    for r in rows:
        by.setdefault(r["meta"]["task_id"], []).append(r)
    return by


def main():
    oracle_only = "--oracle-only" in sys.argv
    oracle = episodes(read_jsonl(DATA / "sft/oracle_train.jsonl"))
    pool_ids = [json.loads(l)["task_id"] for l in open(DATA / "tasks/train_pool.jsonl")]
    offered_ids = pool_ids[:OFFERED]
    rng = random.Random(0)
    c_ids = rng.sample(offered_ids, N_EP)
    variants = {
        "C": [r for t in c_ids for r in oracle[t]],
        "C_25": [r for t in c_ids[: N_EP // 4] for r in oracle[t]],
        "C_50": [r for t in c_ids[: N_EP // 2] for r in oracle[t]],
        "C_full": [r for t in pool_ids for r in oracle[t]],
    }
    if not oracle_only:
        t_eps = episodes(read_jsonl(DATA / "sft/teacher_train.jsonl"))
        if len(t_eps) < N_EP:
            raise SystemExit(f"only {len(t_eps)} accepted teacher episodes (< {N_EP}); lower N_EP and rebuild all")
        b_ids = random.Random(1).sample(sorted(t_eps), N_EP)
        variants["B"] = [r for t in b_ids for r in t_eps[t]]
        variants["C_matched"] = [r for t in b_ids for r in oracle[t]]
    info = json.loads((DATA / "sft/variants.json").read_text()) if (DATA / "sft/variants.json").exists() else {}
    for name, rows in variants.items():
        sha = write_jsonl(DATA / f"sft/variant_{name}.jsonl", rows)
        info[name] = {"episodes": len({r["meta"]["task_id"] for r in rows}), "rows": len(rows), "sha256": sha,
                      "families": dict(Counter(r["meta"]["family"] for r in rows if r["meta"]["turn"] == 1))}
    print(json.dumps({k: {x: v[x] for x in ("episodes", "rows")} for k, v in info.items()}, indent=1))
    (DATA / "sft/variants.json").write_text(json.dumps(info, indent=2))

    cfgdir = Path("configs/train")
    cfgdir.mkdir(parents=True, exist_ok=True)
    f = lambda n: str(DATA / f"sft/variant_{n}.jsonl")
    specs = {
        "B": {"train_files": [f("B")]},
        "C": {"train_files": [f("C")]},
        "Dmix": {"train_files": [f("B"), f("C")]},
        "Dseq": {"stages": [{"train_files": [f("B")], "steps": STEPS // 2},
                            {"train_files": [f("C")], "steps": STEPS - STEPS // 2}]},
        "C_matched": {"train_files": [f("C_matched")]},
        "C_25": {"train_files": [f("C_25")]},
        "C_50": {"train_files": [f("C_50")]},
        "C_full": {"train_files": [f("C_full")]},
    }
    for name, spec in specs.items():
        (cfgdir / f"{name}.json").write_text(json.dumps({**BASE, **spec}, indent=2))
    print("wrote", sorted(p.name for p in cfgdir.iterdir()))


if __name__ == "__main__":
    main()
