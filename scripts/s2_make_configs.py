"""Write the run configs of Study 02, Experiments 0 and 1.

    python scripts/s2_make_configs.py e0     # configs/study02/e0/*.json
    python scripts/s2_make_configs.py e1     # configs/study02/e1/*.json (needs results/study02/e0/decision.json)

A run config names its model revision, its training data (with the SHA-256 the data must have when it is
rebuilt inside a job), its training settings, and the splits to score. Experiment 1 takes its training
settings from the decision recorded at the end of Experiment 0 and from nowhere else.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccl import s2  # noqa: E402

CFG = ROOT / "configs/study02"
S05 = {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "7ae557604adf67be50417f59c2c2f167def9a775"}
S15 = {"repo": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}
V1_MAIN = ["v1/eval_id", "v1/eval_ood", "v1/eval_unseen", "v1/eval_renamed"]
V2_ALL = ["v2/dev", "v2/test_id", "v2/test_comp", "v2/test_depth", "v2/test_tool", "v2/test_domain"]

# Study 01's settings in PEFT terms: MLX-LM scale 20 at rank 16 is lora_alpha 320; the last 16 of 24
# blocks; MLX's Adam has no bias correction.
STUDY01 = {"steps": 250, "batch_size": 4, "grad_accum": 2, "lr": 1e-5, "warmup": 20, "max_grad_norm": 1.0,
           "weight_decay": 0.0, "bias_correction": False, "max_seq_len": 2048, "save_every": 50, "eval_every": 50,
           "lora": {"rank": 16, "alpha": 320, "dropout": 0.0, "layers": list(range(8, 24))}}
# Conventional settings: every block, alpha = 2 x rank, bias-corrected AdamW.
CONVENTIONAL = {**STUDY01, "bias_correction": True, "lora": {"rank": 16, "alpha": 32, "dropout": 0.0, "layers": None}}


def write(exp: str, cfg: dict) -> None:
    d = CFG / exp
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{cfg['name']}.json").write_text(json.dumps({"experiment": exp, **cfg}, indent=1) + "\n")


def e0() -> None:
    variants = json.loads((ROOT / "data/v1/sft/variants.json").read_text())
    data = lambda v: {"kind": "v1", "variant": v, "sha256": variants[v]["sha256"]}  # noqa: E731
    for flavor in ("l4x1", "a10g-small"):  # step A: 50 training steps of variant C, then eval_id
        write("e0", {"name": f"e0_calib_{flavor}", "model": S05, "seed": 0, "data": data("C"),
                     "train": {**STUDY01, "steps": 50}, "score": ["v1/eval_id"]})
    for v in ("B", "C"):  # step B: reproduction
        for seed in (0, 1, 2):
            write("e0", {"name": f"e0_repro_{v}_s{seed}", "model": S05, "seed": seed, "data": data(v),
                         "train": STUDY01, "score": V1_MAIN})
    for lr in (5e-5, 1e-4, 2e-4):  # step C: conventional settings on variant C
        for steps in (250, 500):
            write("e0", {"name": f"e0_conv_C_lr{lr:g}_st{steps}", "model": S05, "seed": 0, "data": data("C"),
                         "train": {**CONVENTIONAL, "lr": lr, "steps": steps}, "score": ["v1/val_pool"] + V1_MAIN})
    # step D: the same adapter scored a second time
    write("e0", {"name": "e0_repeat_B_s0", "model": S05, "seed": 0, "adapter_from": "e0_repro_B_s0", "score": V1_MAIN})


def e1() -> None:
    decision = json.loads((ROOT / "results/study02/e0/decision.json").read_text())
    # 800 steps x 8 episodes = two passes over 3,200 episodes. The length limit is raised so that no episode is
    # dropped: the longest oracle episode of the practice space is a little over 2,048 tokens.
    train = {**decision["train"], "steps": 800, "save_every": 100, "eval_every": 100, "max_seq_len": 4096}
    fams, tools = s2.load_practice()
    score = ["own"] + V2_ALL + V1_MAIN
    coverage = {}
    for r in range(5):
        for d in s2.E1_LEVELS:
            rows, stats = s2.oracle_rows(s2.e1_train_tasks(fams, tools, r, d))
            assert stats["rejected"] == 0 and stats["episodes"] == 3200
            data = {"kind": "e1", "replicate": r, "D": d, "sha256": s2.rows_sha256(rows)}
            coverage[f"r{r}_D{d}"] = {**s2.subset_stats(s2.e1_order(fams, r)[:d]), **stats,
                                      "family_ids": [f["id"] for f in s2.e1_order(fams, r)[:d]]}
            write("e1", {"name": f"e1_s05_D{d}_r{r}", "model": S05, "seed": r, "data": data, "train": train,
                         "score": score, "eval_batch": 128})
            if d in (25, 1600) and r < 3:
                # the larger student needs gradient checkpointing to fit the longest episodes on a 24 GB card
                write("e1", {"name": f"e1_s15_D{d}_r{r}", "model": S15, "seed": r, "data": data,
                             "train": {**train, "grad_checkpoint": True},
                             "score": score, "eval_batch": 64})
    for tag, model, batch in (("s05", S05, 128), ("s15", S15, 64)):
        write("e1", {"name": f"e1_{tag}_base", "model": model, "seed": 0, "score": V2_ALL + V1_MAIN,
                     "eval_batch": batch})
        write("e1", {"name": f"e1_{tag}_fewshot", "model": model, "seed": 0, "fewshot": True,
                     "score": V2_ALL + V1_MAIN, "eval_batch": batch})
    out = ROOT / "results/study02/e1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_sets.json").write_text(json.dumps(coverage, indent=1) + "\n")


if __name__ == "__main__":
    {"e0": e0, "e1": e1}[sys.argv[1]]()
    print("wrote", len(list((CFG / sys.argv[1]).glob("*.json"))), "configs in", CFG / sys.argv[1])
