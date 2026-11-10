"""Study 02: train the adapters of one or more run configs. Resumable; never reads a test split.

    python scripts/s2_train.py configs/study02/e0/e0_repro_C_s0.json [...] --out runs2 [--bucket /out]

Inputs are the Study 01 training files (`data/v1/sft`, rebuilt by scripts/s2_prepare_v1.py) or the
Study 02 practice space and `dev` (`ccl.s2.load_practice`, `data/v2/dev`). Held-out scoring is done by
scripts/s2_score.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccl import s2  # noqa: E402
from ccl.data import write_jsonl  # noqa: E402


def training_files(cfg: dict, data_dir: Path) -> tuple[list[str], str | None, dict]:
    """Materialise the training and validation files of a run and check them against the config's hashes."""
    d = cfg["data"]
    if d["kind"] == "v1":
        from ccl.sft import file_sha256
        path = ROOT / f"data/v1/sft/variant_{d['variant']}.jsonl"
        sha = file_sha256(str(path))
        assert sha == d["sha256"], f"variant_{d['variant']}: {sha[:12]} != {d['sha256'][:12]}"
        return [str(path)], str(ROOT / "data/v1/sft/oracle_val.jsonl"), {"sha256": sha}
    assert d["kind"] == "e1"
    fams, tools = s2.load_practice()
    rows, stats = s2.oracle_rows(s2.e1_train_tasks(fams, tools, d["replicate"], d["D"]))
    sha = s2.rows_sha256(rows)
    assert sha == d["sha256"], f"E1 training set r{d['replicate']} D{d['D']}: {sha[:12]} != {d['sha256'][:12]}"
    path = data_dir / f"e1_r{d['replicate']}_D{d['D']}.jsonl"
    assert write_jsonl(path, rows) == sha
    val = data_dir / "dev_val.jsonl"
    if not val.exists():  # validation loss only: oracle trajectories on the first 80 dev tasks
        write_jsonl(val, s2.oracle_rows(s2.load_split("dev")[:80])[0])
    return [str(path)], str(val), {"sha256": sha, **stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--out", default="runs2")
    ap.add_argument("--bucket", default=None, help="mounted storage directory to mirror outputs to")
    a = ap.parse_args()
    from ccl import hf
    out_root = Path(a.out)
    bucket = Path(a.bucket) if a.bucket else None
    for path in a.configs:
        cfg = json.loads(Path(path).read_text())
        if not cfg.get("train"):
            continue
        out = out_root / cfg["experiment"] / cfg["name"]
        remote = bucket / cfg["experiment"] / cfg["name"] if bucket else None
        s2.restore(remote, out)
        if (out / "train/DONE").exists():
            print(f"[s2_train] {cfg['name']}: already trained", flush=True)
            continue
        t0 = time.time()
        files, val, data = training_files(cfg, out_root / "_data")
        tcfg = {**cfg["train"], "model": s2.fetch_model(cfg["model"]), "train_files": files, "val_file": val,
                "out_dir": str(out / "train"), "seed": cfg["seed"]}
        summary = hf.train(tcfg, on_checkpoint=lambda: s2.sync(out, remote))
        (out / "train_run.json").write_text(json.dumps(
            {"name": cfg["name"], "config_sha256": s2.config_sha256(cfg), "data": data, "summary": summary,
             "seconds_this_invocation": round(time.time() - t0, 1), **s2.environment()}, indent=1))
        s2.sync(out, remote)
        print(f"[s2_train] {cfg['name']}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
