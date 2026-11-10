"""Rebuild the Study 01 training files that Experiment 0 needs, and check them against the committed hashes.

    python scripts/s2_prepare_v1.py

The task pools and oracle data come from `scripts/build_datasets.py`; the teacher data is read from the
committed generation logs (`data/v1/teacher/pass{1,2}.jsonl.gz`), exactly as `scripts/distill.py` assembles it,
so no teacher model is needed. Nothing tracked by git changes: every rebuilt file is compared with the
hash recorded when Study 01 ran.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccl.data import write_jsonl  # noqa: E402
from ccl.sft import file_sha256  # noqa: E402

DATA = ROOT / "data/v1"


def main():
    before = json.loads((DATA / "manifest.json").read_text())["files"]
    variants = json.loads((DATA / "sft/variants.json").read_text())
    subprocess.run([sys.executable, "scripts/build_datasets.py"], cwd=ROOT, check=True, capture_output=True)
    after = json.loads((DATA / "manifest.json").read_text())["files"]
    assert {k: v["sha256"] for k, v in after.items()} == {k: v["sha256"] for k, v in before.items()}, \
        "data/v1 did not rebuild byte for byte"

    rows = []
    for p in (1, 2):
        with gzip.open(DATA / f"teacher/pass{p}.jsonl.gz", "rt") as f:
            rows += [e for line in f for r in [json.loads(line)] if r["accepted"] for e in r["examples"]]
    sha = write_jsonl(DATA / "sft/teacher_train.jsonl", rows)
    want = json.loads((DATA / "teacher/manifest.json").read_text())["sft_sha256"]
    assert sha == want, f"teacher_train.jsonl {sha[:12]} != {want[:12]}"

    subprocess.run([sys.executable, "scripts/make_variants.py"], cwd=ROOT, check=True, capture_output=True)
    for name, info in variants.items():
        got = file_sha256(str(DATA / f"sft/variant_{name}.jsonl"))
        assert got == info["sha256"], f"variant_{name}.jsonl {got[:12]} != {info['sha256'][:12]}"
    print(f"[prepare_v1] ok: {len(after)} data files, teacher_train and {len(variants)} variants match their hashes")


if __name__ == "__main__":
    main()
