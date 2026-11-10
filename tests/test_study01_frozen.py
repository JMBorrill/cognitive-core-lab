"""Study 01 must stay reproducible while Study 02 code is added.

Rebuilds the v1 task pools and oracle data from code and compares every SHA-256 with the
committed manifest. Pure Python, about five seconds; runs on any platform.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v1_datasets_rebuild_byte_for_byte(tmp_path):
    out = tmp_path / "v1"
    subprocess.run([sys.executable, str(ROOT / "scripts/build_datasets.py"), "--out", str(out)],
                   cwd=ROOT, check=True, capture_output=True)
    committed = json.loads((ROOT / "data/v1/manifest.json").read_text())["files"]
    rebuilt = json.loads((out / "manifest.json").read_text())["files"]
    assert {k: v["sha256"] for k, v in rebuilt.items()} == {k: v["sha256"] for k, v in committed.items()}
