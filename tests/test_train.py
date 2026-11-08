"""Training-loop tests (need the local base model; skipped otherwise)."""
from pathlib import Path

import numpy as np
import pytest

from ccl.train import batch_schedule

MODEL = "models/Qwen2.5-0.5B-Instruct"
DATA = "data/v1/sft/oracle_val.jsonl"


def test_batch_schedule_is_deterministic_and_covers_epochs():
    a = batch_schedule(10, 4, 6, seed=3)
    b = batch_schedule(10, 4, 6, seed=3)
    assert all((x == y).all() for x, y in zip(a, b))
    flat = np.concatenate(a)
    assert sorted(flat[:10].tolist()) == list(range(10))  # first epoch is a permutation


@pytest.mark.slow
@pytest.mark.skipif(not Path(MODEL).exists() or not Path(DATA).exists(), reason="model/data not present")
def test_resume_matches_uninterrupted_run(tmp_path):
    import mlx.core as mx

    from ccl.train import train
    cfg = {"model": MODEL, "train_files": [DATA], "steps": 4, "batch_size": 2, "grad_accum": 1,
           "save_every": 100, "eval_every": 100, "seed": 5, "warmup": 1}
    train({**cfg, "out_dir": str(tmp_path / "full")}, fresh=True)
    r = train({**cfg, "out_dir": str(tmp_path / "resumed"), "stop_after": 2}, fresh=True)
    assert r == {"interrupted_at": 2}
    s = train({**cfg, "out_dir": str(tmp_path / "resumed")})
    assert s["resumed_from"] == 2
    a = mx.load(str(tmp_path / "full" / "adapters.safetensors"))
    b = mx.load(str(tmp_path / "resumed" / "adapters.safetensors"))
    for k in a:
        assert np.allclose(np.array(a[k]), np.array(b[k]), atol=1e-4), k
