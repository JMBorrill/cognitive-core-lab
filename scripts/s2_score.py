"""Study 02: score final adapters (or the base model) on held-out splits. The only code that reads
`data/v2/sealed`. Resumable; every first access to a split by a run is appended to its access log.

    python scripts/s2_score.py configs/study02/e1/<run>.json [...] --out runs2 [--bucket /out]

Split names: v1/<file> (Study 01 task files, the external anchor), v2/dev, v2/test_* (sealed) and
`own` (new instances of the run's own training families).
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
from ccl.data import load_tasks  # noqa: E402


def tasks_for(split: str, cfg: dict):
    if split.startswith("v1/"):
        return load_tasks(ROOT / "data/v1/tasks" / f"{split[3:]}.jsonl")
    if split == "own":
        fams, tools = s2.load_practice()
        return s2.e1_own_tasks(fams, tools, cfg["data"]["replicate"], cfg["data"]["D"])
    return s2.load_split(split[3:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--out", default="runs2")
    ap.add_argument("--bucket", default=None)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N tasks of each split (smoke tests)")
    ap.add_argument("--batch", type=int, default=0, help="episodes scored in lock-step (default: the config's value)")
    ap.add_argument("--splits", nargs="*", default=None, help="score only these splits of each config (the run is "
                    "then not marked complete)")
    a = ap.parse_args()
    from ccl import hf
    out_root = Path(a.out)
    bucket = Path(a.bucket) if a.bucket else None
    loaded = (None, None, None)  # (model path, adapter dir) -> model, reused across configs that share it
    for path in a.configs:
        cfg = json.loads(Path(path).read_text())
        name = cfg["name"]
        out = out_root / cfg["experiment"] / name
        remote = bucket / cfg["experiment"] / name if bucket else None
        s2.restore(remote, out)
        if (out / "DONE").exists():
            print(f"[s2_score] {name}: already scored", flush=True)
            continue
        src = cfg.get("adapter_from")  # score another run's adapter again (repeatability check)
        adapter = None
        if cfg.get("train") or src:
            adapter = out / "train" if not src else out_root / cfg["experiment"] / src / "train"
            if src:
                s2.restore(bucket / cfg["experiment"] / src if bucket else None, adapter.parent)
            assert (adapter / "DONE").exists(), f"{name}: no final adapter at {adapter}"
        key = (cfg["model"]["repo"], str(adapter))
        if loaded[0] != key:
            model, tok = hf.load_policy_model(s2.fetch_model(cfg["model"]), adapter)
            loaded = (key, model, tok)
        policy = hf.HFPolicy(loaded[1], loaded[2], name)
        sha = json.loads((adapter / "summary.json").read_text())["adapter_sha256"] if adapter else None
        timings = []
        for split in [x for x in cfg["score"] if a.splits is None or x in a.splits]:
            tasks = tasks_for(split, cfg)
            if a.limit:
                tasks = tasks[: a.limit]
            todo = len(tasks) - len({k for k in s2.done_keys(out / "metrics.jsonl") if k[0] == split})
            if todo and split != "own" and split != "v2/dev":
                out.mkdir(parents=True, exist_ok=True)
                with open(out / "test_access.jsonl", "a") as f:
                    f.write(json.dumps({"time": time.strftime("%FT%TZ", time.gmtime()), "run": name, "split": split,
                                        "tasks": len(tasks), "to_score": todo, "adapter_sha256": sha,
                                        "config_sha256": s2.config_sha256(cfg),
                                        "commit": s2.environment()["commit"]}) + "\n")
            timings.append(s2.evaluate(policy, tasks, split, out, name, batch=a.batch or cfg.get("eval_batch", 64),
                                       fewshot=cfg.get("fewshot", False), checkpoint=lambda: s2.sync(out, remote)))
            print(f"[s2_score] {name} {timings[-1]}", flush=True)
            s2.sync(out, remote)
        (out / "run.json").write_text(json.dumps(
            {"name": name, "experiment": cfg["experiment"], "config_sha256": s2.config_sha256(cfg),
             "adapter_sha256": sha, "fewshot": cfg.get("fewshot", False), "eval_batch": a.batch or cfg.get("eval_batch", 64),
             "scoring": timings, "generation": policy.stats, **s2.environment()}, indent=1))
        if not a.limit and a.splits is None:
            (out / "DONE").write_text("ok\n")
        s2.sync(out, remote)


if __name__ == "__main__":
    main()
