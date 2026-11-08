"""Evaluate one model configuration on the held-out splits.

    python scripts/evaluate.py --name A_base --adapter none
    python scripts/evaluate.py --name B_s0 --adapter runs/B_s0
    python scripts/evaluate.py --name oracle --policy oracle

Writes results/runs/<name>/{metrics.jsonl, traces.jsonl.gz, run.json}. Resumable.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccl.data import load_tasks  # noqa: E402
from ccl.evaluate import episode_metrics  # noqa: E402
from ccl.runtime import run_batch  # noqa: E402

MAIN_SPLITS = ["eval_id", "eval_ood", "eval_unseen", "eval_renamed"]
ABLATION_SPLITS = ["eval_catalog_0", "eval_catalog_5", "eval_catalog_12"]


def salvage(out: Path) -> set:
    """After an interruption, keep only episodes that have both a metrics row and an intact trace."""
    traces = []
    tp = out / "traces.jsonl.gz"
    if tp.exists():
        try:
            with gzip.open(tp, "rt") as f:
                for line in f:
                    try:
                        traces.append(json.loads(line))
                    except json.JSONDecodeError:
                        break
        except (EOFError, OSError):
            pass  # truncated final gzip member: keep what was read
    have = {(t["split"], t["task_id"]) for t in traces}
    rows = []
    for line in open(out / "metrics.jsonl"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (r["split"], r["task_id"]) in have:
            rows.append(r)
    keep = {(r["split"], r["task_id"]) for r in rows}
    with open(out / "metrics.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with gzip.open(tp, "wt") as f:
        f.writelines(json.dumps(t) + "\n" for t in traces if (t["split"], t["task_id"]) in keep)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--policy", default="mlx", choices=["mlx", "oracle"])
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--splits", nargs="*", default=MAIN_SPLITS)
    ap.add_argument("--fewshot", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    out = Path("results/runs") / a.name
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "metrics.jsonl"
    done = salvage(out) if mpath.exists() else set()

    import mlx.core as mx
    if a.policy == "oracle":
        from ccl.oracle import OraclePolicy
        policy = OraclePolicy()
    else:
        from ccl.models import MLXPolicy
        adapter = None if a.adapter == "none" else a.adapter
        policy = MLXPolicy(a.model, adapter_path=adapter, name=a.name, temperature=a.temperature, seed=a.seed)
    mx.reset_peak_memory()

    mf = open(mpath, "a")
    tf = gzip.open(out / "traces.jsonl.gz", "at")
    t_all = time.time()
    split_times = {}
    for split in a.splits:
        tasks = load_tasks(Path(a.data) / "tasks" / f"{split}.jsonl")
        if a.limit:
            tasks = tasks[: a.limit]
        todo = [t for t in tasks if (split, t.task_id) not in done]
        if not todo:
            continue
        t0 = time.time()

        def on_done(ep, split=split):
            m = episode_metrics(ep.task, ep)
            m.update({"split": split, "config": a.name, "graph_seen": "graph_seen" in ep.task.tags})
            mf.write(json.dumps(m) + "\n")
            mf.flush()
            tf.write(json.dumps({"split": split, "task_id": ep.task.task_id, "trace": ep.trace.to_dict()}) + "\n")

        run_batch(todo, policy, batch_size=a.batch, on_done=on_done, fewshot=a.fewshot)
        split_times[split] = time.time() - t0
        print(f"[eval] {a.name} {split}: {len(todo)} tasks in {split_times[split]:.0f}s", flush=True)
    mf.close()
    tf.close()

    run = {"name": a.name, "model": a.model, "adapter": a.adapter, "policy": a.policy, "fewshot": a.fewshot,
           "temperature": a.temperature, "seed": a.seed, "batch": a.batch, "splits": a.splits,
           "wall_time_s": time.time() - t_all, "split_times_s": split_times,
           "peak_mem_gb": mx.get_peak_memory() / 1e9, "machine": platform.machine(),
           "mlx_stats": getattr(policy, "stats", None)}
    prev = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else {}
    if prev.get("splits") and set(prev["splits"]) - set(a.splits):
        run["splits"] = sorted(set(prev["splits"]) | set(a.splits))
        run["previous"] = prev
    (out / "run.json").write_text(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
