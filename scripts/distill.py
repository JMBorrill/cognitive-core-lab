"""Generate teacher demonstrations (config B data) with rejection sampling.

Pass 1: greedy decoding on the first N tasks of the training pool.
Pass 2: one temperature-0.7 sample for every task rejected in pass 1.
Every episode (accepted or not) is logged with its rejection reasons; the run
is resumable because finished task IDs are skipped on restart.

    python scripts/distill.py --n 1600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccl.data import episode_to_examples, load_tasks, read_jsonl, rejection_reasons, write_jsonl  # noqa: E402
from ccl.models import MLXPolicy  # noqa: E402
from ccl.runtime import run_batch  # noqa: E402

TEACHER = {
    "repo": "mlx-community/Qwen3.5-4B-MLX-4bit",
    "upstream": "Qwen/Qwen3.5-4B",
    "licence": "apache-2.0",
    "local_path": "models/Qwen3.5-4B-MLX-4bit",
}


def run_pass(tasks, policy, log_path: Path, pass_id: int, batch_size: int):
    done = {r["task_id"] for r in read_jsonl(log_path)} if log_path.exists() else set()
    todo = [t for t in tasks if t.task_id not in done]
    print(f"[distill] pass {pass_id}: {len(todo)} to run ({len(done)} already logged)", flush=True)
    f = open(log_path, "a")
    t0, n = time.time(), [0]

    def on_done(ep):
        reasons = rejection_reasons(ep.task, ep)
        ex = [] if reasons else episode_to_examples(ep, "teacher")
        for e in ex:
            e["meta"]["sample"] = pass_id
        f.write(json.dumps({"task_id": ep.task.task_id, "pass": pass_id, "accepted": not reasons,
                            "reasons": reasons, "examples": ex, "trace": ep.trace.to_dict()}) + "\n")
        f.flush()
        n[0] += 1
        if n[0] % 50 == 0:
            print(f"[distill] pass {pass_id}: {n[0]}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)

    run_batch(todo, policy, batch_size=batch_size, on_done=on_done)
    f.close()
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--batch", type=int, default=24)
    a = ap.parse_args()
    data = Path(a.data)
    out = data / "teacher"
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(data / "tasks" / "train_pool.jsonl")[: a.n]

    timings = {}
    greedy = MLXPolicy(TEACHER["local_path"], name="teacher-greedy", temperature=0.0)
    timings["pass1_s"] = run_pass(tasks, greedy, out / "pass1.jsonl", 1, a.batch)
    stats1 = dict(greedy.stats)
    rejected = {r["task_id"] for r in read_jsonl(out / "pass1.jsonl") if not r["accepted"]}
    sampler = greedy
    sampler.temperature, sampler.seed, sampler.name = 0.7, 1, "teacher-t0.7"
    timings["pass2_s"] = run_pass([t for t in tasks if t.task_id in rejected], sampler, out / "pass2.jsonl", 2, a.batch)

    rows, summary = [], {}
    for p in (1, 2):
        recs = read_jsonl(out / f"pass{p}.jsonl")
        acc = [r for r in recs if r["accepted"]]
        reasons = Counter(x for r in recs for x in r["reasons"])
        summary[f"pass{p}"] = {"episodes": len(recs), "accepted": len(acc),
                               "acceptance_rate": round(len(acc) / max(1, len(recs)), 4),
                               "rejection_reasons": dict(reasons)}
        rows += [e for r in acc for e in r["examples"]]
    by_family = Counter(e["meta"]["family"] for e in rows if e["meta"]["turn"] == 1)
    sha = write_jsonl(data / "sft" / "teacher_train.jsonl", rows)
    manifest = {
        "teacher": TEACHER,
        "generation": {"pass1": {"temperature": 0.0}, "pass2": {"temperature": 0.7, "top_p": 0.95, "seed": 1},
                       "max_tokens": 256, "thinking": False, "batch_size": a.batch,
                       "prompt": "identical system prompt and catalogue rendering as the student (ccl.runtime)"},
        "tasks_offered": len(tasks), "summary": summary,
        "accepted_episodes": sum(s["accepted"] for s in summary.values()),
        "accepted_by_family": dict(by_family),
        "sft_rows": len(rows), "sft_sha256": sha, "timings_s": timings,
        "token_stats": stats1 | {"note": "pass-1 counters; pass 2 continues on the same object"},
        "cost": {"api_usd": 0.0, "note": "local inference on Apple M4 16GB; cost is wall-clock only"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
