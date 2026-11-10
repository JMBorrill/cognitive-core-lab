"""Exit (and print) as soon as any run in runs/ diverges, the queue logs a failure, or the pipeline finishes.

Divergence uses the pre-registered pilot rule: a training loss after step 10 above 2x the mean of steps 1-10.
"""
import json
import pathlib
import sys
import time

root = pathlib.Path(__file__).resolve().parents[1]
start_line = int(sys.argv[1]) if len(sys.argv) > 1 else 0
while True:
    for log in sorted(root.glob("runs/*/train_log.jsonl")):
        L = [json.loads(x) for x in open(log) if x.strip()]
        if len(L) > 10:
            base = sum(r["loss"] for r in L[:10]) / 10
            bad = [r for r in L[10:] if r["loss"] > 2 * base]
            if bad:
                print(f"DIVERGED {log.parent.name}: base {base:.3f}, step {bad[0]['step']} loss {bad[0]['loss']:.3f}")
                sys.exit(0)
    q = root / "logs/queue.jsonl"
    if q.exists() and any(json.loads(x)["rc"] != 0 for x in open(q)):
        print("QUEUE STEP FAILED"); sys.exit(0)
    tail = (root / "logs/pipeline.log").read_text().splitlines()[start_line:]
    if any("Traceback" in t for t in tail) or any("[pipeline] complete" in t for t in tail):
        print("PIPELINE ENDED:", *tail[-3:], sep="\n"); sys.exit(0)
    time.sleep(120)
