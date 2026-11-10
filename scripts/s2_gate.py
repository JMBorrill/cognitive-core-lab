"""Quality gate, step 3: a strong open-weight reference model attempts the test families before any
student is evaluated on them.

    python scripts/s2_gate.py --out runs2 [--bucket /out]        # inside a GPU job
    python scripts/s2_gate.py --report results/study02/runs/gate  # afterwards, on CPU

The reference model attempts the first task of every test family and the second task of every family
whose first attempt failed. Families it fails twice are listed for reading; a family is removed only if
that reading finds a defect in the task (see experiments/study02_e1_diversity/PLAN.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccl import s2  # noqa: E402

REFERENCE = {"repo": "Qwen/Qwen2.5-7B-Instruct", "revision": "a09a35458c702b33eeacc393d103063234e8bc28"}
SPLITS = ["test_comp", "test_depth", "test_tool", "test_domain", "test_id"]


def attempts(split: str, max_hops: int):
    """(first tasks, second tasks) of a split, one of each per family of at most `max_hops` hops."""
    by = defaultdict(list)
    for t in s2.load_split(split):
        if t.depth <= max_hops:
            by[t.template].append(t)
    return [ts[0] for ts in by.values()], {f: ts[1] for f, ts in by.items()}


def run(a) -> None:
    from ccl import hf
    out = Path(a.out) / a.name
    remote = Path(a.bucket) / a.name if a.bucket else None
    s2.restore(remote, out)
    model, tok = hf.load_policy_model(s2.fetch_model(REFERENCE), None, dtype="bfloat16")
    policy = hf.HFPolicy(model, tok, "reference")
    for split in SPLITS:
        first, second = attempts(split, a.max_hops)
        if a.limit:
            first = first[: a.limit]
        save = lambda: s2.sync(out, remote)  # noqa: E731
        print(s2.evaluate(policy, first, f"{split}#1", out, "reference", batch=a.batch, fewshot=True, checkpoint=save),
              flush=True)
        s2.sync(out, remote)
        rows = [json.loads(l) for l in open(out / "metrics.jsonl")]
        failed = {r["template"] for r in rows if r["split"] == f"{split}#1" and not r["success"]}
        retry = [second[f] for f in sorted(failed)]
        print(s2.evaluate(policy, retry, f"{split}#2", out, "reference", batch=a.batch, fewshot=True, checkpoint=save),
              flush=True)
        s2.sync(out, remote)
    (out / "run.json").write_text(json.dumps({"reference": REFERENCE, "generation": policy.stats,
                                              **s2.environment()}, indent=1))
    (out / "DONE").write_text("ok\n")
    s2.sync(out, remote)


def report(path: str) -> None:
    rows = [json.loads(l) for l in open(Path(path) / "metrics.jsonl")]
    fams = {f["id"]: f for f in s2.read_gz(s2.V2 / "sealed/families.jsonl.gz")}
    by = defaultdict(dict)
    for r in rows:
        split, k = r["split"].split("#")
        by[(split, r["template"])][k] = r
    out = {"reference": "Qwen/Qwen2.5-7B-Instruct (few-shot prompt, greedy)", "splits": {}, "failed_twice": []}
    for split in SPLITS:
        got = {f: v for (s, f), v in by.items() if s == split}
        first = sum(bool(v["1"]["success"]) for v in got.values())
        got = {f: v for f, v in got.items() if "1" in v}
        twice = sorted(f for f, v in got.items() if not v["1"]["success"] and "2" in v and not v["2"]["success"])
        out["splits"][split] = {"families": len(got), "first_attempt_success": first,
                                "second_attempt_success": sum(bool(v["2"]["success"]) for v in got.values() if "2" in v),
                                "failed_twice": len(twice)}
        for f in twice:
            checks = Counter(k[6:] for a in ("1", "2") for k, x in got[f][a].items() if k.startswith("check_") and not x)
            out["failed_twice"].append({"split": split, "family": f, "hops": fams[f]["hops"], "mod": fams[f]["mod"],
                                        "shape": fams[f]["shape"], "failed_checks": dict(checks),
                                        "parse_errors": sum(got[f][a]["parse_errors"] for a in ("1", "2"))})
    (Path(path) / "gate_report.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"splits": out["splits"]}, indent=1))
    for key in ("hops", "mod", "shape"):
        print(key, dict(Counter(x[key] for x in out["failed_twice"]).most_common()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs2")
    ap.add_argument("--bucket", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-hops", type=int, default=6, help="attempt only families of at most this many hops")
    ap.add_argument("--name", default="gate", help="output directory name")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    report(a.report) if a.report else run(a)
