"""Build the versioned task pools, the oracle (independent) SFT data and the leakage audit.

    python scripts/build_datasets.py            # writes data/v1/...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccl.data import episode_to_examples, rejection_reasons, save_tasks, write_jsonl  # noqa: E402
from ccl.oracle import OraclePolicy  # noqa: E402
from ccl.runtime import run_batch  # noqa: E402
from ccl.tasks import TEMPLATES, audit_splits, generate_split, graph_signature, make_task, templates_for  # noqa: E402

SIZES = {"eval_id": 200, "eval_ood": 240, "eval_unseen": 120, "eval_renamed": 160}


def filtered_split(name: str, n: int, banned_requests: set[str], seed: int = 0) -> tuple[list, int]:
    """Generate `n` tasks whose request text never appears in training/validation pools."""
    out, dropped, k = [], 0, 0
    while len(out) < n:
        batch = generate_split(name, n * 2, seed + k)
        for t in batch:
            if len(out) == n:
                break
            if t.request in banned_requests:
                dropped += 1
            else:
                out.append(t)
        k += 1
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v1")
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=150)
    a = ap.parse_args()
    out = Path(a.out)
    manifest = {"version": out.name, "files": {}, "templates": {t: {k: v for k, v in m.items() if k != "fn"}
                                                                for t, m in TEMPLATES.items()}}

    train = generate_split("train", a.train, seed=0)
    val = generate_split("train", a.val, seed=1)
    banned = {t.request for t in train + val}
    evals, dropped = {}, {}
    for name, n in SIZES.items():
        evals[name], dropped[name] = filtered_split(name, n, banned)
    # Catalogue-size ablation: identical in-distribution tasks with a fixed distractor count.
    cat_tpls = templates_for("train")
    for k in (0, 5, 12):
        ts = [make_task(cat_tpls[i % len(cat_tpls)], 50_000_000 + i, "eval", n_distractors=k) for i in range(400)]
        evals[f"eval_catalog_{k}"] = [t for t in ts if t.request not in banned][:96]

    tr_sigs = {graph_signature(t) for t in train}
    for name, ts in evals.items():
        for t in ts:
            t.tags = t.tags + (["graph_seen"] if graph_signature(t) in tr_sigs else ["graph_novel"])

    for name, ts in [("train_pool", train), ("val_pool", val), *evals.items()]:
        manifest["files"][f"tasks/{name}.jsonl"] = {"sha256": save_tasks(out / "tasks" / f"{name}.jsonl", ts),
                                                    "n": len(ts),
                                                    "families": dict(Counter(t.family for t in ts))}

    audit = audit_splits(train + val, evals)
    for name in SIZES:
        audit[name]["dropped_for_request_overlap"] = dropped[name]
    (out / "leakage_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=1))

    # Independent supervision: oracle trajectories, executed and validated like any other policy.
    for name, ts in [("oracle_train", train), ("oracle_val", val)]:
        eps = run_batch(ts, OraclePolicy(), batch_size=256)
        rej = Counter()
        rows = []
        for e in eps:
            r = rejection_reasons(e.task, e)
            if r:
                rej.update(r)
                continue
            rows.extend(episode_to_examples(e, "oracle"))
        manifest["files"][f"sft/{name}.jsonl"] = {
            "sha256": write_jsonl(out / "sft" / f"{name}.jsonl", rows), "rows": len(rows),
            "episodes": len(eps) - sum(1 for e in eps if rejection_reasons(e.task, e)),
            "rejected": dict(rej)}
        print(name, manifest["files"][f"sft/{name}.jsonl"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
