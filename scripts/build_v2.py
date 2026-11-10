"""Build the Study 02 environment: practice space, `dev`, sealed test splits, census and leakage audit.

    python scripts/build_v2.py                  # writes data/v2/...
    python scripts/build_v2.py --out /tmp/v2    # rebuild elsewhere and compare manifest hashes

Layout (training, practice and curriculum code read `practice/` and `dev/` only):

    data/v2/practice/families.jsonl   every practice family with its plans
    data/v2/practice/tools.json       the tools a practice catalogue may show
    data/v2/dev/                      300 families x 2 tasks on dev's own held-out edges, tools and domain
    data/v2/sealed/                   test splits, their families and the full hold-out definition
    data/v2/gate/decisions.json       families removed by the quality gate, with reasons (may be empty)
    data/v2/manifest.json             SHA-256 of every file, census, quality-gate results

Task files are gzip-compressed JSONL; the manifest hashes the uncompressed bytes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccl import compose as C  # noqa: E402
from ccl.evaluate import validate  # noqa: E402
from ccl.oracle import OraclePolicy  # noqa: E402
from ccl.runtime import run_batch  # noqa: E402
from ccl.world2 import DOMAINS, ENTITIES, TOOLS2  # noqa: E402

MIN_PRACTICE = 5000
TASKS_PER_FAMILY = 2
MAX_PLANS = 24  # request-text variety kept per family (families of one call can have hundreds of plans)
# split -> (source families, number of families, hop strata)
EVAL = {
    "dev": [("dev_comp", 150, (2, 3, 4)), ("dev_tool", 90, (1, 2, 3, 4)), ("dev_domain", 60, (1, 2, 3, 4))],
    "test_comp": [("test_comp", 600, (2, 3, 4))],
    "test_depth": [("test_depth", 200, (5, 6))],
    "test_tool": [("test_tool", 200, (1, 2, 3, 4))],
    "test_domain": [("test_domain", 200, (1, 2, 3, 4))],
    "test_id": [("practice", 200, (1, 2, 3, 4))],
}


class NoCalls:
    """Degenerate policy: answer at once."""
    name = "no-calls"

    def generate(self, batch, episodes):
        return [(json.dumps({"final": "Done.", "status": "done"}), 0) for _ in episodes]


class CallEverything:
    """Degenerate policy: call every listed tool once, then answer."""
    name = "call-everything"

    def generate(self, batch, episodes):
        return [(json.dumps({"calls": [{"tool": t.name, "args": {}} for t in e.task.catalog][:6]} if e.turn == 0
                            else {"final": "Done.", "status": "done"}), 0) for e in episodes]


def dump(path: Path, rows: list[dict]) -> dict:
    """Write JSON lines (gzip if the name ends in .gz) and return the hash of the uncompressed bytes."""
    raw = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with open(path, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as g:
            g.write(raw)
    else:
        path.write_bytes(raw)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "n": len(rows)}


def slim(f: dict) -> dict:
    plans = sorted(f["plans"], key=lambda p: C._h(C.SALT, "plan", f["id"], json.dumps(p, sort_keys=True)))
    return {**f, "plans": plans[:MAX_PLANS], "n_plans": len(f["plans"])}


def distractors(source: str, part: dict) -> list[str]:
    pool = set(part["practice_tools"])
    kind, _, what = source.partition("_")
    if what == "tool":
        pool |= set(part[f"{kind}_tools"])
    elif what == "domain":
        pool |= {k for k in TOOLS2 if C.tool_domains(k) <= set(part[f"{kind}_domains"])}
    return sorted(pool)


def census(fams: list[dict]) -> dict:
    return {"families": len(fams), "base_graphs": len({g for f in fams for g in f["graphs"]}),
            "hops": dict(sorted(Counter(f["hops"] for f in fams).items())),
            "modifier": dict(sorted(Counter(f["mod"] for f in fams).items())),
            "shape": dict(sorted(Counter(f["shape"] for f in fams).items())),
            "tools": len({t for f in fams for t in f["tools"]}), "edges": len({e for f in fams for e in f["edges"]})}


def gate(tasks) -> dict:
    """Quality gate, steps 1 and 2: the oracle solves every task and degenerate policies fail."""
    out = {}
    for pol in (OraclePolicy(), NoCalls(), CallEverything()):
        eps = run_batch(tasks, pol, batch_size=512)
        ok = [validate(e.task, e)["success"] for e in eps]
        out[pol.name] = {"success": sum(ok), "n": len(ok)}
        if pol.name == "oracle":
            assert all(ok), "the oracle failed a generated task"
            assert not any(t["kind"] == "invalid" for e in eps for t in e.trace.turns)
            assert not any(r.outcome in ("hallucinated", "invalid_argument") for e in eps for r in e.trace.calls)
        else:  # must fail wherever the task needs a call (no-calls) or has any gold call (call-everything)
            need = [any(not g.optional for g in e.task.gold) if pol.name == "no-calls" else bool(e.task.gold)
                    for e in eps]
            assert not any(o and n for o, n in zip(ok, need)), f"{pol.name} passed a task that needs tools"
            out[pol.name]["on_tasks_needing_tools"] = {"success": 0, "n": sum(need)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/v2"))
    a = ap.parse_args()
    out = Path(a.out)
    space = C.build_space()
    part, fams = space["partition"], space["families"]
    decisions_path = ROOT / "data/v2/gate/decisions.json"
    decisions = json.loads(decisions_path.read_text()) if decisions_path.exists() else {"dropped": {}}
    dropped = set(decisions["dropped"])

    assert len(fams["practice"]) >= MIN_PRACTICE, f"practice space has {len(fams['practice'])} families"
    ids = [f["id"] for fs in fams.values() for f in fs]
    assert len(ids) == len(set(ids)), "a family is in two splits"
    held_tools = set(part["test_tools"]) | set(part["dev_tools"])
    held_edges = set(part["test_edges"]) | set(part["dev_edges"])
    held_dom = set(part["test_domains"]) | set(part["dev_domains"])
    for f in fams["practice"]:
        assert not set(f["tools"]) & held_tools and not set(f["edges"]) & held_edges
        assert not {x for t in f["tools"] for x in C.tool_domains(t)} & held_dom and f["hops"] <= C.PRACTICE_HOPS

    manifest = {"version": "v2", "salt": C.SALT, "files": {}, "census": {}, "quality_gate": {},
                "schema": {"domains": len(DOMAINS), "entities": len(ENTITIES), "tools": len(TOOLS2),
                           "tools_by_kind": dict(sorted(Counter(t.kind for t in TOOLS2.values()).items()))},
                "partition": {k: len(v) for k, v in part.items()},
                "practice_tools_used_in_practice_graphs": len(
                    {t for f in fams["practice"] for t in f["tools"]} |
                    {C.BACKUP_OF[t] for f in fams["practice"] for t in f["tools"] if t in C.BACKUP_OF}),
                "gate_decisions": {"dropped": len(dropped)}}
    files = manifest["files"]
    files["practice/families.jsonl.gz"] = dump(out / "practice/families.jsonl.gz", [slim(f) for f in fams["practice"]])
    (out / "practice").mkdir(parents=True, exist_ok=True)
    (out / "practice/tools.json").write_text(json.dumps(part["practice_tools"], indent=1) + "\n")
    files["practice/tools.json"] = {"sha256": hashlib.sha256((out / "practice/tools.json").read_bytes()).hexdigest(),
                                    "n": len(part["practice_tools"])}
    manifest["census"]["practice"] = census(fams["practice"])
    for s in C.SPLITS[1:]:
        manifest["census"][f"available:{s}"] = census(fams[s])

    sealed_fams, all_eval = [], {}
    for split, parts in EVAL.items():
        tasks, chosen = [], []
        for source, n, hops in parts:
            order = [f for f in C.balanced_order(fams[source], split, hops, horizon=n) if f["id"] not in dropped]
            assert len(order) >= n, f"{split}: only {len(order)} families in {source}"
            pool = distractors(source, part)
            for f in order[:n]:
                chosen.append({**slim(f), "eval_split": split})
                tasks += [C.make_task(f, i, "eval", split, pool) for i in range(TASKS_PER_FAMILY)]
        where = "dev" if split == "dev" else "sealed"
        files[f"{where}/{split}.jsonl.gz"] = dump(out / where / f"{split}.jsonl.gz", [t.to_dict() for t in tasks])
        manifest["census"][split] = {**census(chosen), "tasks": len(tasks),
                                     "reserved_phrasing": sum("phr=reserved" in t.tags for t in tasks)}
        manifest["quality_gate"][split] = gate(tasks)
        all_eval[split] = tasks
        if split == "dev":
            files["dev/families.jsonl.gz"] = dump(out / "dev/families.jsonl.gz", chosen)
        else:
            sealed_fams += chosen
    files["sealed/families.jsonl.gz"] = dump(out / "sealed/families.jsonl.gz", sealed_fams)
    (out / "sealed/partition.json").write_text(json.dumps(part, indent=1) + "\n")
    files["sealed/partition.json"] = {"sha256": hashlib.sha256((out / "sealed/partition.json").read_bytes()).hexdigest()}

    # ---- leakage audit: evaluation instances against training instances of every practice family
    train = [C.make_task(f, i, "train", "practice", part["practice_tools"]) for f in fams["practice"] for i in (0, 1)]
    manifest["quality_gate"]["practice_sample"] = gate(train)
    tr_req = {t.request for t in train}
    tr_ids = {k for t in train for tab in t.world.values() if isinstance(tab, dict) for k in tab}
    tr_fam = {f["id"] for f in fams["practice"]}
    audit = {}
    for split, tasks in all_eval.items():
        ev_ids = {k for t in tasks for tab in t.world.values() if isinstance(tab, dict) for k in tab}
        fam_overlap = len({t.template for t in tasks} & tr_fam)
        audit[split] = {"n": len(tasks), "request_overlap": sum(t.request in tr_req for t in tasks),
                        "record_id_overlap": len(ev_ids & tr_ids), "family_overlap_with_practice": fam_overlap,
                        "held_out_tools_in_catalogues": len({s.key for t in tasks for s in t.catalog} & held_tools)}
        assert audit[split]["record_id_overlap"] == 0 and audit[split]["request_overlap"] == 0
        assert fam_overlap == (len({t.template for t in tasks}) if split == "test_id" else 0)
    assert not {s.key for t in train for s in t.catalog} - set(part["practice_tools"])
    manifest["leakage_audit"] = {"training_sample": {"tasks": len(train), "instances_per_family": 2}, "splits": audit}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("schema", "partition", "census", "quality_gate", "leakage_audit")},
                     indent=1))


if __name__ == "__main__":
    main()
