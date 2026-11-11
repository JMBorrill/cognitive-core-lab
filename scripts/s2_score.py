"""Study 02: score final adapters (or the base model) on held-out splits. The only code that reads
`data/v2/sealed`, including for the complete-world twins of the test tasks (`--twins`, splits `v2/<split>_adv`). Resumable; every first access to a split by a run is appended to its access log.

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
    if split.endswith("_adv"):  # complete-world twins (Experiment 2, amendments 1 and 2), rebuilt and checked
        from ccl import adversarial, loop
        tasks = adversarial.twins(s2.load_split(split[3:-4]))
        want = json.loads(TWINS.read_text())["splits"][split]
        assert len(tasks) == want["twins"] and loop.tasks_sha256(tasks) == want["sha256"], f"{split}: not the pre-registered set"
        return tasks
    return s2.load_split(split[3:])


TWINS = ROOT / "configs/study02/e2/twins.json"
TWIN_SPLITS = ("dev", "test_comp", "test_depth", "test_tool", "test_domain", "test_id")


def twin_manifest() -> dict:
    """Build the twins of every evaluation split and record what identifies them: counts and SHA-256.
    Checked here, by program, without a model and without printing a task: the oracle solves every twin
    with the transcript it has on the original, and request, gold graph, answer, catalogue and faults are
    the original's. Nothing about any model's outcome on a test split is read or written."""
    from collections import Counter
    from ccl import adversarial, loop, verify
    from ccl.evaluate import validate
    from ccl.oracle import OraclePolicy
    from ccl.runtime import run_batch
    out = {"built_by": "scripts/s2_score.py --twins (ccl.adversarial.twins)", "rule": "every task in which a gold call "
           "depends on a list selection and whose expected status is done or blocked", "version": adversarial.VERSION, "splits": {}}
    for name in TWIN_SPLITS:
        tasks = s2.load_split(name)
        by_id = {t.task_id: t for t in tasks}
        tw = adversarial.twins(tasks)
        orig = [by_id[next(x for x in t.tags if x.startswith("twin_of="))[8:]] for t in tw]
        on_twin = run_batch(tw, OraclePolicy(), batch_size=512)
        on_orig = run_batch(orig, OraclePolicy(), batch_size=512)
        same = sum(validate(t, e)["success"] and verify.parts_ok(t, e.trace.final) and not verify.shortcut(e.trace.to_dict())
                   and e.trace.messages[1:] == f.trace.messages[1:]
                   and (t.request, t.gold, t.answer_facts, t.catalog, t.faults) == (o.request, o.gold, o.answer_facts, o.catalog, o.faults)
                   for t, o, e, f in zip(tw, orig, on_twin, on_orig))
        assert same == len(tw) and len({t.task_id for t in tw}) == len(tw), name
        assert not any(adversarial.leaks(t, t.world) for t in tw), name  # no walk from another candidate misses a record
        out["splits"][f"v2/{name}_adv"] = {
            "tasks_in_split": len(tasks), "twins": len(tw), "sha256": loop.tasks_sha256(tw),
            "hops": {str(h): n for h, n in sorted(Counter(t.depth for t in tw).items())},
            "oracle_transcript_identical": same, "walks_from_other_candidates_that_miss_a_record": 0,
            "identical_to_version_1": sum(a.task_id == b.task_id for a, b in zip(tw, adversarial.twins(tasks, version=1)))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--out", default="runs2")
    ap.add_argument("--bucket", default=None)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N tasks of each split (smoke tests)")
    ap.add_argument("--batch", type=int, default=0, help="episodes scored in lock-step (default: the config's value)")
    ap.add_argument("--splits", nargs="*", default=None, help="score only these splits of each config (the run is "
                    "then not marked complete)")
    ap.add_argument("--twins", action="store_true", help="no model: write configs/study02/e2/twins.json and stop")
    a = ap.parse_args()
    if a.twins:
        TWINS.write_text(json.dumps(twin_manifest(), indent=1) + "\n")
        print(json.dumps({k: {x: v[x] for x in ("tasks_in_split", "twins", "hops")} for k, v in
                          json.loads(TWINS.read_text())["splits"].items()}, indent=1))
        return
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
