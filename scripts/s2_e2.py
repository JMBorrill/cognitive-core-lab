"""Study 02, Experiment 2: what a Stage 1 GPU job runs, and the CPU replay of what it produced.

    python scripts/s2_e2.py seed    --replicate 0 --out runs2 [--bucket /out]   # seed set and seed adapter
    python scripts/s2_e2.py bench   --replicate 0 --out runs2 [--bucket /out]   # training cost on a cycle-1 buffer
    python scripts/s2_e2.py calib   --out runs2 [--bucket /out]                 # engine against Experiment 1's rows
    python scripts/s2_e2.py dev     --replicate 0 --out runs2 [--bucket /out]   # seed adapter on dev
    python scripts/s2_e2.py probe   --replicate 0 --tasks 400 --out runs2 [--bucket /out]
    python scripts/s2_e2.py teacher --repo R --revision V --out runs2 [--bucket /out]
    python scripts/s2_e2.py replay  DIR                                         # CPU, any machine

Inputs are the practice space and `dev`, nothing else (tests/test_e2.py). The probe is the beginning of
cycle 1 of the `practice` arm of a replicate: it writes that arm's attempt files, so a later pilot
continues from them instead of sampling the same tasks again.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccl import loop as L  # noqa: E402
from ccl import s2  # noqa: E402
from ccl.data import write_jsonl  # noqa: E402

S15 = {"repo": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}
CALIB = ("e1_s15_D1600_r0", "e1_s15_D25_r0")  # own-family tasks of these runs are scored again (plan §4)
TEACHER_TASKS_PER_HOP, TEACHER_RETRY_T = 50, 0.7  # one greedy attempt, one sampled retry as in Study 01
QUALIFY = {"overall": 0.60, "four_hop": 0.40}  # plan §7, Stage 1(d)


def train_settings() -> dict:
    """Experiment 1's settings for this student (plan §3.1): nothing is tuned in Experiment 2."""
    decision = json.loads((ROOT / "results/study02/e0/decision.json").read_text())
    return {**decision["train"], "save_every": 100, "eval_every": 100, "max_seq_len": 4096, "grad_checkpoint": True}


def dirs(a, name: str) -> tuple[Path, Path | None]:
    return Path(a.out) / "e2" / name, (Path(a.bucket) / "e2" / name if a.bucket else None)


def finish(out: Path, remote: Path | None, result: dict) -> None:
    (out / "result.json").write_text(json.dumps({**result, **s2.environment()}, indent=1))
    s2.sync(out, remote)
    print(f"[s2_e2] {out.name}: " + json.dumps({k: v for k, v in result.items() if not isinstance(v, (dict, list))}),
          flush=True)


def adapter_dir(a, experiment: str, name: str) -> Path:
    """A trained adapter, taken from the bucket if it is not here (four small files, no checkpoints)."""
    local = Path(a.out) / experiment / name / "train"
    if not (local / "DONE").exists() and a.bucket:
        local.mkdir(parents=True, exist_ok=True)
        for f in ("adapters.safetensors", "adapter_config.json", "summary.json", "DONE"):
            shutil.copyfile(Path(a.bucket) / experiment / name / "train" / f, local / f)
    assert (local / "DONE").exists(), f"no final adapter at {local}"
    return local


def seed_rows(replicate: int) -> tuple[list[dict], str]:
    fams, tools = s2.load_practice()
    rows, stats = s2.oracle_rows(L.seed_tasks(fams, tools, replicate))
    assert stats["rejected"] == 0 and stats["episodes"] == L.SEED_FAMILIES * L.SEED_INSTANCES
    return rows, s2.rows_sha256(rows)


def val_file(data: Path) -> str:
    """Validation loss only, as in Experiment 1: oracle trajectories on the first 80 dev tasks."""
    val = data / "dev_val.jsonl"
    if not val.exists():
        write_jsonl(val, s2.oracle_rows(s2.load_split("dev")[:80])[0])
    return str(val)


# ------------------------------------------------------------------ seed adapter

def cmd_seed(a) -> None:
    from ccl import hf
    name = f"r{a.replicate}_seed"
    out, remote = dirs(a, name)
    s2.restore(remote, out)
    if (out / "train/DONE").exists():
        return print(f"[s2_e2] {name}: already trained", flush=True)
    t0 = time.time()
    rows, sha = seed_rows(a.replicate)
    data = Path(a.out) / "_data"
    assert write_jsonl(data / f"e2_seed_r{a.replicate}.jsonl", rows) == sha
    cfg = {**train_settings(), "steps": len({r["meta"]["task_id"] for r in rows}) // 4,  # two passes of 8 episodes
           "model": s2.fetch_model(S15), "train_files": [str(data / f"e2_seed_r{a.replicate}.jsonl")],
           "val_file": val_file(data), "out_dir": str(out / "train"), "seed": a.replicate}
    summary = hf.train(cfg, on_checkpoint=lambda: s2.sync(out, remote))
    finish(out, remote, {"name": name, "seed_set_sha256": sha, "steps": cfg["steps"],
                         "train_seconds": round(summary["train_time_s"], 1),
                         "seconds_per_step": round(summary["train_time_s"] / cfg["steps"], 3),
                         "peak_mem_gb": summary["peak_mem_gb"], "adapter_sha256": summary["adapter_sha256"],
                         "seconds": round(time.time() - t0, 1)})


def cmd_bench(a) -> None:
    """What a training step costs on the data a practice arm trains on after cycle 1 (seed set plus oracle
    episodes for the first 480 stream tasks: the stream's mix of one to four hops), with gradient
    checkpointing always on (Experiment 1), on for long micro-batches only, and off. Adapters are discarded."""
    import gc

    import torch
    from ccl import hf
    name = f"r{a.replicate}_bench"
    out, remote = dirs(a, name)
    fams, tools = s2.load_practice()
    rows = seed_rows(a.replicate)[0] + s2.oracle_rows(L.Stream(fams, tools, a.replicate).first(L.CYCLE_ACCEPT))[0]
    data = Path(a.out) / "_data"
    sha = write_jsonl(data / f"e2_bench_r{a.replicate}.jsonl", rows)
    result = {"name": name, "data_sha256": sha, "episodes": len({r["meta"]["task_id"] for r in rows}), "settings": {}}
    for label, ckpt in (("always", True), ("long_only", "auto"), ("off", False)):
        cfg = {**train_settings(), "steps": a.steps, "save_every": 10 ** 6, "eval_every": 10 ** 6, "val_file": None,
               "grad_checkpoint": ckpt, "model": s2.fetch_model(S15), "seed": a.replicate,
               "train_files": [str(data / f"e2_bench_r{a.replicate}.jsonl")], "out_dir": str(out / label)}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        failed = None
        try:
            summary = hf.train(cfg, fresh=True)
        except torch.OutOfMemoryError as e:
            failed = repr(e)[:200]
        if failed:  # outside the handler, where the failed step's memory can be released
            gc.collect()
            torch.cuda.empty_cache()
            result["settings"][label] = {"out_of_memory": failed}
        else:
            log = [json.loads(l) for l in open(out / label / "train_log.jsonl")]
            result["settings"][label] = {
                "seconds_per_step": round(summary["train_time_s"] / a.steps, 3), "peak_mem_gb": summary["peak_mem_gb"],
                "tokens_per_step": round(sum(r["seq_tokens"] for r in log) / len(log)),
                "tokens_per_second": round(log[-1]["tok_per_s"]), "last_loss": log[-1]["loss"]}
        shutil.rmtree(out / label, ignore_errors=True)
        finish(out, remote, result)


# ------------------------------------------------------------------ engine calibration

def cmd_calib(a) -> None:
    from ccl import engine, hf
    from ccl.runtime import run_batch
    fams, tools = s2.load_practice()
    for run in CALIB:
        cfg = json.loads((ROOT / f"configs/study02/e1/{run}.json").read_text())
        name = f"calib{a.tag}_{run}"  # a later engine revision is measured under a tag; nothing is overwritten
        out, remote = dirs(a, name)
        s2.restore(remote, out)
        if (out / "result.json").exists():
            continue
        model, tok = engine.load_policy_model(s2.fetch_model(cfg["model"]), adapter_dir(a, "e1", run))
        tasks = s2.e1_own_tasks(fams, tools, cfg["data"]["replicate"], cfg["data"]["D"])
        # on this GPU, in bfloat16: the cached engine against full re-encoding by the Experiment 1 policy
        probe = sorted(tasks, key=lambda t: -t.depth)[:24]
        new = engine.Engine(model, tok, name, grouped=a.grouped).run(probe)
        old = run_batch(probe, hf.HFPolicy(model, tok, name), batch_size=24)
        same = [[t["raw"] for t in x.trace.turns] == [t["raw"] for t in y.trace.turns] for x, y in zip(new, old)]
        eng = engine.Engine(model, tok, name, token_budget=a.token_budget, max_rows=a.max_rows, grouped=a.grouped)
        timing = s2.evaluate(eng, tasks, "own", out, name)
        rows = [json.loads(l) for l in open(out / "metrics.jsonl")]
        with gzip.open(ROOT / f"results/study02/runs/e1/{run}/metrics.jsonl.gz", "rt") as f:
            before = {r["task_id"]: r["success"] for r in map(json.loads, f) if r["split"] == "own"}
        both = [r for r in rows if r["task_id"] in before]  # all of them, unless the task files differ
        finish(out, remote, {
            "name": name, "episodes": len(rows), "episodes_compared": len(both), "seconds": timing["seconds"],
            "seconds_per_episode": round(timing["seconds"] / max(1, timing["run_now"]), 4),
            "success": sum(r["success"] for r in rows) / len(rows),
            "success_experiment_1": sum(before[r["task_id"]] for r in both) / max(1, len(both)),
            "changed_outcome": sum(r["success"] != before[r["task_id"]] for r in both),
            "same_text_as_full_re_encoding": f"{sum(same)}/{len(same)}", "engine": eng.stats,
            "token_budget": a.token_budget, "max_rows": a.max_rows, "grouped_attention": a.grouped})
        del model
        r = json.loads((out / "result.json").read_text())
        if r["changed_outcome"] > a.max_changed * r["episodes_compared"]:  # stops the job before anything is built on it
            sys.exit(f"[s2_e2] {name}: {r['changed_outcome']} of {r['episodes_compared']} outcomes changed")


# ------------------------------------------------------------------ seed adapter on dev, and the yield probe

def cmd_dev(a) -> None:
    from ccl import engine, hf
    name = f"r{a.replicate}_seed"
    out, remote = dirs(a, name)
    model_path, adapter = s2.fetch_model(S15), adapter_dir(a, "e2", name)
    if a.engine == "e1":  # the Experiment 1 path, if the new engine did not pass its calibration
        model, tok = hf.load_policy_model(model_path, adapter)
        policy = hf.HFPolicy(model, tok, name)
    else:
        model, tok = engine.load_policy_model(model_path, adapter)
        policy = engine.Engine(model, tok, name, token_budget=a.token_budget, max_rows=a.max_rows)
    timing = s2.evaluate(policy, s2.load_split("dev"), "v2/dev", out, name, batch=32,
                         checkpoint=lambda: s2.sync(out, remote))
    rows = [json.loads(l) for l in open(out / "metrics.jsonl")]
    result = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {"name": name}
    result.update({"dev_engine": a.engine, "dev_tasks": len(rows), "dev_success": sum(r["success"] for r in rows) / len(rows),
                   "dev_seconds": timing["seconds"], "dev_generation": policy.stats})
    finish(out, remote, result)


def cmd_probe(a) -> None:
    from ccl import engine
    arm = "practice"
    name = f"r{a.replicate}_{arm}"
    out, remote = dirs(a, name)
    s2.restore(remote, out)
    fams, tools = s2.load_practice()
    model, tok = engine.load_policy_model(s2.fetch_model(S15), adapter_dir(a, "e2", f"r{a.replicate}_seed"))
    eng = engine.Engine(model, tok, name, temperature=L.TEMPERATURE, top_p=L.TOP_P, token_budget=a.token_budget,
                        max_rows=a.max_rows)
    clock = {"seconds": 0.0, "attempts": 0}

    def run(tasks, seed):
        eng.seed = seed
        t0 = time.time()
        eps = eng.run(tasks, attempts=range(L.ATTEMPTS), stop_on_error=True)
        clock["seconds"] += time.time() - t0
        clock["attempts"] += len(eps)
        s2.sync(out, remote)
        return eps
    arm_ = L.Practice(out, a.replicate, arm, fams, tools)
    rec = arm_.cycle(1, run, max_tasks=a.tasks)
    rec.pop("rows")
    s2.sync(out, remote)
    finish(out, remote, {"name": name, "probe_tasks": len(rec["offered"]), "accepted_tasks": len(rec["accepted"]),
                         "yield": rec["yield"], "cycle": rec, "rollout_seconds": round(clock["seconds"], 1),
                         "attempts_sampled_now": clock["attempts"],
                         "seconds_per_attempt": round(clock["seconds"] / max(1, clock["attempts"]), 4),
                         "engine": eng.stats, "stream_dropped": arm_.stream.dropped,
                         "token_budget": a.token_budget, "max_rows": a.max_rows})


# ------------------------------------------------------------------ teacher qualification

def teacher_tasks(replicate: int = 0):
    """The first 50 tasks of each number of hops in the replicate's stream, with their stream indices."""
    fams, tools = s2.load_practice()
    st = L.Stream(fams, tools, replicate)
    idx, per = [], Counter()
    i = 0
    while len(idx) < 4 * TEACHER_TASKS_PER_HOP:
        h = st.get(i).depth
        if per[h] < TEACHER_TASKS_PER_HOP:
            per[h] += 1
            idx.append(i)
        i += 1
    return idx, [st.get(i) for i in idx]


def cmd_teacher(a) -> None:
    from ccl import engine
    name = "teacher_" + a.repo.split("/")[-1]
    out, remote = dirs(a, name)
    out.mkdir(parents=True, exist_ok=True)
    idx, tasks = teacher_tasks()
    where = {t.task_id: i for i, t in zip(idx, tasks)}
    t0 = time.time()
    model, tok = engine.load_policy_model(s2.fetch_model({"repo": a.repo, "revision": a.revision}), None)
    load_s = time.time() - t0
    kw = dict(token_budget=a.token_budget, max_rows=a.max_rows)
    result = {"name": name, "model": {"repo": a.repo, "revision": a.revision}, "tasks": len(tasks),
              "load_seconds": round(load_s, 1), "thresholds": QUALIFY, "passes": {}}
    rows = []
    todo = tasks
    for label, attempt, temp in (("greedy", 0, 0.0), ("retry", 1, TEACHER_RETRY_T)):
        eng = engine.Engine(model, tok, name, temperature=temp, top_p=L.TOP_P, seed="e2|teacher", **kw)
        t1 = time.time()
        eps = eng.run(todo, fewshot=True, attempts=(attempt,), stop_on_error=True)
        new = [L.attempt_row(where[e.task.task_id], e) for e in eps]
        rows += new
        result["passes"][label] = {"attempts": len(eps), "accepted": sum(not r["rejection"] for r in new),
                                   "seconds": round(time.time() - t1, 1), "engine": eng.stats}
        with gzip.open(out / "attempts.jsonl.gz", "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        todo = [e.task for e, r in zip(eps, new) if r["rejection"]]
        ok = {r["task_id"] for r in rows if not r["rejection"]}
        by_hop = defaultdict(list)
        for t in tasks:
            by_hop[t.depth].append(t.task_id in ok)
        result.update({"accepted": len(ok), "acceptance": len(ok) / len(tasks),
                       "acceptance_by_hops": {str(h): sum(v) / len(v) for h, v in sorted(by_hop.items())},
                       "rollout_seconds": round(sum(p["seconds"] for p in result["passes"].values()), 1),
                       "complete": label == "retry"})
        result["qualifies"] = bool(result["complete"] and result["acceptance"] >= QUALIFY["overall"]
                                   and result["acceptance_by_hops"]["4"] >= QUALIFY["four_hop"])
        finish(out, remote, result)


# ------------------------------------------------------------------ replay (CPU, a separate process)

def cmd_replay(a) -> None:
    """Replay every attempt that the acceptance rule would admit, from its recorded turns, against a fresh
    simulator; nothing here was computed by the process that sampled the attempts."""
    from ccl.runtime import FEWSHOT, Episode
    d = Path(a.dir)
    fams, tools = s2.load_practice()
    st = L.Stream(fams, tools, a.replicate)
    files = sorted(d.glob("attempts/*.jsonl.gz")) or [d / "attempts.jsonl.gz"]
    checked, failures = 0, []
    for f in files:
        for r in s2.read_gz(f):
            if r["rejection"]:
                continue
            task = st.get(r["index"])
            assert task.task_id == r["task_id"]
            if r["trace"]["messages"][0]["content"].endswith(FEWSHOT):  # a teacher sees the few-shot prompt
                ep = Episode(task, "replay", fewshot=True)
                for t in r["trace"]["turns"]:
                    ep.step(t["raw"])
                ok = ep.done and ep.trace.messages == r["trace"]["messages"] and not s2.rejection(task, ep)
                why = "ok" if ok else "differs on replay"
            else:
                why = L.replay(task, r["trace"])[1]
            checked += 1
            if why != "ok":
                failures.append({"file": f.name, "index": r["index"], "attempt": r["attempt"], "why": why})
    out = {"dir": d.name, "accepted_attempts_replayed": checked, "failed": len(failures), "failures": failures}
    (d / "replay.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "failures"}))
    sys.exit(1 if failures else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["seed", "bench", "calib", "dev", "probe", "teacher", "replay"])
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--out", default="runs2")
    ap.add_argument("--bucket", default=None, help="mounted storage directory to mirror outputs to")
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--tasks", type=int, default=400)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--engine", choices=["new", "e1"], default="new")
    ap.add_argument("--repo")
    ap.add_argument("--revision")
    ap.add_argument("--token-budget", type=int, default=160_000)
    ap.add_argument("--max-rows", type=int, default=128)
    ap.add_argument("--tag", default="", help="calib: suffix of the result directories")
    ap.add_argument("--grouped", action="store_true", help="calib: with R6, which is otherwise off everywhere")
    ap.add_argument("--max-changed", type=float, default=1.0, help="calib: fail above this share of changed outcomes")
    a = ap.parse_args()
    globals()[f"cmd_{a.command}"](a)


if __name__ == "__main__":
    main()
