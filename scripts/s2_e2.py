"""Study 02, Experiment 2: what a Stage 1 GPU job runs, and the CPU replay of what it produced.

    python scripts/s2_e2.py seed    --replicate 0 --out runs2 [--bucket /out]   # seed set and seed adapter
    python scripts/s2_e2.py bench   --replicate 0 --out runs2 [--bucket /out]   # training cost on a cycle-1 buffer
    python scripts/s2_e2.py calib   --out runs2 [--bucket /out]                 # engine against Experiment 1's rows
    python scripts/s2_e2.py dev     --replicate 0 --out runs2 [--bucket /out]   # seed adapter on dev
    python scripts/s2_e2.py probe   --replicate 0 --tasks 400 --out runs2 [--bucket /out]
    python scripts/s2_e2.py teacher --repo R --revision V --out runs2 [--bucket /out]
    python scripts/s2_e2.py replay  DIR                                         # CPU, any machine
    python scripts/s2_e2.py cycle   --replicate 0 --arm practice --cycle 1 --out runs2 [--bucket /out]   # Stage 2
    python scripts/s2_e2.py e1dev   --out runs2 [--bucket /out]                 # dev for Experiment 1's 1,600-family adapters
    python scripts/s2_e2.py audit   DIR                                         # CPU: the verifier audit's sample

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
    finish(out, remote, {"name": name, "seed_set_sha256": sha, "steps": cfg["steps"],  # also kept as train_result.json
                         "train_seconds": round(summary["train_time_s"], 1),
                         "seconds_per_step": round(summary["train_time_s"] / cfg["steps"], 3),
                         "peak_mem_gb": summary["peak_mem_gb"], "adapter_sha256": summary["adapter_sha256"],
                         "seconds": round(time.time() - t0, 1)})
    shutil.copyfile(out / "result.json", out / "train_result.json")  # `dev` writes result.json again
    s2.sync(out, remote)


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

def adversarial_tasks():
    """The adversarial set of amendment 1: `dev` tasks on worlds with complete candidate records."""
    from ccl import adversarial
    return adversarial.adversarial_dev()


def cmd_dev(a) -> None:
    """The seed adapter on `dev`. With a tag the result goes to its own directory (`r0_seed<tag>`), so an
    adapter can be scored again on another engine beside the first scoring; with --adv also on the
    adversarial set."""
    from ccl import engine, hf
    name = f"r{a.replicate}_seed{a.tag}"
    out, remote = dirs(a, name)
    s2.restore(remote, out)
    model_path, adapter = s2.fetch_model(S15), adapter_dir(a, "e2", f"r{a.replicate}_seed")
    if a.engine == "e1":  # the Experiment 1 path, if the new engine did not pass its calibration
        model, tok = hf.load_policy_model(model_path, adapter)
        policy = hf.HFPolicy(model, tok, name)
    else:
        model, tok = engine.load_policy_model(model_path, adapter)
        policy = engine.Engine(model, tok, name, token_budget=a.token_budget, max_rows=a.max_rows, grouped=a.grouped)
    timing = s2.evaluate(policy, s2.load_split("dev"), "v2/dev", out, name, batch=32,
                         checkpoint=lambda: s2.sync(out, remote))
    result = {"name": name, "dev_engine": a.engine, "grouped_attention": a.grouped, "dev_seconds": timing["seconds"]}
    if a.adv:
        result["adv_seconds"] = s2.evaluate(policy, adversarial_tasks(), "v2/dev_adv", out, name, batch=32,
                                            checkpoint=lambda: s2.sync(out, remote))["seconds"]
    rows = [r for r in map(json.loads, open(out / "metrics.jsonl")) if r["split"] == "v2/dev"]
    result.update({"dev_tasks": len(rows), "dev_success": sum(r["success"] for r in rows) / len(rows),
                   "dev_generation": policy.stats})
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


# ------------------------------------------------------------------ one cycle of a practice arm (plan §3.5)

def cmd_cycle(a) -> None:
    """Rollouts with the current adapter, replay, buffer; a new adapter trained from the base model on seed
    plus buffer; that adapter on `dev`. Every part is skipped if it is already done, so a job that was cut
    off is submitted again as it is. The run stops (exit 3) if the cycle's yield is under 20% or `dev` falls
    by more than 5 points against the adapter that produced the cycle's rollouts (plan §7)."""
    import gc

    import torch
    from ccl import engine, hf
    k, name = a.cycle, f"r{a.replicate}_{a.arm}{a.tag}"  # a tag keeps a repeated pilot beside the first
    out, remote = dirs(a, name)
    s2.restore(remote, out)
    sub, sub_remote = out / f"c{k}", (remote / f"c{k}" if remote else None)
    sub.mkdir(parents=True, exist_ok=True)
    result = json.loads((sub / "result.json").read_text()) if (sub / "result.json").exists() else {"name": f"{name}_c{k}"}
    fams, tools = s2.load_practice()
    arm = L.Practice(out, a.replicate, a.arm, fams, tools)
    before = f"r{a.replicate}_seed" if k == 1 else f"{name}/c{k - 1}"  # the adapter this cycle's rollouts come from
    scored = f"r{a.replicate}_seed{a.tag}" if k == 1 else before  # and where its dev rows are
    if len(arm.state["cycles"]) < k:  # ---- rollouts
        model, tok = engine.load_policy_model(s2.fetch_model(S15), adapter_dir(a, "e2", before))
        eng = engine.Engine(model, tok, name, temperature=L.TEMPERATURE, top_p=L.TOP_P, token_budget=a.token_budget,
                            max_rows=a.max_rows, grouped=a.grouped)
        clock = {"seconds": 0.0, "attempts": 0}

        def run(tasks, seed):
            eng.seed = seed
            t0 = time.time()
            eps = eng.run(tasks, attempts=range(L.ATTEMPTS), stop_on_error=True)
            clock["seconds"] += time.time() - t0
            clock["attempts"] += len(eps)
            s2.sync(out, remote)
            return eps
        rec = arm.cycle(k, run)
        result.update({"rollout_seconds_this_job": round(clock["seconds"], 1), "attempts_sampled_this_job": clock["attempts"],
                       "seconds_per_attempt": round(clock["seconds"] / max(1, clock["attempts"]), 4), "engine": eng.stats})
        del model, eng
        gc.collect()
        torch.cuda.empty_cache()
    rec = arm.state["cycles"][k - 1]
    result.update({"cycle": k, "protocol": arm.protocol, "grouped_attention": a.grouped,
                   "tasks_offered": len(rec["offered"]), "accepted": len(rec["accepted"]), "yield": rec["yield"],
                   "accepted_strict": rec.get("accepted_strict", 0), "replay_failed": rec.get("replay_failed", 0),
                   "attempts": rec["attempts"], "buffer_rows_sha256": rec["rows_sha256"]})
    finish(sub, sub_remote, result)
    s2.sync(out, remote)
    if rec["yield"] < 0.20:
        sys.exit(3)
    # ---- a new adapter from the base model on seed plus buffer, two passes at the planned number of steps
    rows = seed_rows(a.replicate)[0] + [r for c in range(1, k + 1) for r in s2.read_gz(out / f"buffer/c{c}.jsonl.gz")]
    data = Path(a.out) / "_data"
    sha = write_jsonl(data / f"e2_{name}_c{k}.jsonl", rows)
    steps = (L.SEED_FAMILIES * L.SEED_INSTANCES + k * L.CYCLE_ACCEPT) // 4
    cfg = {**train_settings(), "steps": steps, "model": s2.fetch_model(S15), "train_files": [str(data / f"e2_{name}_c{k}.jsonl")],
           "val_file": val_file(data), "out_dir": str(sub / "train"), "seed": a.replicate}
    summary = hf.train(cfg, on_checkpoint=lambda: s2.sync(out, remote))
    result.update({"train_data_sha256": sha, "train_episodes": len({(r["meta"]["task_id"], r["meta"]["source"]) for r in rows}),
                   "steps": steps, "adapter_sha256": summary["adapter_sha256"], "train_seconds": round(summary["train_time_s"], 1),
                   "seconds_per_step": round(summary["train_time_s"] / max(1, steps - summary["resumed_from"]), 3)})
    finish(sub, sub_remote, result)
    gc.collect()
    torch.cuda.empty_cache()
    # ---- that adapter on dev
    model, tok = engine.load_policy_model(s2.fetch_model(S15), sub / "train")
    policy = engine.Engine(model, tok, f"{name}_c{k}", token_budget=a.token_budget, max_rows=a.max_rows, grouped=a.grouped)
    timing = s2.evaluate(policy, s2.load_split("dev"), "v2/dev", sub, f"{name}_c{k}", checkpoint=lambda: s2.sync(out, remote))
    rows = [r for r in map(json.loads, open(sub / "metrics.jsonl")) if r["split"] == "v2/dev"]
    dev = sum(r["success"] for r in rows) / len(rows)  # the original validator, as in the first pilot
    prior = Path(a.out) / "e2" / scored
    prior.mkdir(parents=True, exist_ok=True)
    for f in ("result.json", "metrics.jsonl"):  # the dev score of the adapter this cycle's rollouts came from
        if not (prior / f).exists() and a.bucket and (Path(a.bucket) / "e2" / scored / f).exists():
            shutil.copyfile(Path(a.bucket) / "e2" / scored / f, prior / f)
    was = [json.loads(l) for l in open(prior / "metrics.jsonl")] if (prior / "metrics.jsonl").exists() else []
    was = [r["success"] for r in was if r["split"] == "v2/dev"]
    result.update({"dev_success": dev, "dev_tasks": len(rows), "dev_seconds_this_job": timing["seconds"],
                   "dev_success_before": sum(was) / len(was) if was else None,
                   "dev_success_clean": sum(r.get("success_clean", r["success"]) for r in rows) / len(rows)})
    finish(sub, sub_remote, result)
    s2.sync(out, remote)
    if was and dev < sum(was) / len(was) - 0.05:
        sys.exit(3)
    if a.adv:  # ---- and on the adversarial set (amendment 1); reported, never used by a stop rule
        t = s2.evaluate(policy, adversarial_tasks(), "v2/dev_adv", sub, f"{name}_c{k}", checkpoint=lambda: s2.sync(out, remote))
        adv = [r for r in map(json.loads, open(sub / "metrics.jsonl")) if r["split"] == "v2/dev_adv"]
        result.update({"adv_tasks": len(adv), "adv_seconds_this_job": t["seconds"],
                       "adv_success_parts": sum(r["success_parts"] for r in adv) / len(adv)})
        finish(sub, sub_remote, result)
        s2.sync(out, remote)


def cmd_e1dev(a) -> None:
    """`dev` for the three 1,600-family adapters of Experiment 1, which that experiment did not score: the
    comparator of Stage 1's criterion (c) and of the pilot's monitoring. Scored after the fact, on the engine
    as calibrated in Stage 1, and labelled so wherever it is quoted."""
    from ccl import engine
    for r in range(3):
        run = f"e1_s15_D1600_r{r}"
        cfg = json.loads((ROOT / f"configs/study02/e1/{run}.json").read_text())
        name = f"e1dev_{run}"
        out, remote = dirs(a, name)
        s2.restore(remote, out)
        if (out / "result.json").exists():
            continue
        model, tok = engine.load_policy_model(s2.fetch_model(cfg["model"]), adapter_dir(a, "e1", run))
        eng = engine.Engine(model, tok, name, token_budget=a.token_budget, max_rows=a.max_rows)
        timing = s2.evaluate(eng, s2.load_split("dev"), "v2/dev", out, name, checkpoint=lambda: s2.sync(out, remote))
        rows = [json.loads(l) for l in open(out / "metrics.jsonl")]
        finish(out, remote, {"name": name, "adapter_of": run, "dev_tasks": len(rows),
                             "dev_success": sum(x["success"] for x in rows) / len(rows), "dev_seconds": timing["seconds"],
                             "engine": eng.stats, "scored_after_the_fact": True})
        del model


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
            protocol = r.get("protocol", 1)  # files written before amendment 1 hold protocol-1 attempts
            task = st.get(r["index"])
            assert task.task_id == r["task_id"]
            if r["trace"]["messages"][0]["content"].endswith(FEWSHOT):  # a teacher sees the few-shot prompt
                ep = Episode(task, "replay", fewshot=True)
                for t in r["trace"]["turns"]:
                    ep.step(t["raw"])
                ok = ep.done and ep.trace.messages == r["trace"]["messages"] and not s2.rejection(task, ep, protocol=protocol)
                why = "ok" if ok else "differs on replay"
            else:
                why = L.replay(task, r["trace"], protocol=protocol)[1]
            checked += 1
            if why != "ok":
                failures.append({"file": f.name, "index": r["index"], "attempt": r["attempt"], "why": why})
    out = {"dir": d.name, "accepted_attempts_replayed": checked, "failed": len(failures), "failures": failures}
    (d / f"replay{a.tag}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "failures"}))
    sys.exit(1 if failures else 0)


def cmd_audit(a) -> None:
    """The verifier audit's sample for a practice arm (plan §5), with what a program can say about each
    episode: whether it makes the oracle's calls, was found by elimination, used an identifier it had not
    observed, or states values the reference answer does not. Episodes that differ from the oracle are listed
    for reading; verdicts are recorded by hand in `audit_verdicts.json` beside the output.

    For an arm run under protocol 2 the procedure is the one pre-registered in amendment 1: the flags also
    include the shortcut and per-part checks of the acceptance rule (which no buffered episode should fail)
    and any use of two candidates of one list, and 20 sampled episodes that carry no flag are marked for
    reading as well, as a check on the flags themselves."""
    import random
    import re

    from ccl import verify

    from ccl.oracle import OraclePolicy
    from ccl.runtime import run_batch
    d = Path(a.dir)
    fams, tools = s2.load_practice()
    arm = L.Practice(d, a.replicate, a.arm, fams, tools)
    value = re.compile(r"-?\d[\d,]*(?:\.\d+)?|[A-Z]{1,3}-\d{2,6}")
    accepted = []
    for c in arm.state["cycles"]:
        recs = [x for f in sorted(d.glob(f"attempts/c{c['cycle']}_*.jsonl.gz")) for x in s2.read_gz(f)]
        by = defaultdict(list)
        for x in recs:
            by[x["index"]].append(x)
        for i in c["accepted"]:
            p = L.choose(by[i], a.arm != "practice-unverified", f"e2|{a.replicate}|{a.arm}|{c['cycle']}")
            accepted.append({"cycle": c["cycle"], "index": i, "attempt": p["attempt"], "strict": p["strict"], "row": p})
    sample = L.audit_sample(accepted, a.sample)
    out = []
    for x in sample:
        task, p = arm.stream.get(x["index"]), x.pop("row")
        o = run_batch([task], OraclePolicy())[0]
        gold = sorted((c.tool_key, json.dumps(c.canonical_args, sort_keys=True)) for c in o.trace.calls)
        mine = sorted((c["tool_key"] or c["tool"], json.dumps(c["canonical_args"], sort_keys=True))
                      for c in p["trace"]["calls"] if c["outcome"] != "limit")
        num = lambda v: v if "-" in v[1:] else str(float(v.replace(",", "")))  # noqa: E731  "87.8" is "87.80"
        said = {num(v) for v in value.findall(p["trace"]["final"] or "")}
        extra = sorted(said - {num(v) for v in value.findall(task.reference_answer + " " + task.request)})
        checks = [("found_by_elimination", L.by_elimination(p["trace"])),
                  ("unobserved_identifier", L.unobserved_identifier(task, p["trace"])),
                  ("values_not_in_the_reference_answer", bool(extra)),
                  ("calls_differ_from_the_oracle", mine != gold)]
        if arm.protocol >= 2:
            checks = [("shortcut", verify.shortcut(p["trace"])), ("answer_parts", not verify.parts_ok(task, p["trace"]["final"])),
                      ("uses_two_candidates_of_a_list", verify.enumerates(p["trace"]))] + checks[1:]
        flags = [k for k, on in checks if on]
        out.append({**x, "task_id": task.task_id, "hops": task.depth, "flags": flags, "extra_values": extra,
                    "request": task.request, "reference_answer": task.reference_answer, "answer": p["trace"]["final"],
                    "calls": [[c["turn"], c["tool"], c["args"], c["outcome"], c["result"].get("error", "")]
                              for c in p["trace"]["calls"]],
                    "oracle_calls": [[c.turn, c.tool, c.args] for c in o.trace.calls]})
    res = {"arm": d.name, "accepted_episodes": len(accepted), "not_strict": sum(not x["strict"] for x in accepted),
           "sampled": len(out), "to_read": sum(bool(x["flags"]) for x in out), "sample": out}
    if arm.protocol >= 2:
        plain = [x for x in out if not x["flags"]]
        for x in random.Random("e2|audit|blind").sample(plain, min(20, len(plain))):
            x["blind"] = True
        res = {**res, "protocol": arm.protocol, "blind": sum(bool(x.get("blind")) for x in out),
               "to_read": sum(bool(x["flags"]) or bool(x.get("blind")) for x in out), "sample": out}
    (d / "audit.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "sample"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["seed", "bench", "calib", "dev", "probe", "teacher", "replay", "cycle", "e1dev",
                                        "audit"])
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
    ap.add_argument("--arm", default="practice", choices=list(L.ARMS))
    ap.add_argument("--cycle", type=int, default=1)
    ap.add_argument("--sample", type=int, default=200, help="audit: episodes to sample")
    ap.add_argument("--tag", default="", help="calib: suffix of the result directories; replay: of replay.json")
    ap.add_argument("--grouped", action="store_true", help="the corrected engine revision R6 (calib, dev, cycle)")
    ap.add_argument("--adv", action="store_true", help="dev, cycle: also score the adversarial set of amendment 1")
    ap.add_argument("--max-changed", type=float, default=1.0, help="calib: fail above this share of changed outcomes")
    a = ap.parse_args()
    globals()[f"cmd_{a.command}"](a)


if __name__ == "__main__":
    main()
