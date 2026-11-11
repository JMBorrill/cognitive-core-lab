"""Study 02 protocol pieces shared by the run, scoring and analysis scripts.

Nothing here imports a tensor library. Training data comes from the practice space only:
`load_practice` reads `data/v2/practice`, and no function in this module opens `data/v2/sealed`
except `load_split`. That is called by the scoring script, by the reference-model gate, and by the
review's task checks (`scripts/s2_review.py`), which involve no student; never by training code
(`tests/test_v2.py` enforces this).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path

from . import compose as C
from . import verify
from .data import episode_to_examples
from .evaluate import _matches, episode_metrics
from .oracle import OraclePolicy
from .runtime import Episode, run_batch
from .tasks import Task

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data/v2"
E1_LEVELS = {25: 128, 100: 32, 400: 8, 1600: 2}  # distinct families -> instances per family (3,200 episodes)
E1_HOPS = (1, 2, 3, 4)
OWN_TASKS = 400  # new instances of a run's own training families
OWN_SEED0 = 1_000_000


def read_gz(path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def load_practice() -> tuple[list[dict], list[str]]:
    """The practice families and the tools a practice catalogue may show. The only training input."""
    return read_gz(V2 / "practice/families.jsonl.gz"), json.loads((V2 / "practice/tools.json").read_text())


def load_split(name: str) -> list[Task]:
    """Evaluation tasks: `dev`, or one of the sealed test splits."""
    where = "dev" if name == "dev" else "sealed"
    return [Task.from_dict(d) for d in read_gz(V2 / where / f"{name}.jsonl.gz")]


# ------------------------------------------------------------------ acceptance and metrics

def rejection(task: Task, ep: Episode, m: dict | None = None, protocol: int = 1) -> list[str]:
    """The one acceptance rule for every data source: the episode passes the validator and the
    runtime recorded no parse error, unknown tool or invalid argument. Criteria that need the
    gold graph (unnecessary or dependency-unsafe calls) are measured, never used here.

    Protocol 2 (Experiment 2, amendment 1; `ccl.verify`) adds two conditions that read only the episode
    and the task's stated answer: a multi-part answer gives each fact for its own part, and the trace
    shows no candidate of a list being told apart by whether its continuation exists. The default is
    protocol 1, the rule of Experiments 0 and 1, whose training sets and results must not change."""
    m = m or episode_metrics(task, ep)
    why = [k for k, bad in (("not_success", not m["success"]), ("parse_error", m["parse_errors"]),
                            ("unknown_tool", m["hallucinated_tools"]), ("invalid_args", m["invalid_args"])) if bad]
    if protocol >= 2:
        if m["success"] and not verify.parts_ok(task, ep.trace.final):
            why.append("answer_parts")
        if verify.shortcut(ep.trace.to_dict()):
            why.append("shortcut")
    return why


def dependency_safe(task: Task, ep: Episode) -> bool:
    """Study 01's dependency-safety check with one correction for graphs that call a tool more
    than once: a call is compared with the gold call whose arguments it matches, not with the
    first gold call of that tool. Identical to the original whenever tools are unique."""
    import re
    ident = re.compile(r"^[A-Z]{1,3}-\d{2,6}$")
    known = task.request
    ok_turn: dict[str, int] = {}
    calls = [r for r in ep.trace.calls if r.outcome != "limit"]
    for turn in sorted({r.turn for r in calls}):
        now = [r for r in calls if r.turn == turn]
        for r in now:
            flat = [y for x in (r.args or {}).values() for y in (x if isinstance(x, list) else [x])]
            if any(isinstance(x, str) and ident.match(x.strip()) and x.strip() not in known for x in flat):
                return False
            same = [g for g in task.gold if r.tool_key in (g.tool, g.fallback)]
            g = next((g for g in same if _matches(g, r)), next((g for g in same if g.deps), None))
            if g and any(ok_turn.get(d, 10 ** 9) >= turn for d in g.deps):
                return False
        for r in now:
            for g in task.gold:
                if r.outcome == "ok" and _matches(g, r):
                    ok_turn.setdefault(g.id, turn)
        known += " " + json.dumps([r.result for r in now])
    return True


def metrics_row(task: Task, ep: Episode, split: str, config: str) -> dict:
    m = episode_metrics(task, ep)
    tags = dict(t.split("=", 1) for t in task.tags if "=" in t)
    safe = dependency_safe(task, ep)
    parts, cut = verify.parts_ok(task, ep.trace.final), verify.shortcut(ep.trace.to_dict())
    # Added with protocol 2. `success` and `accepted` keep their meaning (the original validator, protocol 1).
    m.update({"check_parts": float(parts), "shortcut": cut, "success_parts": float(bool(m["success"]) and parts),
              "success_clean": float(bool(m["success"]) and parts and not cut)})
    m.update({"split": split, "config": config, "accepted": not rejection(task, ep, m),
              "dependency_safe_v2": float(safe),
              "strict_success": float(bool(m["success"]) and safe and m["unnecessary_calls"] == 0),
              "hops": int(tags.get("hops", task.depth)), "mod": tags.get("mod", ""), "phr": tags.get("phr", ""),
              "graph_seen": "graph_seen" in task.tags})
    return m


# ------------------------------------------------------------------ Experiment 1 data

def e1_order(families: list[dict], replicate: int) -> list[dict]:
    """Replicate `r` draws one seeded, balanced order of the practice space; the subset for D
    distinct families is its first D entries, so the levels are nested."""
    return C.balanced_order(families, f"e1|r{replicate}", E1_HOPS, horizon=max(E1_LEVELS))


def e1_train_tasks(families: list[dict], tools: list[str], replicate: int, d: int) -> list[Task]:
    subset = e1_order(families, replicate)[:d]
    return [C.make_task(f, i, "train", "practice", tools) for f in subset for i in range(E1_LEVELS[d])]


def e1_own_tasks(families: list[dict], tools: list[str], replicate: int, d: int) -> list[Task]:
    """New instances of the run's own training families, from the evaluation value pool."""
    subset = e1_order(families, replicate)[:d]
    per = max(1, OWN_TASKS // len(subset))
    return [C.make_task(f, OWN_SEED0 + i, "eval", "own", tools) for f in subset[:OWN_TASKS] for i in range(per)]


def oracle_rows(tasks: list[Task]) -> tuple[list[dict], dict]:
    """Oracle trajectories kept under the common acceptance rule, as SFT rows."""
    eps = run_batch(tasks, OraclePolicy(), batch_size=512)
    rows, rejected = [], 0
    for e in eps:
        if rejection(e.task, e):
            rejected += 1
        else:
            rows.extend(episode_to_examples(e, "oracle"))
    return rows, {"episodes": len(eps) - rejected, "rejected": rejected, "rows": len(rows)}


def rows_sha256(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode())
    return h.hexdigest()


def subset_stats(subset: list[dict]) -> dict:
    """What a training subset covers: measured for every run, so that the number of families can
    be told apart from tool coverage, depth and compositional complexity."""
    from collections import Counter
    return {"families": len(subset), "base_graphs": len({g for f in subset for g in f["graphs"]}),
            "tools": sorted({t for f in subset for t in f["tools"]}),
            "edges": sorted({e for f in subset for e in f["edges"]}),
            "hops": dict(sorted(Counter(f["hops"] for f in subset).items())),
            "modifier": dict(sorted(Counter(f["mod"] for f in subset).items())),
            "shape": dict(sorted(Counter(f["shape"] for f in subset).items())),
            "mean_calls": round(sum(f["calls"] for f in subset) / len(subset), 3)}


# ------------------------------------------------------------------ evaluation

def done_keys(metrics_path: Path) -> set:
    keys = set()
    if metrics_path.exists():
        for line in open(metrics_path):
            try:
                r = json.loads(line)
                keys.add((r["split"], r["task_id"]))
            except (json.JSONDecodeError, KeyError):
                pass  # a line cut off by an interruption is rerun
    return keys


def evaluate(policy, tasks: list[Task], split: str, out: Path, config: str, batch: int = 64,
             fewshot: bool = False, checkpoint=None) -> dict:
    """Score `tasks` and append one metrics row and one trace per episode. Resumable; `checkpoint`
    is called every 200 finished episodes (used to mirror partial results to durable storage)."""
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "metrics.jsonl"
    if mpath.exists() and mpath.read_bytes()[-1:] not in (b"", b"\n"):
        with open(mpath, "a") as f:  # finish a line cut off by an interruption
            f.write("\n")
    done = done_keys(mpath)
    todo = [t for t in tasks if (split, t.task_id) not in done]
    # Episodes advance in lock-step and every prompt in a batch is padded to the longest, so tasks of similar
    # depth and prompt length are scored together. The order is fixed and does not depend on any outcome.
    todo.sort(key=lambda t: (-t.depth, -len(t.request) - sum(len(x.signature()) for x in t.catalog), t.task_id))
    t0 = time.time()
    with open(mpath, "a") as mf, gzip.open(out / "traces.jsonl.gz", "at") as tf:
        n = [0]

        def on_done(ep):
            mf.write(json.dumps(metrics_row(ep.task, ep, split, config)) + "\n")
            mf.flush()
            tf.write(json.dumps({"split": split, "task_id": ep.task.task_id, "trace": ep.trace.to_dict()}) + "\n")
            n[0] += 1
            if checkpoint and n[0] % 200 == 0:
                tf.flush()
                checkpoint()
        if todo and hasattr(policy, "run"):  # ccl.engine.Engine drives its own groups, sized by tokens
            policy.run(todo, on_done=on_done, fewshot=fewshot)
        elif todo:
            run_batch(todo, policy, batch_size=batch, on_done=on_done, fewshot=fewshot)
    return {"split": split, "tasks": len(tasks), "run_now": len(todo), "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------ run plumbing (used inside GPU jobs)

def config_sha256(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def fetch_model(spec: dict) -> str:
    """Local path of a base model: a directory under models/ if present, else a pinned Hub snapshot."""
    local = ROOT / "models" / spec["repo"].split("/")[-1]
    if local.exists():
        return str(local)
    from huggingface_hub import snapshot_download
    return snapshot_download(spec["repo"], revision=spec["revision"],
                             allow_patterns=["*.json", "*.txt", "*.safetensors"])


def sync(src: Path, dst: Path | None) -> None:
    """Mirror `src` into `dst` (a mounted storage bucket): copy files that are new or changed and drop
    checkpoints that no longer exist locally. Whole files only, so an interruption cannot corrupt one."""
    import shutil
    if dst is None or not src.exists():
        return
    for p in sorted(src.rglob("*")):
        if p.is_file():
            q = dst / p.relative_to(src)
            if not q.exists() or q.stat().st_size != p.stat().st_size or q.stat().st_mtime < p.stat().st_mtime:
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, q)
    for q in sorted(dst.glob("**/checkpoints/step_*")):
        if not (src / q.relative_to(dst)).exists():
            shutil.rmtree(q, ignore_errors=True)


def restore(dst: Path | None, local: Path) -> None:
    """Copy a run directory back from the bucket so that an interrupted job resumes."""
    import shutil
    if dst is not None and dst.exists() and not local.exists():
        shutil.copytree(dst, local)


def environment() -> dict:
    import platform
    import subprocess
    import os
    info = {"python": platform.python_version(), "machine": platform.machine(),
            "job_id": os.environ.get("JOB_ID"), "accelerator": os.environ.get("ACCELERATOR")}
    try:
        info["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                        text=True).stdout.strip()
    except OSError:
        info["commit"] = None
    try:
        import peft
        import torch
        import transformers
        info.update({"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except ImportError:
        pass
    return info
