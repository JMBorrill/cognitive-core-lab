"""Submit Study 02 GPU jobs to Hugging Face Jobs under a hard spending cap, and record what they cost.

    python scripts/s2_launch.py status                    # ledger, provider-reported usage, open jobs
    python scripts/s2_launch.py submit JOB.json [--dry-run]
    python scripts/s2_launch.py settle                    # close the books on finished jobs
    python scripts/s2_launch.py pull EXPERIMENT           # copy result files from the bucket into results/study02/runs
    python scripts/s2_launch.py logs JOB_ID | cancel JOB_ID

How the cap holds. Before a job is submitted its worst case (timeout x price per minute) is written to
the ledger as a reservation, and the submission is refused if the billed cost of closed jobs, plus every
open reservation, plus the new one would exceed the cap minus a small reserve. The provider stops a job at
its timeout, so a job can never cost more than its reservation. When a job ends, the reservation is replaced
by the cost the provider billed for it, and the provider's running total is recorded; should that total ever
exceed the ledger's own sum, it is the one that counts. Jobs are never retried automatically.

The account's token is injected by the session's network proxy; this script never reads or stores it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET = ROOT / "configs/study02/budget.json"
LEDGER = ROOT / "results/study02/ledger.jsonl"
TERMINAL = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}
os.environ.setdefault("HF_TOKEN", "hf_proxy_injected")  # placeholder; the proxy supplies the real credential


def now() -> str:
    return time.strftime("%FT%TZ", time.gmtime())


def read_ledger(path: Path = LEDGER) -> list[dict]:
    return [json.loads(l) for l in open(path)] if path.exists() else []


def append(event: dict, path: Path = LEDGER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"time": now(), **event}) + "\n")


def books(ledger: list[dict]) -> dict:
    """Money spent by closed jobs, money reserved by open jobs, and the open jobs themselves.

    A closed job counts at the provider's billed cost once that is known (`provider_cost` events), otherwise
    at the ledger's own estimate. An open job counts at its full reservation, which it cannot exceed. Their
    sum is an upper bound on the final bill; the provider's running total is a lower bound, and the larger of
    the two is what counts as committed."""
    jobs: dict[str, dict] = {}
    for e in ledger:
        j = jobs.setdefault(e.get("key", "-"), {})
        if e["event"] == "reserve":
            j.update(reserved=e["reserved_usd"], open=True, flavor=e["flavor"], price=e["price_per_min"])
        elif e["event"] == "submitted":
            j["job_id"] = e["job_id"]
        elif e["event"] == "submit_failed":
            j.update(open=False, cost=0.0)
        elif e["event"] == "settled":
            j.update(open=False, cost=e["cost_usd"])
        elif e["event"] == "provider_cost":
            j.update(cost=e["cost_usd"])
    provider = next((e["provider_used_usd"] for e in reversed(ledger) if e["event"] == "reconcile"), 0.0)
    settled = sum(j.get("cost", 0.0) for j in jobs.values() if not j.get("open"))
    reserved = sum(j["reserved"] for j in jobs.values() if j.get("open"))
    return {"settled_usd": round(settled, 4), "provider_used_usd": provider, "reserved_usd": round(reserved, 4),
            "committed_usd": round(max(settled + reserved, provider), 4),
            "open": {k: j for k, j in jobs.items() if j.get("open")},
            "closed": {k: j for k, j in jobs.items() if "reserved" in j and not j.get("open")}}


def may_submit(ledger: list[dict], budget: dict, reservation: float) -> tuple[bool, str]:
    """The budget rule. Pure, so that it can be tested without a network."""
    b = books(ledger)
    limit = budget["cap_usd"] - budget["reserve_usd"]
    total = b["committed_usd"] + reservation
    if len(b["open"]) >= budget["max_parallel"]:
        return False, f"{len(b['open'])} jobs already open (max_parallel {budget['max_parallel']})"
    if total > limit + 1e-9:
        return False, (f"refused: {b['committed_usd']:.2f} USD already committed (closed jobs {b['settled_usd']:.2f}, "
                       f"open reservations {b['reserved_usd']:.2f}, provider total {b['provider_used_usd']:.2f}) + this "
                       f"job {reservation:.2f} = {total:.2f} exceeds {limit:.2f} "
                       f"(cap {budget['cap_usd']:.2f} minus reserve {budget['reserve_usd']:.2f})")
    return True, f"ok: {total:.2f} of {limit:.2f} USD committed after this job"


def reservation_usd(timeout_min: int, price_per_min: float) -> float:
    return round((timeout_min + 1) * price_per_min, 4)  # one extra minute for per-minute rounding


def job_cost_usd(billable_secs: float, price_per_min: float) -> float:
    """Cost of a finished job. The provider bills started minutes including start-up, which came to one
    minute more than the reported lifetime on the first job, so a minute is added."""
    return round((math.ceil(billable_secs / 60) + 1) * price_per_min, 4) if billable_secs else 0.0


# ------------------------------------------------------------------ provider calls

def get_json(path: str):
    req = urllib.request.Request(f"https://huggingface.co{path}", headers={"Authorization": "Bearer proxy"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def live_price_per_min(flavor: str) -> float:
    hw = next(h for h in get_json("/api/jobs/hardware") if h["name"] == flavor)
    assert hw["unitLabel"] == "minute"
    return hw["unitCostMicroUSD"] / 1e6


def provider_usage() -> tuple[float, dict[str, float]]:
    """The provider's own figures: total used this period, and the billed cost of each job so far."""
    u = get_json("/api/settings/billing/usage/jobs")["usage"]
    return u["usedMicroUsd"] / 1e6, {d["jobId"]: d["totalCostMicroUsd"] / 1e6 for d in u.get("jobDetails", [])}


def api():
    from huggingface_hub import HfApi
    return HfApi()


def job_command(spec: dict, budget: dict, commit: str) -> list[str]:
    cfgs = " ".join(spec["configs"])
    steps = [f"git clone --quiet {budget['repo']} /work/repo", "cd /work/repo", f"git checkout --quiet {commit}",
             "uv pip install --system --quiet -r requirements-cuda.lock",
             "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
             # stop at once if this PyTorch build cannot use the GPU: nothing may fall back to the CPU
             "python -c \"import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
             "print('torch', torch.__version__, torch.cuda.get_device_name(0))\""]
    if "prepare_v1" in spec["stages"]:
        steps.append("python scripts/s2_prepare_v1.py")
    if "train" in spec["stages"]:
        steps.append(f"python scripts/s2_train.py {cfgs} --out /work/runs --bucket /out")
    if "score" in spec["stages"]:
        steps.append(f"python scripts/s2_score.py {cfgs} --out /work/runs --bucket /out")
    steps += spec.get("extra", [])
    return ["bash", "-c", "set -euo pipefail; " + " && ".join(steps)]


def head_commit() -> str:
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "ccl", "scripts", "configs", "data",
                            "requirements-cuda.lock", ":!configs/study02/jobs"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    assert not dirty, "commit and push code, configs and data before submitting a job:\n" + dirty
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    remote = subprocess.run(["git", "branch", "-r", "--contains", commit], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    assert remote, f"commit {commit[:10]} is not pushed; the job could not check it out"
    return commit


# ------------------------------------------------------------------ commands

def submit(spec_path: str, dry_run: bool) -> None:
    budget = json.loads(BUDGET.read_text())
    spec = json.loads(Path(spec_path).read_text())
    ledger = read_ledger()
    assert spec["key"] not in {e.get("key") for e in ledger}, f"job key {spec['key']} was already used"
    assert spec["flavor"] in budget["flavors"], f"flavor {spec['flavor']} is not approved in the budget file"
    price = live_price_per_min(spec["flavor"])
    assert price <= budget["flavors"][spec["flavor"]] / 60 * 1.001, "the live price is above the budgeted price"
    need = reservation_usd(spec["timeout_min"], price)
    ok, why = may_submit(ledger, budget, need)
    print(f"[launch] {spec['key']}: {spec['flavor']} for at most {spec['timeout_min']} min, "
          f"reservation {need:.2f} USD. {why}")
    if not ok:
        sys.exit(2)
    commit = head_commit()
    command = job_command(spec, budget, commit)
    if dry_run:
        print("[launch] dry run, nothing submitted:\n ", command[-1])
        return
    append({"event": "reserve", "key": spec["key"], "flavor": spec["flavor"], "timeout_min": spec["timeout_min"],
            "price_per_min": price, "reserved_usd": need, "commit": commit, "configs": spec["configs"]})
    try:
        from huggingface_hub import Volume
        job = api().run_job(image=budget["image"], command=command, flavor=spec["flavor"],
                            timeout=f"{spec['timeout_min']}m", attempts=1, name=spec["key"],
                            labels={"project": "ccl-study02", "experiment": spec.get("experiment", "x")},
                            volumes=[Volume(type="bucket", source=budget["bucket"], mount_path="/out")])
    except Exception as e:  # nothing runs, so the reservation is released
        append({"event": "submit_failed", "key": spec["key"], "error": repr(e)[:500]})
        raise
    append({"event": "submitted", "key": spec["key"], "job_id": job.id, "url": job.url})
    print(f"[launch] submitted {job.id} {job.url}")


def settle() -> None:
    a = api()
    total, billed = provider_usage()
    state = books(read_ledger())
    for key, j in state["open"].items():
        if "job_id" not in j:
            continue
        info = a.inspect_job(job_id=j["job_id"])
        d = info.durations
        print(f"[settle] {key} {j['job_id']}: {info.status.stage} {info.status.message or ''} durations={d}")
        if info.status.stage in TERMINAL:
            secs = (d.running_secs or 0) if d else 0
            cost = billed.get(j["job_id"], job_cost_usd(secs, j["price"]))  # the provider's bill where it has one
            append({"event": "settled", "key": key, "job_id": j["job_id"], "stage": info.status.stage,
                    "message": info.status.message, "running_secs": secs, "cost_usd": round(cost, 4),
                    "cost_source": "provider" if j["job_id"] in billed else "estimate", "reserved_usd": j["reserved"]})
    for key, j in state["closed"].items():  # replace earlier estimates by the billed cost
        if j.get("job_id") in billed and abs(billed[j["job_id"]] - j.get("cost", 0.0)) > 0.0005:
            append({"event": "provider_cost", "key": key, "job_id": j["job_id"], "cost_usd": round(billed[j["job_id"]], 4)})
    append({"event": "reconcile", "provider_used_usd": total, "ledger_settled_usd": books(read_ledger())["settled_usd"]})
    status()


def status() -> None:
    budget = json.loads(BUDGET.read_text())
    b = books(read_ledger())
    print(json.dumps({"cap_usd": budget["cap_usd"], "reserve_usd": budget["reserve_usd"],
                      **{k: v for k, v in b.items() if k not in ("open", "closed")}, "open_jobs": sorted(b["open"]),
                      "available_usd": round(budget["cap_usd"] - budget["reserve_usd"] - b["committed_usd"], 2)},
                     indent=1))


def pull(experiment: str) -> None:
    """Copy the small result files of an experiment from the bucket (no checkpoints, no weights)."""
    budget = json.loads(BUDGET.read_text())
    a = api()
    keep = ("metrics.jsonl", "run.json", "train_run.json", "test_access.jsonl", "train_log.jsonl", "config.json",
            "summary.json", "adapter_config.json", "DONE")
    files = [f for f in a.list_bucket_tree(budget["bucket"], prefix=experiment + "/", recursive=True)
             if getattr(f, "type", "file") == "file" and f.path.endswith(keep) and "/checkpoints/" not in f.path]
    dest = ROOT / "results/study02/runs"
    pairs = [(f.path, str(dest / f.path)) for f in files]
    for _, local in pairs:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
    a.download_bucket_files(budget["bucket"], pairs)
    print(f"[pull] {len(pairs)} files -> {dest / experiment}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["status", "submit", "settle", "pull", "logs", "cancel"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.command == "status":
        status()
    elif a.command == "submit":
        submit(a.arg, a.dry_run)
    elif a.command == "settle":
        settle()
    elif a.command == "pull":
        pull(a.arg)
    elif a.command == "logs":
        for line in api().fetch_job_logs(job_id=a.arg):
            print(line)
    elif a.command == "cancel":
        api().cancel_job(job_id=a.arg)
        print("[cancel] requested; run settle to close the books")


if __name__ == "__main__":
    main()
