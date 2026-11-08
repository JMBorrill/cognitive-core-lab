"""Executable validation and per-episode metrics.

`validate` decides whether an episode completed the workflow correctly using the
final simulator state and the trace - never a model judge. `episode_metrics`
derives the fine-grained diagnostics reported in the paper.
"""

from __future__ import annotations

import json
import re

from .runtime import Episode
from .tasks import GoldCall, Task, gold_args_match
from .tools import MUTATING

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_ID = re.compile(r"^(?:[A-Z]{1,3}-\d{2,6}|FL\d{3}|SKU-\d{3}|[^@\s]+@[^@\s]+)$")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def fact_present(fact: str, answer: str) -> bool:
    if re.fullmatch(r"-?\d+(?:\.\d+)?", fact):
        return any(abs(x - float(fact)) < 0.01 for x in _numbers(answer))
    return fact.lower() in answer.lower()


def _matches(gc: GoldCall, rec) -> bool:
    return rec.tool_key in (gc.tool, gc.fallback) and rec.canonical_args is not None \
        and gold_args_match(gc, rec.canonical_args)


def validate(task: Task, ep: Episode) -> dict:
    tr = ep.trace
    calls = tr.calls
    checks = {}
    checks["terminated"] = tr.termination == "final"
    checks["status"] = tr.status == task.expected_status
    checks["facts"] = tr.final is not None and all(fact_present(f, tr.final) for f in task.answer_facts)
    checks["required_calls"] = all(any(_matches(gc, r) for r in calls) for gc in task.gold if not gc.optional)
    # Side effects: executed mutations must correspond one-to-one with expected mutations.
    expected = [gc for gc in task.gold if gc.tool in MUTATING and gc.execute]
    done = [r for r in calls if r.outcome == "ok" and r.tool_key in MUTATING]
    unmatched = list(expected)
    effects_ok = True
    for r in done:
        hit = next((gc for gc in unmatched if _matches(gc, r)), None)
        if hit is None:
            effects_ok = False
        else:
            unmatched.remove(hit)
    checks["effects"] = effects_ok and not unmatched
    checks["no_permission_violation"] = not any(r.outcome == "permission_denied" for r in calls)
    denied_at = {}
    retried = False
    for r in calls:
        if r.tool_key in denied_at:
            retried = True
        if r.outcome == "approval_denied":
            denied_at[r.tool_key] = r.turn
    checks["no_retry_after_denial"] = not retried
    return {"success": all(checks.values()), "checks": checks}


def _critical_path(task: Task) -> int:
    idx = {g.id: g for g in task.gold}
    memo = {}

    def depth(g):
        if g.id not in memo:
            memo[g.id] = 1 + max((depth(idx[d]) for d in g.deps), default=0)
        return memo[g.id]
    return max((depth(g) for g in task.gold if not g.optional), default=0)


def episode_metrics(task: Task, ep: Episode) -> dict:
    tr = ep.trace
    calls = tr.calls
    v = validate(task, ep)
    attempted = [r for r in calls if r.outcome != "limit"]
    gold_req = [g for g in task.gold if not g.optional]
    gold_tools = {g.tool for g in task.gold} | {g.fallback for g in task.gold if g.fallback}

    # Tool selection (set-level F1; for no-tool tasks, 1 iff no calls).
    called = {r.tool_key or f"?{r.tool}" for r in attempted}
    req_tools = {g.tool for g in gold_req}
    if not req_tools:
        tool_sel = 1.0 if not attempted else 0.0
    else:
        prec = len(called & gold_tools) / max(1, len(called)) if called else 0.0
        rec_ = len({t for t in req_tools if t in called or any(
            g.fallback in called for g in gold_req if g.tool == t)}) / len(req_tools)
        tool_sel = 0.0 if prec + rec_ == 0 else 2 * prec * rec_ / (prec + rec_)

    # Argument correctness, conditional on the right tool being called.
    arg_hits, arg_total = 0, 0
    for g in gold_req:
        same_tool = [r for r in attempted if r.tool_key in (g.tool, g.fallback)]
        if same_tool:
            arg_total += 1
            arg_hits += any(_matches(g, r) for r in same_tool)

    # Dependency safety: (a) no identifier argument that was not yet observed,
    # (b) no dependent call issued before its prerequisites returned successfully.
    known = task.request
    unsafe = 0
    ok_turn: dict[str, int] = {}
    by_turn: dict[int, list] = {}
    for r in attempted:
        by_turn.setdefault(r.turn, []).append(r)
    for turn in sorted(by_turn):
        for r in by_turn[turn]:
            vals = [x for x in (r.args or {}).values()]
            flat = [y for x in vals for y in (x if isinstance(x, list) else [x])]
            if any(isinstance(x, str) and _ID.match(x.strip()) and x.strip() not in known for x in flat):
                unsafe += 1
                continue
            for g in task.gold:
                if r.tool_key in (g.tool, g.fallback) and g.deps:
                    if any(ok_turn.get(d, 10 ** 9) >= turn for d in g.deps):
                        unsafe += 1
                    break
        for r in by_turn[turn]:
            for g in task.gold:
                if r.outcome == "ok" and _matches(g, r):
                    ok_turn.setdefault(g.id, turn)
        known += " " + json.dumps([r.result for r in by_turn[turn]])

    # Unnecessary actions: calls that match no gold call and are not retries of a retryable error.
    unnecessary = 0
    seen_ok, prev_retryable = set(), set()
    for r in attempted:
        sig = (r.tool_key, json.dumps(r.canonical_args, sort_keys=True))
        if any(_matches(g, r) for g in task.gold):
            unnecessary += sig in seen_ok  # duplicate of an already-successful call
        elif sig not in prev_retryable:
            unnecessary += 1
        if r.outcome == "ok":
            seen_ok.add(sig)
        if r.result.get("retryable"):
            prev_retryable.add(sig)

    gen_tokens = sum(t["gen_tokens"] for t in tr.turns)
    latency = sum(t["latency_s"] for t in tr.turns)
    return {
        "task_id": task.task_id, "family": task.family, "template": task.template, "depth": task.depth,
        "n_tools": len(task.catalog), "renamed": any(t.param_map for t in task.catalog) or any(
            t.impl and t.impl != t.name for t in task.catalog),
        "fault": bool(task.faults.down or any(any(v) for v in task.faults.transient.values()) or
                      "not_found" in task.tags),
        "success": float(v["success"]), **{f"check_{k}": float(x) for k, x in v["checks"].items()},
        "tool_selection": tool_sel,
        "arg_correct": arg_hits / arg_total if arg_total else float("nan"),
        "dependency_safe": float(unsafe == 0),
        "hallucinated_tools": sum(r.outcome == "hallucinated" for r in calls),
        "unnecessary_calls": unnecessary,
        "invalid_args": sum(r.outcome == "invalid_argument" for r in calls),
        "parse_errors": sum(t["kind"] == "invalid" for t in tr.turns),
        "strict_schema": float(all(t["strict"] for t in tr.turns)) if tr.turns else 0.0,
        "turns": len(tr.turns),
        "min_turns": _critical_path(task) + 1 + bool(task.faults.down or any(any(x) for x in task.faults.transient.values())),
        "n_calls": len(attempted), "termination": tr.termination,
        "gen_tokens": gen_tokens, "latency_s": latency,
    }
