"""Agent orchestration runtime.

The loop: render prompt -> policy emits one JSON action -> parse + validate ->
enforce permissions / approval / limits -> execute on the simulator -> append
observations -> repeat until a final answer or a limit is hit.

`Episode` is a step-able state machine so that many episodes can be advanced in
lock-step with a batched model (`run_batch`). Prompt rendering here is the single
source of truth used by data generation, training and evaluation.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .tasks import Task
from .tools import Simulator, ToolError, validate_args

STATUSES = ("done", "refused", "blocked", "failed")


@dataclass(frozen=True)
class Limits:
    max_turns: int = 8
    max_calls_per_turn: int = 6
    max_total_calls: int = 16
    max_parse_errors: int = 2
    tool_timeout_s: float = 2.0  # wall-clock guard on simulator execution


# ------------------------------------------------------------------ prompts

SYSTEM_TEMPLATE = """You are a careful tool-using assistant. Today is {today} ({weekday}). The user's role is: {role}.
Reply with exactly one JSON object and nothing else.
To call tools: {{"calls": [{{"tool": "<name>", "args": {{...}}}}]}}
Put several calls in one list only when they are independent; they then run in parallel.
To finish: {{"final": "<answer for the user>", "status": "done|refused|blocked|failed"}}
Rules:
- Use only the tools listed below, with exactly the listed parameter names.
- Take argument values from the request or from earlier observations. Never invent IDs.
- Tools tagged [role: X] may only be used when the user's role is X; otherwise do not call them and finish with status "refused".
- Tools tagged [approval] ask a human first. If approval is denied, do not retry; finish with status "blocked".
- If a tool fails with "retryable": true, retry it once. If a tool is unavailable, use an alternative tool if one exists.
- If the request cannot be completed (e.g. something is not found), finish with status "failed" and explain.
- Stop as soon as the request is satisfied; do not make unnecessary calls.
Tools:
{tools}"""


FEWSHOT = """
Examples (illustrative tools that are NOT available to you):
User: What is the population of the city where Ada works?
Assistant: {"calls": [{"tool": "find_person", "args": {"name": "Ada"}}]}
User: Observations:
{"tool": "find_person", "result": {"person_id": "P-17", "city_id": "CT-9"}}
Assistant: {"calls": [{"tool": "city_stats", "args": {"city_id": "CT-9"}}]}
User: Observations:
{"tool": "city_stats", "result": {"population": 52000}}
Assistant: {"final": "About 52,000 people live there.", "status": "done"}
User: Check the price of apples and pears.
Assistant: {"calls": [{"tool": "price", "args": {"item": "apples"}}, {"tool": "price", "args": {"item": "pears"}}]}"""


def render_system(task: Task, fewshot: bool = False) -> str:
    from datetime import date
    wd = date.fromisoformat(task.today).strftime("%A")
    tools = "\n".join(f"- {t.signature()}" for t in task.catalog)
    sys = SYSTEM_TEMPLATE.format(today=task.today, weekday=wd, role=task.role, tools=tools)
    return sys + FEWSHOT if fewshot else sys


def render_observations(results: list[dict]) -> str:
    return "Observations:\n" + "\n".join(json.dumps(r, ensure_ascii=False) for r in results)


def canonical_action_text(action: dict) -> str:
    """Normalised serialisation used for training targets."""
    if "final" in action:
        return json.dumps({"final": action["final"], "status": action["status"]}, ensure_ascii=False)
    return json.dumps({"calls": [{"tool": c["tool"], "args": c["args"]} for c in action["calls"]]},
                      ensure_ascii=False)


# ------------------------------------------------------------------ parsing

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json)?")


@dataclass
class Parsed:
    kind: str  # calls | final | invalid
    calls: list[dict] = field(default_factory=list)
    final: str = ""
    status: str = ""
    error: str = ""
    strict: bool = True  # output was exactly one schema-conforming JSON object


def parse_action(text: str) -> Parsed:
    raw = text.strip()
    s = _FENCE.sub("", _THINK.sub("", raw)).strip()
    i = s.find("{")
    if i < 0:
        return Parsed("invalid", error="no JSON object found", strict=False)
    try:
        obj, end = json.JSONDecoder().raw_decode(s[i:])
    except json.JSONDecodeError as e:
        return Parsed("invalid", error=f"invalid JSON: {e.msg}", strict=False)
    strict = raw.startswith("{") and s[i + end:].strip() == ""
    if not isinstance(obj, dict):
        return Parsed("invalid", error="top-level value must be an object", strict=False)
    if "final" in obj:
        status = obj.get("status", "done")
        if status not in STATUSES:
            strict = False
            status = "done"
        if not isinstance(obj["final"], str):
            return Parsed("invalid", error="final must be a string", strict=False)
        return Parsed("final", final=obj["final"], status=status,
                      strict=strict and "status" in obj and set(obj) == {"final", "status"})
    if "tool" in obj and "calls" not in obj:  # lenient: a bare call object
        obj, strict = {"calls": [obj]}, False
    if "calls" in obj:
        calls = obj["calls"]
        if isinstance(calls, dict):
            calls, strict = [calls], False
        if not isinstance(calls, list) or not calls:
            return Parsed("invalid", error="calls must be a non-empty list", strict=False)
        out = []
        for c in calls:
            if not isinstance(c, dict) or not isinstance(c.get("tool"), str):
                return Parsed("invalid", error="each call needs a string 'tool'", strict=False)
            args = c.get("args", {})
            if not isinstance(args, dict):
                return Parsed("invalid", error="args must be an object", strict=False)
            out.append({"tool": c["tool"], "args": args})
        return Parsed("calls", calls=out, strict=strict and set(obj) == {"calls"})
    return Parsed("invalid", error="object must contain 'calls' or 'final'", strict=False)


# ------------------------------------------------------------------ approval

class Approver(Protocol):
    def __call__(self, task: Task, tool_key: str, args: dict) -> bool: ...


def task_approver(task: Task, tool_key: str, args: dict) -> bool:
    """Simulated human: follows the task's approval plan (default approve)."""
    return task.approvals.get(tool_key, True)


def make_fixed_approver(decision: bool) -> Approver:
    return lambda task, tool_key, args: decision


# ------------------------------------------------------------------ episode

@dataclass
class CallRecord:
    turn: int
    tool: str  # exposed name as emitted
    tool_key: str | None  # canonical simulator key (None if hallucinated)
    args: dict
    canonical_args: dict | None
    outcome: str  # ok | error | hallucinated | permission_denied | approval_denied | invalid_argument | limit
    result: dict
    approval: bool | None = None


@dataclass
class Trace:
    task_id: str
    model: str
    turns: list[dict] = field(default_factory=list)  # {turn, raw, parsed_kind, strict, latency_s, gen_tokens}
    calls: list[CallRecord] = field(default_factory=list)
    final: str | None = None
    status: str | None = None
    termination: str = ""  # final | max_turns | parse_errors | call_budget
    messages: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


class Episode:
    def __init__(self, task: Task, model_name: str, limits: Limits = Limits(),
                 approver: Approver = task_approver, fewshot: bool = False):
        self.task, self.limits, self.approver = task, limits, approver
        self.sim = Simulator(task.world, task.faults)
        self.messages = [{"role": "system", "content": render_system(task, fewshot)},
                         {"role": "user", "content": task.request}]
        self.trace = Trace(task.task_id, model_name)
        self.turn = 0
        self.parse_errors = 0
        self.done = False

    def step(self, text: str, latency_s: float = 0.0, gen_tokens: int = 0) -> None:
        assert not self.done
        lim = self.limits
        self.turn += 1
        p = parse_action(text)
        self.trace.turns.append({"turn": self.turn, "raw": text, "kind": p.kind, "strict": p.strict,
                                 "error": p.error, "latency_s": round(latency_s, 4), "gen_tokens": gen_tokens})
        self.messages.append({"role": "assistant", "content": text})
        if p.kind == "final":
            self.trace.final, self.trace.status = p.final, p.status
            return self._finish("final")
        if p.kind == "invalid":
            self.parse_errors += 1
            if self.parse_errors > lim.max_parse_errors:
                return self._finish("parse_errors")
            self.messages.append({"role": "user", "content": render_observations(
                [{"error": "invalid_action", "detail": p.error}])})
        else:
            results = [self._execute(c, i, len(p.calls)) for i, c in enumerate(p.calls)]
            self.messages.append({"role": "user", "content": render_observations(results)})
            if sum(r.outcome != "limit" for r in self.trace.calls) >= lim.max_total_calls:
                return self._finish("call_budget")
        if self.turn >= lim.max_turns:
            self._finish("max_turns")

    def _execute(self, call: dict, idx: int, n_in_turn: int) -> dict:
        lim, task = self.limits, self.task
        name, args = call["tool"], call["args"]
        spec = task.tool_by_name(name)
        rec = CallRecord(self.turn, name, spec.key if spec else None, args, None, "ok", {})
        executed = sum(r.outcome != "limit" for r in self.trace.calls)
        if idx >= lim.max_calls_per_turn or executed >= lim.max_total_calls:
            rec.outcome, rec.result = "limit", {"error": "call_limit", "detail": "too many calls"}
        elif spec is None:
            rec.outcome, rec.result = "hallucinated", {"error": "unknown_tool", "detail": f"no tool named {name}"}
        elif spec.required_role and spec.required_role != task.role:
            rec.outcome, rec.result = "permission_denied", {
                "error": "permission_denied", "detail": f"{name} requires role {spec.required_role}"}
        else:
            try:
                rec.canonical_args = validate_args(spec, args)
                if spec.consequential:
                    rec.approval = bool(self.approver(task, spec.key, rec.canonical_args))
                    if not rec.approval:
                        raise ToolError("approval_denied", "a human reviewer denied this action")
                t0 = time.perf_counter()
                rec.result = self.sim.execute(spec, rec.canonical_args)
                if time.perf_counter() - t0 > lim.tool_timeout_s:
                    raise ToolError("timeout", "tool exceeded time limit", retryable=True)
            except ToolError as e:
                rec.result = e.as_obs()
                rec.outcome = {"approval_denied": "approval_denied",
                               "invalid_argument": "invalid_argument"}.get(e.code, "error")
        self.trace.calls.append(rec)
        return {"tool": name, **({"result": rec.result} if rec.outcome == "ok" else rec.result)}

    def _finish(self, why: str) -> None:
        self.trace.termination = why
        self.trace.messages = self.messages
        self.done = True

    @property
    def world(self) -> dict:
        return self.sim.world


# ------------------------------------------------------------------ policies & drivers

class Policy(Protocol):
    name: str

    def generate(self, batch: list[list[dict]], episodes: list[Episode]) -> list[tuple[str, int]]:
        """Return (text, n_generated_tokens) per conversation."""
        ...


def run_batch(tasks: list[Task], policy: Policy, limits: Limits = Limits(), approver: Approver = task_approver,
              batch_size: int = 16, on_done: Callable[[Episode], None] | None = None,
              fewshot: bool = False) -> list[Episode]:
    """Advance up to `batch_size` episodes in lock-step; refill as episodes finish."""
    pending = list(tasks)
    active: list[Episode] = []
    finished: list[Episode] = []
    while pending or active:
        while pending and len(active) < batch_size:
            active.append(Episode(pending.pop(0), policy.name, limits, approver, fewshot))
        t0 = time.perf_counter()
        outs = policy.generate([e.messages for e in active], active)
        dt = (time.perf_counter() - t0) / max(1, len(active))
        for e, (text, ntok) in zip(active, outs):
            e.step(text, latency_s=dt, gen_tokens=ntok)
        for e in [e for e in active if e.done]:
            finished.append(e)
            if on_done:
                on_done(e)
        active = [e for e in active if not e.done]
    order = {t.task_id: i for i, t in enumerate(tasks)}
    return sorted(finished, key=lambda e: order[e.task.task_id])
