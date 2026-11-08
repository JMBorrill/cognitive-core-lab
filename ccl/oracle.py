"""Programmatic oracle policy: the source of independently generated supervision (config C).

It reads only the task's gold DAG and the observations returned by the simulator,
issuing every ready call in one turn (maximal safe parallelism), retrying
retryable failures once, switching to declared fallbacks, and stopping on
denial / not-found. Its trajectories are validated exactly like the teacher's.
"""

from __future__ import annotations

from .runtime import Episode, canonical_action_text
from .tasks import GoldCall, gold_args_match


def _concrete(gc: GoldCall) -> dict:
    out = {}
    for k, v in gc.args.items():
        if k in gc.display:
            out[k] = gc.display[k]
        elif isinstance(v, list) and gc.match.get(k) == "contains":
            out[k] = ". ".join(str(x) for x in v).capitalize() + "."
        else:
            out[k] = v
    return out


class OraclePolicy:
    name = "oracle"

    def generate(self, batch, episodes):
        return [(self.act(e), 0) for e in episodes]

    def act(self, ep: Episode) -> str:
        task = ep.task
        status: dict[str, str] = {}  # gold id -> ok | retry | fallback | denied | failed
        tried: dict[str, int] = {}
        for rec in ep.trace.calls:
            for gc in task.gold:
                key_ok = rec.tool_key in (gc.tool, gc.fallback)
                if not key_ok or rec.canonical_args is None or not gold_args_match(gc, rec.canonical_args):
                    continue
                tried[gc.id] = tried.get(gc.id, 0) + 1
                err = rec.result.get("error") if rec.outcome != "ok" else None
                if err is None:
                    status[gc.id] = "ok"
                elif err == "approval_denied":
                    status[gc.id] = "denied"
                elif rec.result.get("retryable") and tried[gc.id] < 2:
                    status[gc.id] = "retry"
                elif err == "service_unavailable" and gc.fallback and rec.tool_key == gc.tool:
                    status[gc.id] = "fallback"
                else:
                    status[gc.id] = "failed"
                break
        if any(v == "denied" for v in status.values()):
            return canonical_action_text({"final": task.reference_answer, "status": "blocked"})
        if any(v == "failed" for v in status.values()):
            return canonical_action_text({"final": task.reference_answer, "status": "failed"})
        if task.expected_status == "refused":
            return canonical_action_text({"final": task.reference_answer, "status": "refused"})
        ready = []
        for gc in task.gold:
            st = status.get(gc.id)
            if st == "ok":
                continue
            if all(status.get(d) == "ok" for d in gc.deps):
                tool = gc.fallback if st == "fallback" else gc.tool
                spec = task.exposed(tool)
                args = {spec.exposed_param(k): v for k, v in _concrete(gc).items()}
                ready.append({"tool": spec.name, "args": args})
        if ready:
            return canonical_action_text({"calls": ready})
        return canonical_action_text({"final": task.reference_answer, "status": task.expected_status})
