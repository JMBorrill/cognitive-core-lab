"""Turning verified episodes into supervised fine-tuning examples.

Each accepted episode yields one example per assistant turn:
  {"messages": <system, user, assistant/observation history>, "completion": <action JSON>}
Assistant turns (in the target and in the history) are re-serialised with
`canonical_action_text`, so teacher and oracle data share one surface format and
differ only in the decisions they encode.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .evaluate import episode_metrics
from .runtime import Episode, canonical_action_text, parse_action
from .tasks import Task

REJECT_KEYS = ("not_success", "parse_error", "hallucinated_tool", "unnecessary_calls", "dependency_unsafe",
               "invalid_args", "non_strict")


def rejection_reasons(task: Task, ep: Episode) -> list[str]:
    m = episode_metrics(task, ep)
    r = []
    if not m["success"]:
        r.append("not_success")
    if m["parse_errors"]:
        r.append("parse_error")
    if m["hallucinated_tools"]:
        r.append("hallucinated_tool")
    if m["unnecessary_calls"]:
        r.append("unnecessary_calls")
    if not m["dependency_safe"]:
        r.append("dependency_unsafe")
    if m["invalid_args"]:
        r.append("invalid_args")
    return r


def _canonical(text: str) -> str:
    p = parse_action(text)
    if p.kind == "final":
        return canonical_action_text({"final": p.final, "status": p.status})
    if p.kind == "calls":
        return canonical_action_text({"calls": p.calls})
    raise ValueError("cannot canonicalise an invalid action")


def episode_to_examples(ep: Episode, source: str) -> list[dict]:
    msgs = ep.trace.messages or ep.messages
    out, hist = [], []
    for m in msgs:
        if m["role"] == "assistant":
            c = _canonical(m["content"])
            out.append({"messages": [dict(x) for x in hist], "completion": c,
                        "meta": {"task_id": ep.task.task_id, "template": ep.task.template,
                                 "family": ep.task.family, "source": source, "turn": len(out) + 1}})
            hist.append({"role": "assistant", "content": c})
        else:
            hist.append(dict(m))
    return out


def write_jsonl(path: str | Path, rows) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in open(path)]


def save_tasks(path, tasks: list[Task]) -> str:
    return write_jsonl(path, (t.to_dict() for t in tasks))


def load_tasks(path) -> list[Task]:
    return [Task.from_dict(d) for d in read_jsonl(path)]
