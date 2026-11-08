"""Interactive agent runtime: run any configuration on a stored task or a free-form request.

    python scripts/agent.py --config B --task-split eval_ood --index 3
    python scripts/agent.py --config C --request "Email Dana Novak that the venue is booked" --world-from eval_id:0

Consequential tools ([approval]) prompt a human on the terminal unless --approve/--deny is given.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccl.data import load_tasks  # noqa: E402
from ccl.runtime import Episode, Limits, make_fixed_approver  # noqa: E402

CONFIGS = json.loads((Path(__file__).resolve().parents[1] / "configs" / "models.json").read_text())


def terminal_approver(task, tool_key, args):
    print(f"\n[approval] {tool_key} {json.dumps(args)}")
    return input("approve? [y/N] ").strip().lower() == "y"


def load_policy(config: str):
    c = CONFIGS[config]
    if c.get("policy") == "oracle":
        from ccl.oracle import OraclePolicy
        return OraclePolicy()
    from ccl.models import MLXPolicy
    return MLXPolicy(c["model"], adapter_path=c.get("adapter"), name=config)


def run(policy, task, approver, limits=Limits(), verbose=True, fewshot=False):
    ep = Episode(task, policy.name, limits, approver, fewshot=fewshot)
    while not ep.done:
        text, ntok = policy.generate([ep.messages], [ep])[0]
        n_before = len(ep.trace.calls)
        ep.step(text, gen_tokens=ntok)
        if verbose:
            print(f"\n[model] {text}")
            for c in ep.trace.calls[n_before:]:
                print(f"  -> {c.tool}: {c.outcome} {json.dumps(c.result)[:200]}")
    return ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="A")
    ap.add_argument("--task-split", default="eval_id")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--request", help="free-form request; uses the world/catalogue of the selected task")
    ap.add_argument("--approve", action="store_true")
    ap.add_argument("--deny", action="store_true")
    a = ap.parse_args()
    task = load_tasks(f"data/v1/tasks/{a.task_split}.jsonl")[a.index]
    if a.request:
        task = copy.deepcopy(task)
        task.request, task.answer_facts, task.gold = a.request, [], []
    approver = make_fixed_approver(True) if a.approve else make_fixed_approver(False) if a.deny else terminal_approver
    print(f"[task] {task.task_id}  role={task.role}\n[user] {task.request}")
    ep = run(load_policy(a.config), task, approver)
    print(f"\n[end] termination={ep.trace.termination} status={ep.trace.status}")
    if not a.request:
        from ccl.evaluate import validate
        print("[validate]", json.dumps(validate(task, ep)))


if __name__ == "__main__":
    main()
