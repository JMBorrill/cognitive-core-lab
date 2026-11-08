"""Local inference server for the showcase's live mode (stdlib only).

    python scripts/serve.py --port 8765

GET  /health                      -> configs and which adapters are present
GET  /tasks?split=eval_ood        -> task summaries for the picker
POST /run {"config": "B", "split": "eval_ood", "index": 3, "request": optional,
           "approval": "task" | "approve" | "deny"}
     -> {"mode": "live", "trace": ..., "validation": ..., "metrics": ..., "task": ...}

Binds to 127.0.0.1 only. One model is resident at a time; requests are serialised.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccl.data import load_tasks  # noqa: E402
from ccl.evaluate import episode_metrics, validate  # noqa: E402
from ccl.runtime import Episode, Limits, make_fixed_approver, task_approver  # noqa: E402
from ccl.tasks import spec_to_dict  # noqa: E402

CONFIGS = json.loads((ROOT / "configs" / "models.json").read_text())
SPLITS = ["eval_id", "eval_ood", "eval_unseen", "eval_renamed"]
_lock = threading.Lock()
_policy = {"name": None, "obj": None}
_tasks: dict[str, list] = {}


def tasks(split):
    if split not in SPLITS:
        raise ValueError("unknown split")
    if split not in _tasks:
        _tasks[split] = load_tasks(ROOT / "data" / "v1" / "tasks" / f"{split}.jsonl")
    return _tasks[split]


def available(cfg: dict) -> bool:
    if cfg.get("policy") == "oracle":
        return True
    ok = (ROOT / cfg["model"]).exists()
    return ok and (cfg.get("adapter") is None or (ROOT / cfg["adapter"] / "adapters.safetensors").exists())


def policy_for(name: str):
    if _policy["name"] != name:
        cfg = CONFIGS[name]
        _policy["obj"] = None
        if cfg.get("policy") == "oracle":
            from ccl.oracle import OraclePolicy
            _policy["obj"] = OraclePolicy()
        else:
            from ccl.models import MLXPolicy
            _policy["obj"] = MLXPolicy(str(ROOT / cfg["model"]),
                                       adapter_path=str(ROOT / cfg["adapter"]) if cfg.get("adapter") else None,
                                       name=name)
        _policy["name"] = name
    return _policy["obj"]


def task_summary(t, i):
    return {"index": i, "task_id": t.task_id, "family": t.family, "template": t.template, "role": t.role,
            "request": t.request, "depth": t.depth}


def run(body: dict) -> dict:
    name = body.get("config", "A")
    if name not in CONFIGS or not available(CONFIGS[name]):
        raise ValueError(f"config {name} not available locally")
    task = tasks(body.get("split", "eval_id"))[int(body.get("index", 0))]
    custom = bool(body.get("request"))
    if custom:
        task = copy.deepcopy(task)
        task.request = str(body["request"])[:500]
        task.gold, task.answer_facts = [], []
    approver = {"approve": make_fixed_approver(True), "deny": make_fixed_approver(False)}.get(
        body.get("approval", "task"), task_approver)
    with _lock:
        pol = policy_for(name)
        ep = Episode(task, name, Limits(), approver, fewshot=bool(CONFIGS[name].get("fewshot")))
        t0 = time.perf_counter()
        while not ep.done:
            ts = time.perf_counter()
            text, ntok = pol.generate([ep.messages], [ep])[0]
            ep.step(text, latency_s=time.perf_counter() - ts, gen_tokens=ntok)
        wall = time.perf_counter() - t0
    out = {"mode": "live", "config": name, "label": CONFIGS[name]["label"], "wall_time_s": wall,
           "custom_request": custom, "trace": ep.trace.to_dict(),
           "task": {**task_summary(task, int(body.get("index", 0))),
                    "catalog": [spec_to_dict(s) for s in task.catalog], "expected_status": task.expected_status}}
    if not custom:
        out["validation"] = validate(task, ep)
        out["metrics"] = episode_metrics(task, ep)
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/health":
                return self._send(200, {"ok": True, "configs": {k: {"label": v["label"], "available": available(v)}
                                                                for k, v in CONFIGS.items()}})
            if u.path == "/tasks":
                split = parse_qs(u.query).get("split", ["eval_id"])[0]
                return self._send(200, [task_summary(t, i) for i, t in enumerate(tasks(split))])
            self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": str(e)})

    def do_POST(self):
        if self.path != "/run":
            return self._send(404, {"error": "not found"})
        try:
            n = min(int(self.headers.get("Content-Length", 0)), 100_000)
            self._send(200, run(json.loads(self.rfile.read(n) or b"{}")))
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": str(e)})

    def log_message(self, fmt, *args):
        sys.stderr.write("[serve] " + fmt % args + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    print(f"[serve] http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
