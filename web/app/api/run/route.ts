// Proxy to the local Python inference server (scripts/serve.py).
//
//   GET  /api/run                 -> server /health
//   GET  /api/run?tasks=<split>   -> server /tasks?split=<split>
//   POST /api/run                 -> server /run   (body validated here first)
//
// Errors are always JSON: {error, message, hint?, server_url?}. When the Python server is not
// reachable the route answers 503 — the UI then shows how to start it and never falls back to replay.

import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const SERVER_URL = (process.env.CCL_SERVER_URL ?? "http://127.0.0.1:8765").replace(/\/+$/, "");
const SPLITS = new Set(["eval_id", "eval_ood", "eval_unseen", "eval_renamed"]);
const APPROVAL = new Set(["task", "approve", "deny"]);
const HINT = "Start the local inference server from the repository root:  python scripts/serve.py --port 8765";
// Stay below undici's default 300 s headers timeout so a slow run reports a clear 504.
const RUN_TIMEOUT_MS = 290 * 1000;

function err(status: number, error: string, message: string, extra: Record<string, unknown> = {}) {
  return NextResponse.json({ error, message, server_url: SERVER_URL, ...extra }, { status });
}

async function upstream(pathname: string, init: RequestInit, timeoutMs: number) {
  let res: Response;
  try {
    res = await fetch(`${SERVER_URL}${pathname}`, {
      ...init,
      cache: "no-store",
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (e) {
    const name = e instanceof Error ? e.name : "";
    const code = String((e as { cause?: { code?: unknown } })?.cause?.code ?? "");
    if (name === "TimeoutError" || name === "AbortError" || code.includes("TIMEOUT")) {
      return err(504, "live_server_timeout", `The local inference server did not answer within ${timeoutMs / 1000} s.`, {
        hint: "Long runs (e.g. the 4B teacher) can be slow; check the server terminal for progress.",
      });
    }
    if (code === "UND_ERR_SOCKET" || code === "ECONNRESET") {
      return err(502, "live_server_error", "The inference server closed the connection mid-request.", {
        hint: "Check the terminal running scripts/serve.py for a traceback.",
      });
    }
    return err(503, "live_server_unreachable", "Live mode needs the local Python inference server, which is not reachable.", {
      hint: HINT,
      detail: code || undefined,
    });
  }
  const text = await res.text();
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return err(502, "bad_upstream_response", "The inference server returned a non-JSON response.", {
      detail: text.slice(0, 500),
    });
  }
  if (!res.ok) {
    const detail = (body as { error?: unknown })?.error ?? body;
    // 4xx from serve.py means the request was rejected (unknown config, missing adapter, bad index).
    const status = res.status >= 400 && res.status < 500 ? res.status : 502;
    return err(status, "upstream_error", typeof detail === "string" ? detail : "The inference server rejected the request.", {
      detail,
    });
  }
  return NextResponse.json(body, { status: 200, headers: { "Cache-Control": "no-store" } });
}

export async function GET(req: Request) {
  const split = new URL(req.url).searchParams.get("tasks");
  if (split !== null) {
    if (!SPLITS.has(split)) return err(400, "bad_request", `Unknown split "${split}".`);
    return upstream(`/tasks?split=${encodeURIComponent(split)}`, { method: "GET" }, 15_000);
  }
  return upstream("/health", { method: "GET" }, 5_000);
}

export async function POST(req: Request) {
  let raw: unknown;
  try {
    raw = await req.json();
  } catch {
    return err(400, "bad_request", "Request body must be JSON.");
  }
  if (typeof raw !== "object" || raw === null) return err(400, "bad_request", "Request body must be a JSON object.");
  const b = raw as Record<string, unknown>;
  const config = b.config;
  const split = b.split ?? "eval_id";
  const index = b.index ?? 0;
  const approval = b.approval ?? "task";
  const request = b.request;

  if (typeof config !== "string" || !/^[A-Za-z0-9_]{1,40}$/.test(config))
    return err(400, "bad_request", "`config` must be a configuration key such as \"B\".");
  if (typeof split !== "string" || !SPLITS.has(split)) return err(400, "bad_request", "`split` is not a known evaluation split.");
  if (typeof index !== "number" || !Number.isInteger(index) || index < 0 || index > 100_000)
    return err(400, "bad_request", "`index` must be a non-negative integer.");
  if (typeof approval !== "string" || !APPROVAL.has(approval))
    return err(400, "bad_request", "`approval` must be one of task | approve | deny.");
  if (request !== undefined && request !== null && (typeof request !== "string" || request.length > 500))
    return err(400, "bad_request", "`request` must be a string of at most 500 characters.");

  const payload: Record<string, unknown> = { config, split, index, approval };
  if (typeof request === "string" && request.trim()) payload.request = request.trim();

  return upstream(
    "/run",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) },
    RUN_TIMEOUT_MS,
  );
}
