"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ApiError, LiveHealth, LiveRunResponse, LiveTaskSummary, ShowcaseTask } from "@/lib/types";
import type { ConfigInfo } from "@/lib/configs";
import { SPLIT_ORDER, SPLIT_SHORT, humanize, isNum } from "@/lib/configs";
import TraceView from "./TraceView";
import { Catalog } from "./TaskPanel";
import { LiveBadge } from "./ModeBadges";

type Health = { state: "checking" } | { state: "ok"; data: LiveHealth } | { state: "down"; error: ApiError };

type Spec = {
  name?: string;
  description?: string;
  params?: { name: string; type: string; required?: boolean }[];
  consequential?: boolean;
  required_role?: string | null;
};

function specToSignature(s: unknown): string {
  const t = s as Spec;
  const params = (t.params ?? []).map((p) => `${p.name}${p.required === false ? "?" : ""}: ${p.type}`).join(", ");
  const tags = `${t.consequential ? " [approval]" : ""}${t.required_role ? ` [role: ${t.required_role}]` : ""}`;
  return `${t.name ?? "?"}(${params})${tags}${t.description ? ` - ${t.description}` : ""}`;
}

async function readError(res: Response): Promise<ApiError> {
  try {
    const j = await res.json();
    return { error: j.error ?? `http_${res.status}`, message: j.message ?? res.statusText, hint: j.hint, server_url: j.server_url, detail: j.detail };
  } catch {
    return { error: `http_${res.status}`, message: res.statusText || "Request failed" };
  }
}

function ServerDown({ error, onRetry }: { error: ApiError; onRetry: () => void }) {
  return (
    <div className="card stack" role="alert">
      <div className="row">
        <span className="chip chip-critical">
          <span aria-hidden="true">✕</span> live server not reachable
        </span>
        <span className="muted small">{error.server_url ?? ""}</span>
      </div>
      <p style={{ margin: 0 }}>
        {error.message} Live mode runs the selected configuration on this machine through the Python inference server;
        it never substitutes recorded traces. Switch to <strong>Replay</strong> to browse recorded evaluation traces.
      </p>
      <div>
        <div className="tiny muted" style={{ marginBottom: 4 }}>
          Start it from the repository root (MLX on Apple silicon; one model is resident at a time):
        </div>
        <pre className="codeblock">{`cd cognitive-core-lab
source .venv/bin/activate
python scripts/serve.py --port 8765

# the web app proxies to $CCL_SERVER_URL (default http://127.0.0.1:8765)`}</pre>
      </div>
      <div>
        <button className="btn" onClick={onRetry}>
          Check again
        </button>
      </div>
    </div>
  );
}

export default function LivePanel({ task, configs }: { task: ShowcaseTask; configs: ConfigInfo[] }) {
  const [health, setHealth] = useState<Health>({ state: "checking" });
  const [config, setConfig] = useState("");
  const [source, setSource] = useState<"selected" | "any">("selected");
  const [anySplit, setAnySplit] = useState(task.split);
  const [anyTasks, setAnyTasks] = useState<LiveTaskSummary[] | null>(null);
  const [anyIndex, setAnyIndex] = useState(0);
  const [custom, setCustom] = useState("");
  const [approval, setApproval] = useState<"task" | "approve" | "deny">("task");
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState<LiveRunResponse | null>(null);
  const [runError, setRunError] = useState<ApiError | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const checkHealth = useCallback(async () => {
    setHealth({ state: "checking" });
    try {
      const res = await fetch("/api/run", { cache: "no-store" });
      if (!res.ok) return setHealth({ state: "down", error: await readError(res) });
      const data = (await res.json()) as LiveHealth;
      setHealth({ state: "ok", data });
      setConfig((cur) => {
        if (cur && data.configs[cur]?.available) return cur;
        const first = Object.entries(data.configs).find(([, v]) => v.available);
        return first ? first[0] : "";
      });
    } catch {
      setHealth({
        state: "down",
        error: { error: "network", message: "Could not reach the web app's /api/run route." },
      });
    }
  }, []);

  useEffect(() => {
    checkHealth();
    return () => abortRef.current?.abort();
  }, [checkHealth]);

  // Task list for "any evaluation task" (served by the Python server, not the showcase sample).
  useEffect(() => {
    if (source !== "any" || health.state !== "ok") return;
    let cancelled = false;
    setAnyTasks(null);
    fetch(`/api/run?tasks=${encodeURIComponent(anySplit)}`, { cache: "no-store" })
      .then(async (r) => (r.ok ? ((await r.json()) as LiveTaskSummary[]) : Promise.reject(await readError(r))))
      .then((list) => {
        if (cancelled) return;
        setAnyTasks(list);
        setAnyIndex((i) => (i < list.length ? i : 0));
      })
      .catch((e: ApiError) => !cancelled && setRunError(e));
    return () => {
      cancelled = true;
    };
  }, [source, anySplit, health.state]);

  useEffect(() => {
    if (!running) return;
    const t0 = Date.now();
    const id = window.setInterval(() => setElapsed((Date.now() - t0) / 1000), 200);
    return () => window.clearInterval(id);
  }, [running]);

  async function run() {
    const split = source === "selected" ? task.split : anySplit;
    const index = source === "selected" ? task.index : anyIndex;
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setRunning(true);
    setElapsed(0);
    setRunError(null);
    setResult(null);
    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config, split, index, approval, request: custom.trim() || undefined }),
        signal: ctrl.signal,
      });
      if (!res.ok) {
        const e = await readError(res);
        setRunError(e);
        if (e.error === "live_server_unreachable") setHealth({ state: "down", error: e });
      } else {
        setResult((await res.json()) as LiveRunResponse);
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") setRunError({ error: "network", message: "The request to /api/run failed." });
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  }

  if (health.state === "checking") {
    return (
      <div className="card" aria-live="polite">
        <span className="muted">Checking for the local inference server…</span>
      </div>
    );
  }
  if (health.state === "down") return <ServerDown error={health.error} onRetry={checkHealth} />;

  const serverCfgs = health.data.configs;
  const label = (k: string) => serverCfgs[k]?.label ?? configs.find((c) => c.key === k)?.label ?? k;
  const anyTask = anyTasks?.find((t) => t.index === anyIndex);

  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="card stack">
        <div className="row">
          <LiveBadge />
          <span className="chip chip-good">
            <span aria-hidden="true">✓</span> server connected
          </span>
        </div>
        <div className="controls">
          <div className="field">
            <label htmlFor="l-cfg">Configuration</label>
            <select id="l-cfg" value={config} onChange={(e) => setConfig(e.target.value)} disabled={running}>
              {Object.entries(serverCfgs).map(([k, v]) => (
                <option key={k} value={k} disabled={!v.available}>
                  {v.label}
                  {v.available ? "" : " — weights/adapter not present"}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="l-src">Task</label>
            <select
              id="l-src"
              value={source}
              onChange={(e) => setSource(e.target.value as "selected" | "any")}
              disabled={running}
            >
              <option value="selected">Selected showcase task ({task.key})</option>
              <option value="any">Any evaluation task…</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="l-appr">Approval policy</label>
            <select
              id="l-appr"
              value={approval}
              onChange={(e) => setApproval(e.target.value as "task" | "approve" | "deny")}
              disabled={running}
            >
              <option value="task">Follow the task&apos;s approval plan</option>
              <option value="approve">Approver always approves</option>
              <option value="deny">Approver always denies</option>
            </select>
          </div>
        </div>
        {source === "any" ? (
          <div className="controls">
            <div className="field">
              <label htmlFor="l-split">Split</label>
              <select id="l-split" value={anySplit} onChange={(e) => setAnySplit(e.target.value)} disabled={running}>
                {SPLIT_ORDER.map((s) => (
                  <option key={s} value={s}>
                    {SPLIT_SHORT[s]} ({s})
                  </option>
                ))}
              </select>
            </div>
            <div className="field" style={{ gridColumn: "1 / -1" }}>
              <label htmlFor="l-task">Task index</label>
              <select
                id="l-task"
                value={anyIndex}
                onChange={(e) => setAnyIndex(Number(e.target.value))}
                disabled={running || !anyTasks}
              >
                {anyTasks ? (
                  anyTasks.map((t) => (
                    <option key={t.index} value={t.index}>
                      {t.index} · {t.family} · {t.request.slice(0, 70)}
                    </option>
                  ))
                ) : (
                  <option>Loading tasks from the server…</option>
                )}
              </select>
            </div>
          </div>
        ) : null}
        {source === "any" && anyTask ? (
          <blockquote className="request">{anyTask.request}</blockquote>
        ) : null}
        <div className="field">
          <label htmlFor="l-req">Custom request (optional)</label>
          <textarea
            id="l-req"
            rows={2}
            maxLength={500}
            value={custom}
            disabled={running}
            placeholder="Replace the task's request with your own wording. The task's tool catalogue and simulated world are kept."
            onChange={(e) => setCustom(e.target.value)}
          />
          <span className="tiny muted">
            {custom.length}/500 · A custom request has no gold workflow, so validator checks are not computed.
          </span>
        </div>
        <div className="row">
          <button className="btn btn-primary" onClick={run} disabled={running || !config}>
            {running ? "Running…" : "Run live episode"}
          </button>
          {running ? (
            <>
              <button className="btn" onClick={() => abortRef.current?.abort()}>
                Cancel
              </button>
              <span className="muted small num" aria-live="polite">
                {label(config)} · {elapsed.toFixed(1)} s elapsed
              </span>
            </>
          ) : null}
          {!config ? <span className="muted small">No configuration is available on the server.</span> : null}
        </div>
        {running ? (
          <p className="tiny muted" style={{ margin: 0 }}>
            The first run of a configuration loads its weights; requests are serialised on the server. Cancelling stops
            waiting here but the server finishes the episode.
          </p>
        ) : null}
      </div>

      {runError ? (
        <div className="note note-critical" role="alert">
          <strong>Live run failed:</strong> {runError.message}
          {runError.hint ? <div className="small">{runError.hint}</div> : null}
        </div>
      ) : null}

      {result ? <LiveResult result={result} /> : null}
    </div>
  );
}

function LiveResult({ result }: { result: LiveRunResponse }) {
  const metrics = Object.entries(result.metrics ?? {}).filter(
    ([k, v]) => isNum(v) && !k.startsWith("check_") && k !== "success",
  ) as [string, number][];
  return (
    <section className="card stack" aria-label="Live result">
      <TraceView
        trace={result.trace}
        success={result.validation?.success}
        checks={result.validation?.checks}
        title={<h3>{result.label}</h3>}
        badge={<LiveBadge detail={`${result.wall_time_s.toFixed(1)} s wall time`} />}
      />
      <div className="grid grid-2" style={{ alignItems: "start" }}>
        <div>
          <h4 style={{ marginBottom: 6 }}>Task run</h4>
          <p className="small" style={{ marginBottom: 6 }}>
            <span className="mono">{result.task.task_id}</span> · {result.task.family} · role {result.task.role} ·
            expected status {result.task.expected_status}
          </p>
          {result.custom_request ? (
            <div className="note note-warn small" style={{ marginBottom: 8 }}>
              Custom request — the gold workflow was cleared, so this run is not scored.
            </div>
          ) : null}
          <blockquote className="request">{result.task.request}</blockquote>
          <details style={{ marginTop: 8 }}>
            <summary>Tool catalogue ({result.task.catalog.length})</summary>
            <div style={{ marginTop: 6 }}>
              <Catalog catalog={result.task.catalog.map(specToSignature)} />
            </div>
          </details>
        </div>
        {metrics.length ? (
          <div>
            <h4 style={{ marginBottom: 6 }}>Episode metrics (from the evaluator)</h4>
            <dl className="kv">
              {metrics.map(([k, v]) => (
                <div key={k} style={{ display: "contents" }}>
                  <dt>{humanize(k)}</dt>
                  <dd className="num">{Number.isInteger(v) ? v : v.toFixed(3)}</dd>
                </div>
              ))}
            </dl>
          </div>
        ) : null}
      </div>
    </section>
  );
}
