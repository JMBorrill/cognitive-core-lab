import type { ReactNode } from "react";
import type { Trace, TraceCall } from "@/lib/types";
import { FAILURE_HELP, humanize, isNum } from "@/lib/configs";

const OUTCOME: Record<string, { label: string; cls: string; icon: string }> = {
  ok: { label: "ok", cls: "chip-good", icon: "✓" },
  error: { label: "tool error", cls: "chip-warn", icon: "!" },
  hallucinated: { label: "hallucinated tool", cls: "chip-critical", icon: "✕" },
  permission_denied: { label: "permission denied", cls: "chip-critical", icon: "✕" },
  approval_denied: { label: "approval denied", cls: "chip-serious", icon: "⊘" },
  invalid_argument: { label: "invalid argument", cls: "chip-serious", icon: "✕" },
  limit: { label: "call limit", cls: "", icon: "⊘" },
};

const CHECK_LABEL: Record<string, string> = {
  terminated: "terminated with a final action",
  status: "correct final status",
  facts: "answer contains required facts",
  required_calls: "required calls made",
  effects: "side effects match",
  no_permission_violation: "no permission violation",
  no_retry_after_denial: "no retry after denial",
};

const STATUS_CLS: Record<string, string> = {
  done: "chip-good",
  refused: "chip-serious",
  blocked: "chip-warn",
  failed: "chip-critical",
};

export function OutcomeChip({ outcome }: { outcome: string }) {
  const o = OUTCOME[outcome] ?? { label: humanize(outcome), cls: "", icon: "·" };
  return (
    <span className={`chip ${o.cls}`}>
      <span aria-hidden="true">{o.icon}</span>
      {o.label}
    </span>
  );
}

export function StatusChip({ status }: { status: string | null }) {
  if (!status) return <span className="chip">no status</span>;
  return <span className={`chip ${STATUS_CLS[status] ?? ""}`}>status: {status}</span>;
}

export function SuccessChip({ success }: { success: boolean }) {
  return success ? (
    <span className="chip chip-good">
      <span aria-hidden="true">✓</span> success
    </span>
  ) : (
    <span className="chip chip-critical">
      <span aria-hidden="true">✕</span> failure
    </span>
  );
}

function json(x: unknown): string {
  if (x === undefined) return "";
  const s = JSON.stringify(x);
  if (s === undefined) return String(x);
  return s.length > 90 ? JSON.stringify(x, null, 2) : s;
}

function CallCard({ c }: { c: TraceCall }) {
  const canonicalDiffers =
    c.canonical_args !== undefined && c.canonical_args !== null && json(c.canonical_args) !== json(c.args);
  return (
    <div className="call">
      <div className="call-head">
        <span className="tool">{c.tool}</span>
        {c.tool_key && c.tool_key !== c.tool ? <span className="muted tiny mono">→ {c.tool_key}</span> : null}
        <span className="spacer" />
        {c.approval === true ? <span className="chip chip-outline">approver: approved</span> : null}
        {c.approval === false ? <span className="chip chip-outline">approver: denied</span> : null}
        <OutcomeChip outcome={c.outcome} />
      </div>
      <dl className="call-body">
        <dt>args</dt>
        <dd>{json(c.args)}</dd>
        {canonicalDiffers ? (
          <>
            <dt>canonical</dt>
            <dd>{json(c.canonical_args)}</dd>
          </>
        ) : null}
        <dt>{c.outcome === "ok" ? "result" : "observation"}</dt>
        <dd>{json(c.result)}</dd>
      </dl>
    </div>
  );
}

type Props = {
  trace: Trace;
  success?: boolean;
  checks?: Record<string, boolean>;
  failureMode?: string | null;
  badge?: ReactNode;
  title?: ReactNode;
};

export default function TraceView({ trace, success, checks, failureMode, badge, title }: Props) {
  const callsByTurn = new Map<number, TraceCall[]>();
  for (const c of trace.calls) {
    const arr = callsByTurn.get(c.turn) ?? [];
    arr.push(c);
    callsByTurn.set(c.turn, arr);
  }
  const genTokens = trace.turns.reduce((s, t) => s + (t.gen_tokens || 0), 0);

  return (
    <div className="trace-panel">
      <div className="trace-head">
        {title}
        {badge}
        <div className="row">
          {success !== undefined ? <SuccessChip success={success} /> : null}
          <StatusChip status={trace.status} />
          <span className="chip">
            {trace.turns.length} turn{trace.turns.length === 1 ? "" : "s"} · {trace.calls.length} call
            {trace.calls.length === 1 ? "" : "s"}
          </span>
          {trace.termination && trace.termination !== "final" ? (
            <span className="chip chip-warn">ended: {humanize(trace.termination)}</span>
          ) : null}
          {genTokens > 0 ? <span className="chip">{genTokens} generated tokens</span> : null}
          {failureMode ? (
            <span className="chip chip-critical" title={FAILURE_HELP[failureMode]}>
              failure mode: {humanize(failureMode)}
            </span>
          ) : null}
        </div>
      </div>

      <ol className="timeline" aria-label="Turn-by-turn trace">
        {trace.turns.map((t) => {
          const calls = callsByTurn.get(t.turn) ?? [];
          return (
            <li key={t.turn} className={t.kind === "final" ? "final" : undefined}>
              <div className="turn-head">
                <span className="t">Turn {t.turn}</span>
                <span className="chip">
                  {t.kind === "calls" ? "tool calls" : t.kind === "final" ? "final answer" : "unparseable"}
                </span>
                {t.strict ? (
                  <span className="chip chip-outline">strict JSON</span>
                ) : (
                  <span className="chip chip-warn">non-strict output</span>
                )}
                {t.gen_tokens > 0 ? <span className="muted tiny">{t.gen_tokens} tok</span> : null}
                {isNum(t.latency_s) && t.latency_s > 0 ? (
                  <span className="muted tiny">{t.latency_s.toFixed(2)} s</span>
                ) : null}
              </div>
              <div className="tiny muted" style={{ marginBottom: 3 }}>
                Model output (raw)
              </div>
              <pre className="raw">{t.raw || <span className="muted">(empty)</span>}</pre>
              {t.error ? (
                <div className="note note-critical small" style={{ marginTop: 6 }}>
                  Parse error: {t.error}
                </div>
              ) : null}
              {calls.length ? (
                <div className="calls">
                  <div className="parallel-tag">
                    {calls.length > 1
                      ? `${calls.length} calls issued together (executed in parallel)`
                      : "Parsed call → runtime observation"}
                  </div>
                  {calls.map((c, i) => (
                    <CallCard key={i} c={c} />
                  ))}
                </div>
              ) : null}
            </li>
          );
        })}
      </ol>

      <div className="stack" style={{ marginTop: 4 }}>
        <div>
          <h4 style={{ marginBottom: 6 }}>Final answer</h4>
          {trace.final !== null && trace.final !== undefined ? (
            <div className="final-box">
              <div className="row" style={{ marginBottom: 6 }}>
                <StatusChip status={trace.status} />
              </div>
              {trace.final || <span className="muted">(empty answer)</span>}
            </div>
          ) : (
            <p className="muted small">
              No final action — the episode ended by <code>{trace.termination || "unknown"}</code>.
            </p>
          )}
        </div>
        {checks && Object.keys(checks).length ? (
          <div>
            <h4 style={{ marginBottom: 6 }}>Validator checks</h4>
            <ul className="checks" style={{ listStyle: "none", padding: 0, margin: 0 }}>
              {Object.entries(checks).map(([k, v]) => (
                <li key={k}>
                  <span className={`chip ${v ? "chip-good" : "chip-critical"}`}>
                    <span aria-hidden="true">{v ? "✓" : "✕"}</span>
                    <span className="sr-only">{v ? "passed:" : "failed:"}</span>
                    {CHECK_LABEL[k] ?? humanize(k)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        {trace.messages && trace.messages.length ? (
          <details>
            <summary>Full conversation as seen by the model ({trace.messages.length} messages)</summary>
            <div className="stack" style={{ marginTop: 8 }}>
              {trace.messages.map((m, i) => (
                <div key={i}>
                  <div className="tiny muted" style={{ textTransform: "uppercase", letterSpacing: "0.05em" }}>
                    {m.role}
                  </div>
                  <pre className="raw">{m.content}</pre>
                </div>
              ))}
            </div>
          </details>
        ) : null}
      </div>
    </div>
  );
}
