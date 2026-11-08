"use client";

import { useEffect, useMemo, useState } from "react";
import type { ReplayTrace, ShowcaseTask } from "@/lib/types";
import type { ConfigInfo } from "@/lib/configs";
import { FAMILY_HELP, SPLIT_SHORT, SPLIT_ORDER } from "@/lib/configs";
import TaskPanel from "./TaskPanel";
import TraceView from "./TraceView";
import LivePanel from "./LivePanel";
import { ReplayBadge } from "./ModeBadges";

type Props = {
  tasks: ShowcaseTask[];
  traces: Record<string, Record<string, ReplayTrace>>;
  configs: ConfigInfo[];
  initial: { task?: string; config?: string; compare?: string; mode?: string };
};

function truncate(s: string, n: number) {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

export default function TraceExplorer({ tasks, traces, configs, initial }: Props) {
  const withTraces = configs.filter((c) => traces[c.key] && Object.keys(traces[c.key]).length);
  const cfgKeys = new Set(configs.map((c) => c.key));
  const taskKeys = new Set(tasks.map((t) => t.key));

  const [mode, setMode] = useState<"replay" | "live">(initial.mode === "live" ? "live" : "replay");
  const [split, setSplit] = useState("all");
  const [family, setFamily] = useState("all");
  const [taskKey, setTaskKey] = useState(
    initial.task && taskKeys.has(initial.task) ? initial.task : (tasks[0]?.key ?? ""),
  );
  const [cfgA, setCfgA] = useState(
    initial.config && cfgKeys.has(initial.config) ? initial.config : (withTraces[0]?.key ?? configs[0]?.key ?? ""),
  );
  const [cfgB, setCfgB] = useState(
    initial.compare && cfgKeys.has(initial.compare) && initial.compare !== cfgA ? initial.compare : "",
  );
  const chooseA = (k: string) => {
    setCfgA(k);
    if (k === cfgB) setCfgB("");
  };

  const splits = SPLIT_ORDER.filter((s) => tasks.some((t) => t.split === s));
  const families = useMemo(
    () => [...new Set(tasks.filter((t) => split === "all" || t.split === split).map((t) => t.family))].sort(),
    [tasks, split],
  );
  const filtered = useMemo(
    () => tasks.filter((t) => (split === "all" || t.split === split) && (family === "all" || t.family === family)),
    [tasks, split, family],
  );

  // Keep the selected task inside the current filter.
  useEffect(() => {
    if (filtered.length && !filtered.some((t) => t.key === taskKey)) setTaskKey(filtered[0].key);
  }, [filtered, taskKey]);
  useEffect(() => {
    if (family !== "all" && !families.includes(family)) setFamily("all");
  }, [families, family]);

  // Shareable URL state.
  useEffect(() => {
    const p = new URLSearchParams();
    if (taskKey) p.set("task", taskKey);
    if (cfgA) p.set("config", cfgA);
    if (cfgB) p.set("compare", cfgB);
    if (mode === "live") p.set("mode", "live");
    window.history.replaceState(null, "", `${window.location.pathname}?${p.toString()}`);
  }, [taskKey, cfgA, cfgB, mode]);

  const task = tasks.find((t) => t.key === taskKey);
  const idx = filtered.findIndex((t) => t.key === taskKey);
  const cfg = (k: string) => configs.find((c) => c.key === k);

  if (!tasks.length) {
    return (
      <div className="empty">
        <strong>No showcase tasks exported yet.</strong> Run <code>python scripts/export_web.py</code> after at least one
        evaluation run exists.
      </div>
    );
  }

  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="card stack">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div className="segmented" role="group" aria-label="Trace source">
            <button aria-pressed={mode === "replay"} onClick={() => setMode("replay")}>
              Replay (recorded)
            </button>
            <button aria-pressed={mode === "live"} onClick={() => setMode("live")}>
              Live (local inference)
            </button>
          </div>
          <span className="muted small">
            {tasks.length} shared tasks · same list for every configuration
          </span>
        </div>
        <div className="controls">
          <div className="field">
            <label htmlFor="f-split">Split</label>
            <select id="f-split" value={split} onChange={(e) => setSplit(e.target.value)}>
              <option value="all">All splits</option>
              {splits.map((s) => (
                <option key={s} value={s}>
                  {SPLIT_SHORT[s] ?? s} ({s})
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="f-family">Family</label>
            <select id="f-family" value={family} onChange={(e) => setFamily(e.target.value)}>
              <option value="all">All families</option>
              {families.map((f) => (
                <option key={f} value={f} title={FAMILY_HELP[f]}>
                  {f}
                </option>
              ))}
            </select>
          </div>
          <div className="field" style={{ gridColumn: "1 / -1" }}>
            <label htmlFor="f-task">
              Task ({idx + 1} of {filtered.length})
            </label>
            <div className="row" style={{ flexWrap: "nowrap" }}>
              <button
                className="btn"
                aria-label="Previous task"
                disabled={idx <= 0}
                onClick={() => setTaskKey(filtered[idx - 1].key)}
              >
                ←
              </button>
              <select id="f-task" value={taskKey} onChange={(e) => setTaskKey(e.target.value)} style={{ flex: 1 }}>
                {filtered.map((t) => (
                  <option key={t.key} value={t.key}>
                    {t.key} · {t.family} · {truncate(t.request, 70)}
                  </option>
                ))}
              </select>
              <button
                className="btn"
                aria-label="Next task"
                disabled={idx < 0 || idx >= filtered.length - 1}
                onClick={() => setTaskKey(filtered[idx + 1].key)}
              >
                →
              </button>
            </div>
          </div>
        </div>
      </div>

      {task ? <TaskPanel task={task} /> : null}

      {mode === "replay" && task ? (
        <>
          <div className="card stack">
            <div className="controls">
              <div className="field">
                <label htmlFor="f-cfg">Configuration</label>
                <select id="f-cfg" value={cfgA} onChange={(e) => chooseA(e.target.value)}>
                  {configs.map((c) => (
                    <option key={c.key} value={c.key}>
                      {c.label}
                      {traces[c.key] ? "" : " — no results yet"}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label htmlFor="f-cmp">Compare side by side with</label>
                <select id="f-cmp" value={cfgB} onChange={(e) => setCfgB(e.target.value)}>
                  <option value="">— none —</option>
                  {configs
                    .filter((c) => c.key !== cfgA)
                    .map((c) => (
                      <option key={c.key} value={c.key}>
                        {c.label}
                        {traces[c.key] ? "" : " — no results yet"}
                      </option>
                    ))}
                </select>
              </div>
            </div>
            <div>
              <div className="tiny muted" style={{ marginBottom: 6 }}>
                This task across configurations (recorded seed-0 runs) — select to view
              </div>
              <div className="row">
                {configs.map((c) => {
                  const tr = traces[c.key]?.[task.key];
                  const state = tr ? (tr.success ? "✓" : "✕") : "–";
                  const text = tr ? (tr.success ? "success" : "failure") : "no results yet";
                  return (
                    <button
                      key={c.key}
                      className={`chip ${tr ? (tr.success ? "chip-good" : "chip-critical") : "chip-outline"}`}
                      style={{ cursor: "pointer", font: "inherit", fontSize: "0.76rem" }}
                      aria-pressed={c.key === cfgA}
                      aria-label={`${c.label}: ${text}`}
                      onClick={() => chooseA(c.key)}
                    >
                      <span className="swatch" style={{ background: c.color }} aria-hidden="true" />
                      {c.short}
                      <span aria-hidden="true">{state}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <div className={`trace-cols${cfgB ? " two" : ""}`}>
            {[cfgA, cfgB]
              .filter(Boolean)
              .map((k) => {
                const c = cfg(k);
                const tr = traces[k]?.[task.key];
                const title = (
                  <div className="row">
                    <span className="swatch" style={{ background: c?.color }} aria-hidden="true" />
                    <h3>{c?.label ?? k}</h3>
                  </div>
                );
                return (
                  <section key={k} className="card" aria-label={`Trace for ${c?.label ?? k}`}>
                    {tr ? (
                      <TraceView
                        trace={tr}
                        success={tr.success}
                        checks={tr.checks}
                        failureMode={tr.failure_mode}
                        title={title}
                        badge={<ReplayBadge run={tr.source_run} />}
                      />
                    ) : (
                      <div className="stack">
                        {title}
                        <div className="empty">
                          <strong>No results yet.</strong>{" "}
                          {traces[k]
                            ? "This configuration has recorded traces, but not for this task."
                            : "This configuration has not been evaluated yet; its recorded traces will appear here once its evaluation run is exported."}
                        </div>
                      </div>
                    )}
                  </section>
                );
              })}
          </div>
        </>
      ) : null}

      {mode === "live" && task ? <LivePanel task={task} configs={configs} /> : null}
    </div>
  );
}
