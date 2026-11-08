"use client";

import { useState } from "react";
import type { Failures } from "@/lib/types";
import { FAILURE_HELP, SPLIT_SHORT, humanize } from "@/lib/configs";
import TraceView from "./TraceView";
import { ReplayBadge } from "./ModeBadges";

type Cfg = { key: string; label: string; short: string; color: string; replayRun: string };

const ORDER = Object.keys(FAILURE_HELP);

export default function FailureGallery({ configs, failures }: { configs: Cfg[]; failures: Failures }) {
  const present = configs.filter((c) => failures[c.key]);
  const [sel, setSel] = useState(
    () => present.find((c) => Object.keys(failures[c.key].counts ?? {}).length)?.key ?? present[0]?.key ?? "",
  );
  const modes = [
    ...ORDER.filter((m) => present.some((c) => failures[c.key].counts?.[m])),
    ...[...new Set(present.flatMap((c) => Object.keys(failures[c.key].counts ?? {})))].filter((m) => !ORDER.includes(m)),
  ];
  const cur = failures[sel];
  const curCfg = configs.find((c) => c.key === sel);
  const curModes = modes.filter((m) => cur?.counts?.[m]);
  const maxCount = Math.max(1, ...curModes.map((m) => cur.counts[m]));

  return (
    <div className="stack" style={{ gap: 24 }}>
      <div>
        <h2 style={{ marginBottom: 8 }}>Failure modes by configuration</h2>
        <p className="muted small" style={{ maxWidth: 760 }}>
          Each failed episode is assigned the <em>first</em> matching mode in the order listed (malformed output takes
          precedence over a hallucinated tool, and so on), so modes are mutually exclusive. Counts are over all main-split
          episodes of the seed-0 evaluation run.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>failure mode</th>
                {configs.map((c) => (
                  <th key={c.key} className="n">
                    <span className="row" style={{ justifyContent: "flex-end", gap: 5, flexWrap: "nowrap" }}>
                      <span className="swatch" style={{ background: c.color }} aria-hidden="true" />
                      {c.short}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {modes.map((m) => (
                <tr key={m}>
                  <td title={FAILURE_HELP[m]}>{humanize(m)}</td>
                  {configs.map((c) => {
                    const f = failures[c.key];
                    if (!f) return <td key={c.key} className="n muted">—</td>;
                    const n = f.counts?.[m] ?? 0;
                    return (
                      <td key={c.key} className="n">
                        {n}
                        {f.total ? <span className="tiny muted"> ({((n / f.total) * 100).toFixed(1)}%)</span> : null}
                      </td>
                    );
                  })}
                </tr>
              ))}
              <tr>
                <td>
                  <strong>all failures</strong>
                </td>
                {configs.map((c) => {
                  const f = failures[c.key];
                  if (!f) return <td key={c.key} className="n muted small">no results yet</td>;
                  const n = Object.values(f.counts ?? {}).reduce((a, b) => a + b, 0);
                  return (
                    <td key={c.key} className="n">
                      <strong>{n}</strong>
                      <span className="tiny muted"> / {f.total}</span>
                    </td>
                  );
                })}
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <div className="row" style={{ justifyContent: "space-between", marginBottom: 10 }}>
          <h2>Examples</h2>
          <div className="field" style={{ minWidth: 240 }}>
            <label htmlFor="fg-cfg">Configuration</label>
            <select id="fg-cfg" value={sel} onChange={(e) => setSel(e.target.value)}>
              {configs.map((c) => (
                <option key={c.key} value={c.key} disabled={!failures[c.key]}>
                  {c.label}
                  {failures[c.key] ? "" : " — no results yet"}
                </option>
              ))}
            </select>
          </div>
        </div>

        {!cur ? (
          <div className="empty">
            <strong>No results yet</strong> for this configuration.
          </div>
        ) : !curModes.length ? (
          <div className="empty">
            <strong>No failures recorded</strong> for {curCfg?.label ?? sel}: all {cur.total} evaluated episodes passed
            every validator check.
            {sel === "oracle"
              ? " Expected — the oracle reads the gold graph; its clean sheet shows that every task is solvable and the validator is self-consistent."
              : ""}
          </div>
        ) : (
          <div className="stack" style={{ gap: 18 }}>
            <div className="card">
              <ul className="fbars" aria-label={`Failure counts for ${curCfg?.label ?? sel}`}>
                {curModes.map((m) => (
                  <li key={m}>
                    <a href={`#fm-${m}`}>{humanize(m)}</a>
                    <span className="fbar" aria-hidden="true">
                      <span style={{ width: `${(cur.counts[m] / maxCount) * 100}%`, background: curCfg?.color }} />
                    </span>
                    <span className="num small">
                      {cur.counts[m]} <span className="muted tiny">/ {cur.total}</span>
                    </span>
                  </li>
                ))}
              </ul>
            </div>
            {curModes.map((m) => {
              const ex = cur.examples?.[m] ?? [];
              return (
                <section key={m} id={`fm-${m}`} aria-labelledby={`fmh-${m}`}>
                  <h3 id={`fmh-${m}`} style={{ marginBottom: 2 }}>
                    {humanize(m)} <span className="muted small num">· {cur.counts[m]} episodes</span>
                  </h3>
                  <p className="muted small">{FAILURE_HELP[m] ?? ""}</p>
                  {ex.length ? (
                    <div className="stack">
                      {ex.map((e, i) => (
                        <details key={i} className="card">
                          <summary>
                            <span className="row" style={{ display: "inline-flex", gap: 6 }}>
                              <span className="chip">{SPLIT_SHORT[e.split] ?? e.split}</span>
                              <span className="chip">{e.family}</span>
                              <span className="chip">role: {e.role}</span>
                              <span className="chip">expected: {e.expected_status}</span>
                            </span>
                            <div style={{ marginTop: 6, color: "var(--text)", fontWeight: 450 }}>{e.request}</div>
                          </summary>
                          <div style={{ marginTop: 12 }}>
                            <p className="tiny muted mono">
                              {e.split}:{e.index} · {e.task_id}
                            </p>
                            <TraceView
                              trace={e.trace}
                              success={false}
                              failureMode={m}
                              badge={<ReplayBadge run={curCfg?.replayRun ?? sel} />}
                            />
                          </div>
                        </details>
                      ))}
                    </div>
                  ) : (
                    <p className="muted small">No example traces exported for this mode.</p>
                  )}
                </section>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
