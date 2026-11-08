"use client";

import { useState } from "react";
import type { BreakdownCell, CI } from "@/lib/types";
import { SPLIT_SHORT, isNum, pct } from "@/lib/configs";

type Cfg = { key: string; rkey: string; label: string; color: string };

function Tabs({
  items,
  value,
  onChange,
  label,
}: {
  items: { id: string; label: string }[];
  value: string;
  onChange: (id: string) => void;
  label: string;
}) {
  return (
    <div className="tabs" role="group" aria-label={label}>
      {items.map((it) => (
        <button key={it.id} aria-pressed={value === it.id} onClick={() => onChange(it.id)}>
          {it.label}
        </button>
      ))}
    </div>
  );
}

function CICell({ ci, asPct }: { ci?: CI; asPct: boolean }) {
  if (!ci || !isNum(ci.mean)) return <td className="n muted">—</td>;
  const k = asPct ? 100 : 1;
  const d = asPct ? 1 : 2;
  const f = (v: number | null) => (isNum(v) ? (v * k).toFixed(d) : "?");
  return (
    <td className="n" title={`n = ${ci.n}${ci.seeds ? ` · ${ci.seeds} seed(s)` : ""}`}>
      <div>{f(ci.mean)}</div>
      <div className="tiny muted">
        [{f(ci.lo)}, {f(ci.hi)}]
      </div>
    </td>
  );
}

const DIAG: { id: string; label: string; pct: boolean; help: string }[] = [
  { id: "success", label: "Success %", pct: true, help: "All validator checks pass" },
  { id: "tool_selection", label: "Tool selection %", pct: true, help: "Set-level F1 between tools called and gold tools" },
  { id: "arg_correct", label: "Args correct %", pct: true, help: "Gold calls matched with correct arguments, given the right tool was called" },
  { id: "dependency_safe", label: "Dependency-safe %", pct: true, help: "No invented identifiers; no dependent call before its prerequisite returned" },
  { id: "strict_schema", label: "Strict schema %", pct: true, help: "Every turn is exactly one well-formed JSON action" },
  { id: "hallucinated_tools", label: "Halluc. tools", pct: false, help: "Calls to tools not in the catalogue, per episode" },
  { id: "unnecessary_calls", label: "Unneeded calls", pct: false, help: "Calls beyond the gold workflow, per episode" },
  { id: "invalid_args", label: "Invalid args", pct: false, help: "Calls rejected for invalid arguments, per episode" },
  { id: "parse_errors", label: "Parse errors", pct: false, help: "Unparseable turns per episode" },
  { id: "turns", label: "Turns", pct: false, help: "Model turns per episode" },
  { id: "n_calls", label: "Calls", pct: false, help: "Tool calls per episode" },
];

export function DiagnosticsTable({
  configs,
  data,
  splits,
}: {
  configs: Cfg[];
  /** data[rkey][split][metric] */
  data: Record<string, Record<string, Record<string, CI>>>;
  splits: string[];
}) {
  const [split, setSplit] = useState(splits.includes("all_main") ? "all_main" : splits[0]);
  const rows = configs.filter((c) => data[c.rkey]);
  return (
    <div>
      <Tabs
        label="Split"
        value={split}
        onChange={setSplit}
        items={splits.map((s) => ({ id: s, label: SPLIT_SHORT[s] ?? s }))}
      />
      <div className="table-wrap">
        <table>
          <caption className="sr-only">Diagnostics for {SPLIT_SHORT[split] ?? split}: mean with 95% CI</caption>
          <thead>
            <tr>
              <th>configuration</th>
              <th className="n">n</th>
              {DIAG.map((m) => (
                <th key={m.id} className="n" title={m.help}>
                  {m.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((c) => {
              const block = data[c.rkey]?.[split] ?? {};
              return (
                <tr key={c.key}>
                  <td className="cfg">
                    <span className="row" style={{ flexWrap: "nowrap", gap: 6 }}>
                      <span className="swatch" style={{ background: c.color }} aria-hidden="true" />
                      {c.label}
                    </span>
                  </td>
                  <td className="n">{block.success?.n ?? "—"}</td>
                  {DIAG.map((m) => (
                    <CICell key={m.id} ci={block[m.id]} asPct={m.pct} />
                  ))}
                </tr>
              );
            })}
            {configs
              .filter((c) => !data[c.rkey])
              .map((c) => (
                <tr key={c.key}>
                  <td className="cfg muted">
                    <span className="row" style={{ flexWrap: "nowrap", gap: 6 }}>
                      <span className="swatch" style={{ background: c.color }} aria-hidden="true" />
                      {c.label}
                    </span>
                  </td>
                  <td colSpan={DIAG.length + 1} className="muted small">
                    no results yet
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      <dl className="kv tiny" style={{ marginTop: 10 }}>
        {DIAG.map((m) => (
          <div key={m.id} style={{ display: "contents" }}>
            <dt>{m.label}</dt>
            <dd className="muted">{m.help}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

const DIM_LABEL: Record<string, string> = {
  family: "Task family",
  depth: "Workflow depth",
  graph_seen: "Graph seen vs novel",
  fault: "Fault injection",
  split: "Split",
};

function groupLabel(dim: string, g: string) {
  if (dim === "graph_seen") return g === "True" ? "seen in training" : g === "False" ? "novel graph" : g;
  if (dim === "fault") return g === "True" ? "fault / not-found" : g === "False" ? "no fault" : g;
  if (dim === "depth") return `depth ${g}`;
  if (dim === "split") return SPLIT_SHORT[g] ?? g;
  return g;
}

export function BreakdownTables({
  configs,
  data,
}: {
  configs: Cfg[];
  /** data[rkey][dimension][group] */
  data: Record<string, Record<string, Record<string, BreakdownCell>>>;
}) {
  const dims = ["family", "depth", "graph_seen", "fault", "split"].filter((d) =>
    Object.values(data).some((b) => b[d] && Object.keys(b[d]).length),
  );
  const [dim, setDim] = useState(dims[0] ?? "family");
  const rows = configs.filter((c) => data[c.rkey]);
  if (!rows.length) return null;
  const groups = [...new Set(rows.flatMap((c) => Object.keys(data[c.rkey]?.[dim] ?? {})))].sort((a, b) => {
    const na = Number(a);
    const nb = Number(b);
    if (Number.isFinite(na) && Number.isFinite(nb)) return na - nb;
    if (dim === "split") return Object.keys(SPLIT_SHORT).indexOf(a) - Object.keys(SPLIT_SHORT).indexOf(b);
    return a.localeCompare(b);
  });
  return (
    <div>
      <Tabs label="Breakdown dimension" value={dim} onChange={setDim} items={dims.map((d) => ({ id: d, label: DIM_LABEL[d] ?? d }))} />
      <div className="table-wrap">
        <table>
          <caption className="sr-only">Success by {DIM_LABEL[dim] ?? dim}</caption>
          <thead>
            <tr>
              <th>configuration</th>
              {groups.map((g) => (
                <th key={g} className="n">
                  {groupLabel(dim, g)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((c) => (
              <tr key={c.key}>
                <td className="cfg">
                  <span className="row" style={{ flexWrap: "nowrap", gap: 6 }}>
                    <span className="swatch" style={{ background: c.color }} aria-hidden="true" />
                    {c.label}
                  </span>
                </td>
                {groups.map((g) => {
                  const cell = data[c.rkey]?.[dim]?.[g];
                  if (!cell || !isNum(cell.success)) return <td key={g} className="n muted">—</td>;
                  return (
                    <td key={g} className="n">
                      <div>{pct(cell.success)}</div>
                      <div className="minibar" aria-hidden="true">
                        <span style={{ width: `${cell.success * 100}%`, background: c.color }} />
                      </div>
                      <div className="tiny muted">n={cell.n_tasks}</div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
