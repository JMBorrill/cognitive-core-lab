"use client";

import { useState } from "react";
import type { CI } from "@/lib/types";
import { isNum } from "@/lib/configs";

export type DotRow = {
  id: string;
  label: string;
  color: string;
  ci?: CI;
  /** Shown instead of a mark when there is no estimate (e.g. "no results yet"). */
  empty?: string;
};

export type DotGroup = { id: string; title: string; subtitle?: string; rows: DotRow[] };

type Props = {
  groups: DotGroup[];
  /** "pct": values are fractions shown as %, domain 0..1. "num": raw values, domain 0..max. */
  format?: "pct" | "num";
  digits?: number;
  caption?: string;
  unit?: string;
};

function fmt(v: number | null | undefined, format: "pct" | "num", digits: number) {
  if (!isNum(v)) return "—";
  return format === "pct" ? (v * 100).toFixed(digits) : v.toFixed(digits);
}

/**
 * Horizontal dot-and-whisker plot (mean with 95% CI), grouped. HTML layout so text stays
 * legible at every width; each row is focusable and carries its numbers as visible text.
 */
export default function DotPlot({ groups, format = "pct", digits = 1, caption, unit }: Props) {
  const [tip, setTip] = useState<{ row: DotRow; group: DotGroup } | null>(null);

  let max = 1;
  if (format === "num") {
    max = 0;
    for (const g of groups) for (const r of g.rows) if (r.ci && isNum(r.ci.hi)) max = Math.max(max, r.ci.hi);
    max = niceMax(max || 1);
  }
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * max);
  const pos = (v: number) => `${Math.max(0, Math.min(100, (v / max) * 100))}%`;
  const tickLabel = (v: number) => (format === "pct" ? `${Math.round(v * 100)}%` : trimNum(v));

  return (
    <figure className="dotplot" style={{ margin: 0 }}>
      {caption ? <figcaption className="sr-only">{caption}</figcaption> : null}
      <div className="dp-axis" aria-hidden="true">
        <div />
        <div className="dp-track dp-ticks">
          {ticks.map((t, i) => (
            <span key={i} style={{ left: pos(t) }}>
              {tickLabel(t)}
            </span>
          ))}
        </div>
        <div className="dp-value-head">
          {format === "pct" ? "mean % [95% CI]" : `mean${unit ? ` ${unit}` : ""} [95% CI]`}
        </div>
      </div>
      {groups.map((g) => (
        <div key={g.id} className="dp-group" role="group" aria-label={g.title}>
          <div className="dp-group-title">
            <span>{g.title}</span>
            {g.subtitle ? <span className="muted tiny"> {g.subtitle}</span> : null}
          </div>
          {g.rows.map((r) => {
            const has = r.ci && isNum(r.ci.mean);
            const ci = r.ci;
            const text = has
              ? `${fmt(ci!.mean, format, digits)} [${fmt(ci!.lo, format, digits)}, ${fmt(ci!.hi, format, digits)}]`
              : (r.empty ?? "no results yet");
            const aria = `${g.title}, ${r.label}: ${has ? text + (format === "pct" ? " percent" : "") + `, n=${ci!.n}` : text}`;
            return (
              <div
                key={r.id}
                className={`dp-row${has ? "" : " dp-empty"}`}
                role="img"
                tabIndex={has ? 0 : undefined}
                aria-label={aria}
                onMouseEnter={() => has && setTip({ row: r, group: g })}
                onMouseLeave={() => setTip(null)}
                onFocus={() => has && setTip({ row: r, group: g })}
                onBlur={() => setTip(null)}
              >
                <div className="dp-label">
                  <span className="swatch" style={{ background: r.color }} aria-hidden="true" />
                  <span className="dp-label-text">{r.label}</span>
                </div>
                <div className="dp-track" aria-hidden="true">
                  {ticks.map((t, i) => (
                    <span key={i} className="dp-grid" style={{ left: pos(t) }} />
                  ))}
                  {has && isNum(ci!.lo) && isNum(ci!.hi) ? (
                    <span
                      className="dp-ci"
                      style={{
                        left: pos(ci!.lo),
                        width: `calc(${pos(ci!.hi)} - ${pos(ci!.lo)})`,
                        background: r.color,
                      }}
                    />
                  ) : null}
                  {has ? <span className="dp-dot" style={{ left: pos(ci!.mean as number), background: r.color }} /> : null}
                  {tip && tip.row === r && tip.group === g && has ? (
                    <span className="chart-tip dp-tip" aria-hidden="true" style={{ left: pos(ci!.mean as number) }}>
                      <strong>{r.label}</strong> · {g.title}
                      <br />
                      {text}
                      {format === "pct" ? " %" : ""}
                      <br />
                      <span className="muted">
                        n = {ci!.n} tasks
                        {ci!.seeds ? ` · ${ci!.seeds} seed${ci!.seeds > 1 ? "s" : ""}` : ""}
                        {isNum(ci!.seed_sd) ? ` · seed SD ${fmt(ci!.seed_sd, format, digits)}` : ""}
                      </span>
                    </span>
                  ) : null}
                </div>
                <div className={`dp-value num${has ? "" : " muted"}`}>{text}</div>
              </div>
            );
          })}
        </div>
      ))}
    </figure>
  );
}

function niceMax(v: number) {
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (v <= m * p) return m * p;
  return 10 * p;
}

function trimNum(v: number) {
  return Number.isInteger(v) ? String(v) : v.toFixed(v < 1 ? 2 : 1).replace(/\.?0+$/, "");
}
