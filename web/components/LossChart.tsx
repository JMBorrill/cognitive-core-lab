"use client";

import { useState } from "react";
import type { TrainingLogRow } from "@/lib/types";
import { isNum } from "@/lib/configs";

const W = 420;
const H = 200;
const M = { l: 40, r: 12, t: 10, b: 28 };

/** Training loss (line) and validation loss (dots) against step, with a hover crosshair. */
export default function LossChart({
  log,
  color,
  yMax,
  title,
}: {
  log: TrainingLogRow[];
  color: string;
  yMax: number;
  title: string;
}) {
  const [hover, setHover] = useState<TrainingLogRow | null>(null);
  const pts = log.filter((r) => isNum(r.step));
  if (!pts.length) return <p className="muted small">No logged steps.</p>;
  const xMax = Math.max(...pts.map((r) => r.step));
  const xMin = Math.min(...pts.map((r) => r.step));
  const iw = W - M.l - M.r;
  const ih = H - M.t - M.b;
  const x = (s: number) => M.l + (xMax === xMin ? iw / 2 : ((s - xMin) / (xMax - xMin)) * iw);
  const y = (v: number) => M.t + ih - (Math.min(v, yMax) / yMax) * ih;
  const train = pts.filter((r) => isNum(r.loss));
  const val = pts.filter((r) => isNum(r.val_loss));
  const line = train.map((r, i) => `${i ? "L" : "M"}${x(r.step).toFixed(1)},${y(r.loss!).toFixed(1)}`).join("");
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * yMax);
  const xTicks = [xMin, Math.round((xMin + xMax) / 2), xMax];

  function onMove(e: React.MouseEvent<SVGRectElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const sx = ((e.clientX - rect.left) / rect.width) * iw;
    const step = xMin + (sx / iw) * (xMax - xMin);
    let best = pts[0];
    for (const r of pts) if (Math.abs(r.step - step) < Math.abs(best.step - step)) best = r;
    setHover(best);
  }

  const last = train[train.length - 1];
  const lastVal = val[val.length - 1];
  return (
    <figure style={{ margin: 0, position: "relative" }}>
      <svg
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${title}: training loss over ${xMax} steps${last ? `, final ${last.loss!.toFixed(3)}` : ""}${
          lastVal ? `; last validation loss ${lastVal.val_loss!.toFixed(3)}` : ""
        }`}
      >
        {yTicks.map((t) => (
          <g key={t}>
            <line className="grid-line" x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} />
            <text x={M.l - 6} y={y(t) + 3.5} textAnchor="end">
              {t.toFixed(yMax < 2 ? 2 : 1)}
            </text>
          </g>
        ))}
        <line className="axis-line" x1={M.l} x2={W - M.r} y1={M.t + ih} y2={M.t + ih} />
        {xTicks.map((t, i) => (
          <text key={i} x={x(t)} y={H - 10} textAnchor={i === 0 ? "start" : i === 2 ? "end" : "middle"}>
            {t}
          </text>
        ))}
        <text className="axis-label" x={M.l + iw / 2} y={H - 0.5} textAnchor="middle">
          step
        </text>
        <path d={line} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" />
        {val.map((r) => (
          <circle
            key={r.step}
            cx={x(r.step)}
            cy={y(r.val_loss!)}
            r={4}
            fill="var(--surface)"
            stroke="var(--text)"
            strokeWidth={1.5}
          />
        ))}
        {hover ? (
          <line className="axis-line" x1={x(hover.step)} x2={x(hover.step)} y1={M.t} y2={M.t + ih} strokeDasharray="3 3" />
        ) : null}
        <rect
          x={M.l}
          y={M.t}
          width={iw}
          height={ih}
          fill="transparent"
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        />
      </svg>
      {hover ? (
        <div
          className="chart-tip"
          style={{ left: `${(x(hover.step) / W) * 100}%`, top: 0, transform: "translateX(-50%)" }}
        >
          step {hover.step}
          {isNum(hover.loss) ? ` · train ${hover.loss.toFixed(4)}` : ""}
          {isNum(hover.val_loss) ? ` · val ${hover.val_loss.toFixed(4)}` : ""}
          {isNum(hover.tok_per_s) ? ` · ${Math.round(hover.tok_per_s)} tok/s` : ""}
        </div>
      ) : null}
    </figure>
  );
}
