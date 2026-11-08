import type { GoldStep } from "@/lib/types";

const CHAR_W = 6.4;
const NODE_H = 36;
const GAP_X = 34;
const GAP_Y = 12;
const PAD = 6;

/** Small layered DAG of the gold workflow (columns = dependency depth). */
export default function GoldDag({ gold, split }: { gold: GoldStep[]; split?: string }) {
  if (!gold.length) {
    return (
      <p className="muted small" style={{ margin: 0 }}>
        No tool calls in the gold workflow — the correct behaviour is to answer or stop directly.
      </p>
    );
  }
  const byId = new Map(gold.map((g) => [g.id, g]));
  const layer = new Map<string, number>();
  const depthOf = (g: GoldStep, stack: Set<string> = new Set()): number => {
    if (layer.has(g.id)) return layer.get(g.id)!;
    if (stack.has(g.id)) return 0; // defensive: cycles should not exist
    stack.add(g.id);
    const d = Math.max(-1, ...g.deps.map((id) => (byId.has(id) ? depthOf(byId.get(id)!, stack) : -1))) + 1;
    layer.set(g.id, d);
    return d;
  };
  gold.forEach((g) => depthOf(g));
  const nLayers = Math.max(...layer.values()) + 1;
  const cols: GoldStep[][] = Array.from({ length: nLayers }, () => []);
  gold.forEach((g) => cols[layer.get(g.id)!].push(g));

  const nodeW = (g: GoldStep) => Math.max(64, g.tool.length * CHAR_W + 18);
  const colW = cols.map((c) => Math.max(...c.map(nodeW)));
  const colX: number[] = [];
  let x = PAD;
  for (const w of colW) {
    colX.push(x);
    x += w + GAP_X;
  }
  const width = x - GAP_X + PAD;
  const maxRows = Math.max(...cols.map((c) => c.length));
  const height = PAD * 2 + maxRows * NODE_H + (maxRows - 1) * GAP_Y;

  const pos = new Map<string, { x: number; y: number; w: number }>();
  cols.forEach((c, ci) => {
    const colH = c.length * NODE_H + (c.length - 1) * GAP_Y;
    const y0 = (height - colH) / 2;
    c.forEach((g, ri) => pos.set(g.id, { x: colX[ci], y: y0 + ri * (NODE_H + GAP_Y), w: colW[ci] }));
  });

  const hasOptional = gold.some((g) => g.optional);
  const hasNoExec = gold.some((g) => !g.execute);
  const desc = gold
    .map(
      (g) =>
        `${g.id} ${g.tool}${g.deps.length ? ` after ${g.deps.join(", ")}` : ""}${g.optional ? " (optional)" : ""}${
          g.execute ? "" : " (expected not to execute)"
        }`,
    )
    .join("; ");

  return (
    <div>
      <div style={{ overflowX: "auto" }}>
        <svg
          className="dag"
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={`Gold workflow: ${desc}`}
        >
          <defs>
            <marker id="dag-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
              <path d="M0,0 L8,4 L0,8 z" fill="var(--axis)" />
            </marker>
          </defs>
          {gold.flatMap((g) =>
            g.deps
              .filter((d) => pos.has(d))
              .map((d) => {
                const a = pos.get(d)!;
                const b = pos.get(g.id)!;
                const x1 = a.x + nodeW(byId.get(d)!);
                const y1 = a.y + NODE_H / 2;
                const x2 = b.x - 2;
                const y2 = b.y + NODE_H / 2;
                const mx = (x1 + x2) / 2;
                return (
                  <path
                    key={`${d}-${g.id}`}
                    className="edge"
                    d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`}
                    markerEnd="url(#dag-arrow)"
                  />
                );
              }),
          )}
          {gold.map((g) => {
            const p = pos.get(g.id)!;
            const w = nodeW(g);
            return (
              <g
                key={g.id}
                className={`node${g.optional ? " optional" : ""}${g.execute ? "" : " noexec"}`}
                transform={`translate(${p.x},${p.y})`}
              >
                <rect width={w} height={NODE_H} rx={5} />
                <text className="id" x={8} y={12}>
                  {g.id}
                  {g.optional ? " · optional" : ""}
                  {g.execute ? "" : " · not executed"}
                </text>
                <text x={8} y={27}>
                  {g.tool}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <p className="muted tiny" style={{ margin: "6px 0 0" }}>
        Columns are dependency depth; arrows point from prerequisite to dependent call. Nodes in the same column are
        independent and can run in parallel.
        {hasOptional ? " Dashed = optional step." : ""}
        {hasNoExec ? " Shaded = the call is expected but must not take effect (e.g. the approver denies it)." : ""}
        {split === "eval_renamed" ? " Gold steps use canonical simulator names; the model sees renamed aliases." : ""}
      </p>
    </div>
  );
}
