import type { ShowcaseTask } from "@/lib/types";
import { FAMILY_HELP, SPLIT_SHORT } from "@/lib/configs";
import GoldDag from "./GoldDag";

/** Parse "name(params) [approval] [role: x] - description" into parts. */
export function parseSignature(sig: string) {
  const m = sig.match(/^(.*?\))((?:\s*\[[^\]]+\])*)\s*(?:-\s*(.*))?$/s);
  if (!m) return { head: sig, tags: [] as string[], desc: "" };
  const tags = [...(m[2] ?? "").matchAll(/\[([^\]]+)\]/g)].map((x) => x[1]);
  return { head: m[1], tags, desc: m[3] ?? "" };
}

export function Catalog({ catalog }: { catalog: string[] }) {
  return (
    <ul className="catalog" aria-label="Tool catalogue">
      {catalog.map((s, i) => {
        const p = parseSignature(s);
        return (
          <li key={i}>
            <span className="sig">{p.head}</span>
            {p.tags.map((t) => (
              <span key={t} className={`tag ${t === "approval" ? "tag-approval" : "tag-role"}`}>
                {t}
              </span>
            ))}
            {p.desc ? <div className="desc">{p.desc}</div> : null}
          </li>
        );
      })}
    </ul>
  );
}

export default function TaskPanel({ task }: { task: ShowcaseTask }) {
  return (
    <div className="card stack">
      <div className="row">
        <span className="chip" title={task.split}>
          {SPLIT_SHORT[task.split] ?? task.split}
        </span>
        <span className="chip" title={FAMILY_HELP[task.family]}>
          family: {task.family}
        </span>
        <span className="chip mono">{task.template}</span>
        <span className="chip">role: {task.role}</span>
        <span className="chip">depth {task.depth}</span>
        <span className="chip">{task.graph_seen ? "graph seen in training" : "novel task graph"}</span>
        <span className="chip">expected status: {task.expected_status}</span>
      </div>
      <blockquote className="request">{task.request}</blockquote>
      <p className="muted tiny" style={{ margin: 0 }}>
        <span className="mono">{task.task_id}</span> · simulated date {task.today}
        {FAMILY_HELP[task.family] ? ` · ${task.family}: ${FAMILY_HELP[task.family]}` : ""}
      </p>
      <div className="grid grid-2" style={{ alignItems: "start" }}>
        <div>
          <h4 style={{ marginBottom: 6 }}>Tool catalogue shown to the model ({task.catalog.length})</h4>
          <Catalog catalog={task.catalog} />
        </div>
        <div className="stack">
          <div>
            <h4 style={{ marginBottom: 6 }}>Gold dependency graph</h4>
            <GoldDag gold={task.gold} split={task.split} />
          </div>
          {task.answer_facts.length ? (
            <div>
              <h4 style={{ marginBottom: 6 }}>Facts the answer must contain</h4>
              <div className="row">
                {task.answer_facts.map((f) => (
                  <span key={f} className="chip mono">
                    {f}
                  </span>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
