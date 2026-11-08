import type { Metadata } from "next";
import { loadData } from "@/lib/data";
import { Markdown } from "@/lib/markdown";
import { humanize, isNum } from "@/lib/configs";

export const metadata: Metadata = { title: "Methods & reproducibility" };

function isScalar(v: unknown) {
  return v === null || ["string", "number", "boolean"].includes(typeof v);
}

function fmtScalar(v: unknown) {
  if (v === null) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString("en-US") : v.toFixed(3);
  return String(v);
}

function GenericKV({ obj }: { obj: Record<string, unknown> }) {
  return (
    <dl className="kv">
      {Object.entries(obj).map(([key, v]) => (
        <div key={key} style={{ display: "contents" }}>
          <dt>{humanize(key)}</dt>
          <dd>
            {isScalar(v) ? (
              <span className={typeof v === "string" && /^[0-9a-f]{32,}$/.test(v) ? "mono small" : "num"}>
                {fmtScalar(v)}
              </span>
            ) : typeof v === "object" && v && Object.values(v).every(isScalar) && Object.keys(v).length <= 12 ? (
              <span className="row" style={{ gap: 4 }}>
                {Object.entries(v).map(([kk, vv]) => (
                  <span key={kk} className="chip">
                    {humanize(kk)}: {fmtScalar(vv)}
                  </span>
                ))}
              </span>
            ) : (
              <details>
                <summary>show</summary>
                <pre className="codeblock">{JSON.stringify(v, null, 2)}</pre>
              </details>
            )}
          </dd>
        </div>
      ))}
    </dl>
  );
}

const PIPELINE: [string, string][] = [
  ["python scripts/build_datasets.py", "Versioned task pools, oracle (config C) SFT data and the leakage audit → data/v1/"],
  ["python scripts/distill.py --n 1600", "Teacher demonstrations with rejection sampling (config B data) → data/v1/teacher/"],
  ["python -m ccl.train configs/train/<name>.json", "LoRA training per configuration and seed → runs/<config>_s<seed>/"],
  ["python scripts/evaluate.py --name B_s0 --adapter runs/B_s0", "Evaluate a configuration on the held-out splits → results/runs/<name>/"],
  ["python scripts/analyze.py", "Hierarchical-bootstrap CIs, paired comparisons, breakdowns → results/summary.json"],
  ["python scripts/export_web.py", "Publish real run outputs for this site → web/public/data/"],
  ["python scripts/serve.py --port 8765", "Optional: local inference server for the trace explorer's live mode"],
];

export default async function MethodsPage() {
  const { report, datasets, leakage, teacher, manifest } = await loadData([
    "report",
    "datasets",
    "leakage",
    "teacher",
    "manifest",
  ] as const);

  const files = Object.entries(datasets?.files ?? {});
  const templates = Object.entries(datasets?.templates ?? {});
  const leak = Object.entries(leakage ?? {});

  return (
    <>
      <header className="page-head">
        <div className="eyebrow">Methods &amp; reproducibility</div>
        <h1>How the numbers were produced</h1>
        <p className="lede">
          The write-up, the exact data files (by SHA-256), the train/eval leakage audit and the teacher-generation
          record. Everything here is exported from the repository; nothing is entered by hand.
        </p>
      </header>

      <section className="block" aria-labelledby="h-report">
        <div className="block-head">
          <h2 id="h-report">Report</h2>
        </div>
        {report ? (
          <article className="card prose">
            <Markdown source={report} />
          </article>
        ) : (
          <div className="empty">
            <strong>The written report is not published yet.</strong> It will render here from{" "}
            <code>docs/REPORT.md</code> (copied to <code>public/data/report.md</code> by{" "}
            <code>scripts/export_web.py</code>) once the experiments are complete. Until then, see the benchmarks and the
            pipeline below.
          </div>
        )}
      </section>

      <section className="block" aria-labelledby="h-pipeline">
        <div className="block-head">
          <div>
            <h2 id="h-pipeline">Pipeline</h2>
            <p>Run from the repository root, in order. Each stage is resumable and writes versioned outputs.</p>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>command</th>
                <th>what it does</th>
              </tr>
            </thead>
            <tbody>
              {PIPELINE.map(([cmd, what]) => (
                <tr key={cmd}>
                  <td className="mono small" style={{ whiteSpace: "nowrap" }}>
                    {cmd}
                  </td>
                  <td>{what}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {manifest ? (
          <p className="muted small" style={{ marginTop: 8 }}>
            Site data exported {manifest.generated_at}
            {manifest.git_commit ? (
              <>
                {" "}
                at commit <code>{manifest.git_commit}</code>
              </>
            ) : (
              " (no git commit recorded)"
            )}
            .
          </p>
        ) : null}
      </section>

      <section className="block" aria-labelledby="h-data">
        <div className="block-head">
          <div>
            <h2 id="h-data">Datasets{datasets?.version ? ` (${datasets.version})` : ""}</h2>
            <p>Every task pool and training file with its SHA-256, size and per-family composition.</p>
          </div>
        </div>
        {files.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>file</th>
                  <th className="n">size</th>
                  <th>composition</th>
                  <th>sha256</th>
                </tr>
              </thead>
              <tbody>
                {files.map(([name, f]) => (
                  <tr key={name}>
                    <td className="mono small nowrap">{name}</td>
                    <td className="n">
                      {isNum(f.n) ? `${f.n} tasks` : null}
                      {isNum(f.rows) ? (
                        <>
                          {f.rows} rows
                          {isNum(f.episodes) ? <div className="tiny muted">{f.episodes} episodes</div> : null}
                        </>
                      ) : null}
                    </td>
                    <td>
                      {f.families ? (
                        <span className="row" style={{ gap: 4 }}>
                          {Object.entries(f.families).map(([fam, n]) => (
                            <span key={fam} className="chip">
                              {fam} {n}
                            </span>
                          ))}
                        </span>
                      ) : f.rejected && Object.keys(f.rejected).length ? (
                        <span className="small">
                          rejected:{" "}
                          {Object.entries(f.rejected)
                            .map(([r, n]) => `${humanize(r)} ${n}`)
                            .join(", ")}
                        </span>
                      ) : (
                        <span className="muted small">—</span>
                      )}
                    </td>
                    <td className="mono tiny" style={{ wordBreak: "break-all", minWidth: 180 }}>
                      {f.sha256}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty">
            <code>datasets.json</code> has not been exported yet.
          </div>
        )}
        {templates.length ? (
          <details style={{ marginTop: 10 }}>
            <summary>Task templates ({templates.length}) — which split each template belongs to</summary>
            <div className="table-wrap" style={{ marginTop: 8 }}>
              <table>
                <thead>
                  <tr>
                    <th>template</th>
                    <th>family</th>
                    <th>split</th>
                    <th className="n">depth</th>
                  </tr>
                </thead>
                <tbody>
                  {templates.map(([name, t]) => (
                    <tr key={name}>
                      <td className="mono small">{name}</td>
                      <td>{t.family}</td>
                      <td>{t.split}</td>
                      <td className="n">{t.depth}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        ) : null}
      </section>

      <section className="block" aria-labelledby="h-leak">
        <div className="block-head">
          <div>
            <h2 id="h-leak">Leakage audit</h2>
            <p>
              Overlap between each evaluation split and the training pool. Exact request overlap must be zero: candidate tasks
              whose request text appears in the training or validation pools are discarded while the split is generated.
              Shared templates and graph shapes are expected for the in-distribution and renamed splits; they are what
              the OOD and unseen-domain splits remove. The last column lists held-out-domain tools that leaked into
              training catalogues.
            </p>
          </div>
        </div>
        {leak.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>split</th>
                  <th className="n">tasks</th>
                  <th className="n">request overlap</th>
                  <th className="n">dropped for overlap</th>
                  <th className="n">shared templates</th>
                  <th className="n">graph-shape overlap</th>
                  <th>unseen-domain tools found in train</th>
                </tr>
              </thead>
              <tbody>
                {leak.map(([split, r]) => (
                  <tr key={split}>
                    <td className="mono small nowrap">{split}</td>
                    <td className="n">{r.n}</td>
                    <td className="n">
                      {r.request_overlap === 0 ? (
                        <span className="chip chip-good">
                          <span aria-hidden="true">✓</span> 0
                        </span>
                      ) : isNum(r.request_overlap) ? (
                        <span className="chip chip-critical">{r.request_overlap}</span>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td className="n">{r.dropped_for_request_overlap ?? "—"}</td>
                    <td className="n">{r.template_overlap ?? "—"}</td>
                    <td className="n">
                      {isNum(r.graph_signature_overlap_frac) ? `${(r.graph_signature_overlap_frac * 100).toFixed(0)}%` : "—"}
                    </td>
                    <td>
                      {r.unseen_tools_in_train === undefined ? (
                        "—"
                      ) : r.unseen_tools_in_train.length ? (
                        <span className="chip chip-critical">{r.unseen_tools_in_train.join(", ")}</span>
                      ) : (
                        <span className="muted small">none</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty">
            <code>leakage.json</code> has not been exported yet.
          </div>
        )}
      </section>

      <section className="block" aria-labelledby="h-teacher">
        <div className="block-head">
          <div>
            <h2 id="h-teacher">Teacher generation (config B data)</h2>
            <p>
              How the Qwen3.5-4B teacher&apos;s demonstrations were produced and filtered: decoding settings, acceptance
              counts, throughput and cost.
            </p>
          </div>
        </div>
        {teacher ? (
          <div className="card">
            <GenericKV obj={teacher} />
          </div>
        ) : (
          <div className="empty">
            <strong>Teacher generation has not finished yet.</strong> Its manifest (
            <code>data/v1/teacher/manifest.json</code>) is written when <code>scripts/distill.py</code> completes and is
            then exported here.
          </div>
        )}
      </section>
    </>
  );
}
