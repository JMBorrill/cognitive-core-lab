import type { Metadata } from "next";
import { loadData } from "@/lib/data";
import type { CI, Comparison } from "@/lib/types";
import {
  SPLIT_ORDER,
  SPLIT_SHORT,
  byRkey,
  configList,
  isNum,
  num,
  pValue,
  pp,
  runToConfigKey,
} from "@/lib/configs";
import SuccessBySplit from "@/components/SuccessBySplit";
import DotPlot, { type DotGroup } from "@/components/DotPlot";
import { BreakdownTables, DiagnosticsTable } from "@/components/BenchTables";
import LossChart from "@/components/LossChart";
import MissingData from "@/components/MissingData";

export const metadata: Metadata = { title: "Benchmarks" };

function Section({ id, title, intro, children }: { id: string; title: string; intro?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="block" aria-labelledby={id}>
      <div className="block-head">
        <div>
          <h2 id={id}>{title}</h2>
          {intro ? <p>{intro}</p> : null}
        </div>
      </div>
      {children}
    </section>
  );
}

function Forest({ c, domain }: { c: Comparison; domain: number }) {
  const W = 160;
  const H = 18;
  const x = (v: number) => ((Math.max(-domain, Math.min(domain, v)) + domain) / (2 * domain)) * W;
  if (!isNum(c.diff)) return null;
  const color = isNum(c.lo) && isNum(c.hi) && (c.lo > 0 || c.hi < 0) ? "var(--text)" : "var(--text-3)";
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden="true" style={{ display: "block" }}>
      <line x1={x(0)} x2={x(0)} y1={0} y2={H} stroke="var(--axis)" strokeDasharray="2 2" />
      {isNum(c.lo) && isNum(c.hi) ? (
        <line x1={x(c.lo)} x2={x(c.hi)} y1={H / 2} y2={H / 2} stroke={color} strokeWidth={2} strokeLinecap="round" />
      ) : null}
      <circle cx={x(c.diff)} cy={H / 2} r={4} fill={color} stroke="var(--surface)" strokeWidth={1.5} />
    </svg>
  );
}

export default async function Benchmarks() {
  const { manifest, results, training } = await loadData(["manifest", "results", "training"] as const);
  const all = configList(manifest, results);
  const manifestCfgs = all.filter((c) => c.inManifest);

  if (!results) {
    return (
      <>
        <header className="page-head">
          <div className="eyebrow">Benchmarks</div>
          <h1>Benchmarks</h1>
        </header>
        <MissingData file="results.json" what="The benchmark tables" />
      </>
    );
  }

  // Diagnostics data (serialisable subset).
  const diagSplits = [...SPLIT_ORDER, "all_main"];
  const diag: Record<string, Record<string, Record<string, CI>>> = {};
  for (const c of all) {
    if (!results.configs?.[c.rkey]) continue;
    diag[c.rkey] = {};
    for (const s of diagSplits) {
      const block = results.configs[c.rkey][s];
      if (block && typeof block === "object") diag[c.rkey][s] = block as Record<string, CI>;
    }
  }

  // Paired comparisons.
  const comps = Object.entries(results.comparisons ?? {}).map(([k, v]) => {
    const [name, split] = k.split("|");
    const [b, a] = name.split("_vs_");
    return { key: k, a, b, split, v };
  });
  comps.sort(
    (p, q) =>
      `${p.a}|${p.b}`.localeCompare(`${q.a}|${q.b}`) || SPLIT_ORDER.indexOf(p.split) - SPLIT_ORDER.indexOf(q.split),
  );
  const domainRaw = Math.max(0.05, ...comps.flatMap((c) => [c.v.lo, c.v.hi, c.v.diff].filter(isNum).map(Math.abs)));
  const domain = Math.min(1, Math.ceil(domainRaw * 10) / 10);
  const cfgName = (rkey: string) => byRkey(all, rkey)?.short ?? rkey;

  // Catalogue-size ablation.
  const cat = results.ablations?.catalog_size ?? {};
  const catKs = [...new Set(Object.values(cat).flatMap((r) => Object.keys(r)))].sort((a, b) => Number(a) - Number(b));
  const catGroups: DotGroup[] = catKs.map((k) => ({
    id: k,
    title: `${k} distractor tool${k === "1" ? "" : "s"}`,
    subtitle: `eval_catalog_${k}`,
    rows: all
      .filter((c) => c.inManifest || cat[c.rkey])
      .map((c) => ({ id: c.key, label: c.label, color: c.color, ci: cat[c.rkey]?.[k], empty: "no results yet" })),
  }));

  // Efficiency.
  const eff = Object.entries(results.efficiency ?? {});

  // Training curves.
  const runs = Object.entries(training ?? {}).filter(([, r]) => r.log?.length);
  const yMax =
    Math.ceil(
      Math.max(
        0.1,
        ...runs.flatMap(([, r]) => r.log.flatMap((x) => [x.loss, x.val_loss].filter(isNum))),
      ) * 10,
    ) / 10;

  return (
    <>
      <header className="page-head">
        <div className="eyebrow">Benchmarks</div>
        <h1>Every metric, with its uncertainty</h1>
        <p className="lede">
          All numbers come from <code>results.json</code> (produced by <code>scripts/analyze.py</code>). Intervals are
          95% bootstrap CIs; configurations trained with several seeds use a hierarchical bootstrap that resamples
          seeds, then tasks. Configurations that have not been evaluated show <em>no results yet</em>.
        </p>
      </header>

      <Section
        id="h-success"
        title="Workflow success by split"
        intro="Episode-level success: every validator check passes. Pooled = all four main splits together."
      >
        <div className="card">
          <SuccessBySplit manifest={manifest} results={results} configs={all} includePooled />
        </div>
      </Section>

      <Section
        id="h-diag"
        title="Diagnostics"
        intro="Where episodes go wrong. Percentages are per-episode rates; counts are means per episode. Each cell shows mean and [95% CI]."
      >
        <DiagnosticsTable
          configs={all.map(({ key, rkey, label, color }) => ({ key, rkey, label, color }))}
          data={diag}
          splits={diagSplits}
        />
      </Section>

      <Section
        id="h-comp"
        title="Paired comparisons"
        intro={
          <>
            Difference in success rate (percentage points), computed per task on identical tasks (seed-averaged), with a
            bootstrap 95% CI and a two-sided sign-flip permutation p-value. Positive means the first configuration is
            better. Dark marks = CI excludes zero. No correction for multiple comparisons is applied.
          </>
        }
      >
        {comps.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>comparison</th>
                  <th>split</th>
                  <th className="n">Δ success (pp)</th>
                  <th className="n">95% CI</th>
                  <th>
                    −{Math.round(domain * 100)} · 0 · +{Math.round(domain * 100)} pp
                  </th>
                  <th className="n">p</th>
                  <th className="n">n tasks</th>
                </tr>
              </thead>
              <tbody>
                {comps.map((c) => (
                  <tr key={c.key}>
                    <td className="nowrap">
                      <strong>{cfgName(c.b)}</strong> vs {cfgName(c.a)}
                    </td>
                    <td className="nowrap">{SPLIT_SHORT[c.split] ?? c.split}</td>
                    <td className="n">{pp(c.v.diff)}</td>
                    <td className="n">
                      [{pp(c.v.lo)}, {pp(c.v.hi)}]
                    </td>
                    <td>
                      <Forest c={c.v} domain={domain} />
                    </td>
                    <td className="n">{pValue(c.v.p)}</td>
                    <td className="n">{c.v.n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty">
            <strong>No paired comparisons yet.</strong> They are computed against the zero-shot baseline (A) and between
            trained configurations once at least two of them have been evaluated.
          </div>
        )}
      </Section>

      <Section
        id="h-break"
        title="Breakdowns"
        intro="Success pooled over the four main splits and averaged over seeds, sliced by task property. No intervals here — small cells are noisy, so read them with their n."
      >
        {Object.keys(results.breakdowns ?? {}).length ? (
          <BreakdownTables
            configs={all.map(({ key, rkey, label, color }) => ({ key, rkey, label, color }))}
            data={results.breakdowns}
          />
        ) : (
          <div className="empty">No breakdowns yet.</div>
        )}
      </Section>

      <Section
        id="h-cat"
        title="Catalogue-size ablation"
        intro="The same tasks with k irrelevant tools added to the catalogue. A model that selects tools by reading descriptions should be flat; one that pattern-matches will degrade as k grows."
      >
        {catGroups.length ? (
          <div className="card">
            <DotPlot groups={catGroups} caption="Success by number of distractor tools, 95% CI" />
          </div>
        ) : (
          <div className="empty">No catalogue-size ablation results yet.</div>
        )}
      </Section>

      <Section
        id="h-eff"
        title="Efficiency"
        intro="Per evaluation run, measured on the local machine (Apple silicon, MLX). Latency is end-to-end per episode, including every turn."
      >
        {eff.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>run</th>
                  <th>configuration</th>
                  <th className="n">episodes</th>
                  <th className="n">mean turns</th>
                  <th className="n">latency / episode (s)</th>
                  <th className="n">gen tok/s</th>
                  <th className="n">total tok/s</th>
                  <th className="n">gen tok / episode</th>
                  <th className="n">peak mem (GB)</th>
                  <th className="n">wall time (s)</th>
                </tr>
              </thead>
              <tbody>
                {eff.map(([run, e]) => {
                  const c = byRkey(all, runToConfigKey(run));
                  return (
                    <tr key={run}>
                      <td className="mono nowrap">{run}</td>
                      <td className="cfg">
                        <span className="row" style={{ flexWrap: "nowrap", gap: 6 }}>
                          {c ? <span className="swatch" style={{ background: c.color }} aria-hidden="true" /> : null}
                          {c?.label ?? "—"}
                        </span>
                        {c?.key === "oracle" ? <div className="tiny muted">programmatic — no model inference</div> : null}
                      </td>
                      <td className="n">{e.episodes ?? "—"}</td>
                      <td className="n">{num(e.mean_turns)}</td>
                      <td className="n">{num(e.mean_episode_latency_s)}</td>
                      <td className="n">{num(e.gen_tokens_per_s, 0)}</td>
                      <td className="n">{num(e.total_tokens_per_s, 0)}</td>
                      <td className="n">{num(e.mean_gen_tokens_per_episode, 1)}</td>
                      <td className="n">{num(e.peak_mem_gb)}</td>
                      <td className="n">{num(e.wall_time_s, 1)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty">No efficiency measurements yet.</div>
        )}
      </Section>

      <Section
        id="h-train"
        title="Training curves"
        intro="LoRA training loss per step (line) and validation loss at evaluation checkpoints (dots), one panel per run, on a shared y-axis."
      >
        {runs.length ? (
          <>
            <div className="grid grid-3">
              {runs.map(([name, r]) => {
                const c = byRkey(all, runToConfigKey(name)) ?? manifestCfgs.find((m) => m.key === runToConfigKey(name));
                const s = r.summary ?? {};
                return (
                  <div key={name} className="card stack">
                    <div className="row">
                      <span className="swatch" style={{ background: c?.color ?? "var(--series-other)" }} aria-hidden="true" />
                      <h3 className="mono" style={{ fontSize: "0.9rem" }}>
                        {name}
                      </h3>
                      <span className="muted small">{c?.short ?? ""}</span>
                    </div>
                    <LossChart log={r.log} color={c?.color ?? "var(--series-other)"} yMax={yMax} title={name} />
                    <dl className="kv tiny">
                      {(
                        [
                          ["steps", s.steps],
                          ["trainable params", s.trainable_params],
                          ["train time (s)", isNum(s.train_time_s) ? s.train_time_s.toFixed(0) : undefined],
                          ["final val loss", isNum(s.final_val_loss) ? s.final_val_loss.toFixed(4) : undefined],
                          ["peak mem (GB)", isNum(s.peak_mem_gb) ? s.peak_mem_gb.toFixed(2) : undefined],
                          ["adapter sha256", typeof s.adapter_sha256 === "string" ? s.adapter_sha256.slice(0, 16) + "…" : undefined],
                        ] as [string, unknown][]
                      )
                        .filter(([, v]) => v !== undefined && v !== null)
                        .map(([k, v]) => (
                          <div key={k} style={{ display: "contents" }}>
                            <dt>{k}</dt>
                            <dd className="num mono">{String(v)}</dd>
                          </div>
                        ))}
                    </dl>
                  </div>
                );
              })}
            </div>
            <div className="chart-legend">
              <span>
                <svg width="22" height="8" aria-hidden="true">
                  <line x1="0" x2="22" y1="4" y2="4" stroke="var(--text-2)" strokeWidth="2" />
                </svg>
                training loss (colour = configuration)
              </span>
              <span>
                <svg width="10" height="10" aria-hidden="true">
                  <circle cx="5" cy="5" r="3.5" fill="var(--surface)" stroke="var(--text)" strokeWidth="1.5" />
                </svg>
                validation loss
              </span>
            </div>
          </>
        ) : (
          <div className="empty">
            <strong>No training runs exported yet.</strong> Curves appear here from <code>runs/&lt;config&gt;_s&lt;seed&gt;/train_log.jsonl</code>{" "}
            once LoRA training has run and <code>scripts/export_web.py</code> is re-run.
          </div>
        )}
      </Section>
    </>
  );
}
