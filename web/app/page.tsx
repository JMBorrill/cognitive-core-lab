import Link from "next/link";
import { loadData } from "@/lib/data";
import { configList, hasResults, seedCount } from "@/lib/configs";
import SuccessBySplit from "@/components/SuccessBySplit";
import MissingData from "@/components/MissingData";

const GROUPS: { title: string; keys: string[]; blurb: string }[] = [
  { title: "Baselines (no weight change)", keys: ["A", "A_fewshot"], blurb: "What the 0.5B model does out of the box, with and without worked examples in the prompt." },
  { title: "LoRA on teacher demonstrations", keys: ["B"], blurb: "Distillation: trajectories from a 4B teacher, kept only if they execute and validate." },
  { title: "LoRA on independent trajectories", keys: ["C"], blurb: "Trajectories produced by a programmatic policy, also execution-verified — no stronger model involved." },
  { title: "Combined", keys: ["D", "Dseq"], blurb: "Both data sources: shuffled together, or one after the other." },
  { title: "References", keys: ["teacher", "oracle"], blurb: "Not candidates — context for how far the student is from a larger model and from the ceiling." },
];

export default async function Overview() {
  const { manifest, results } = await loadData(["manifest", "results"] as const);
  const configs = configList(manifest);
  const withResults = configs.filter((c) => hasResults(results, c.rkey));
  const grouped = new Set(GROUPS.flatMap((g) => g.keys));
  const ungrouped = configs.filter((c) => !grouped.has(c.key));

  return (
    <>
      <header className="page-head">
        <div className="eyebrow">Research question</div>
        <h1>Can LoRA fine-tuning teach a 0.5B model to choose and orchestrate tools?</h1>
        <p className="lede">
          Small models are cheap enough to run an agent loop locally, but they tend to pick the wrong tool, invent
          arguments and lose track of multi-step plans. This project measures whether low-rank fine-tuning of{" "}
          <strong>Qwen2.5-0.5B-Instruct</strong> fixes that — and whether it matters <em>where</em> the training
          trajectories come from — on identical held-out tasks, with confidence intervals, under a strict runtime.
        </p>
        <div className="row" style={{ marginTop: 14 }}>
          <Link className="btn btn-primary" href="/traces">
            Explore traces
          </Link>
          <Link className="btn" href="/benchmarks">
            Benchmarks
          </Link>
          <Link className="btn" href="/methods">
            Methods &amp; data
          </Link>
        </div>
      </header>

      {manifest ? (
        <div className="card grid grid-4" style={{ gap: 16 }}>
          <div className="stat">
            <span className="v num">
              {withResults.length}/{configs.length}
            </span>
            <span className="k">configurations with results</span>
          </div>
          <div className="stat">
            <span className="v num">{Object.keys(manifest.splits).length}</span>
            <span className="k">held-out evaluation splits</span>
          </div>
          <div className="stat">
            <span className="v" style={{ fontSize: "1rem", paddingTop: 6 }}>
              {new Date(manifest.generated_at).toISOString().replace("T", " ").slice(0, 16)} UTC
            </span>
            <span className="k">data exported</span>
          </div>
          <div className="stat">
            <span className="v mono" style={{ fontSize: "1rem", paddingTop: 6 }}>
              {manifest.git_commit ?? "—"}
            </span>
            <span className="k">git commit of export</span>
          </div>
        </div>
      ) : (
        <MissingData file="manifest.json" what="The overview" />
      )}

      <section className="block" aria-labelledby="h-headline">
        <div className="block-head">
          <div>
            <h2 id="h-headline">Workflow success by evaluation split</h2>
            <p>
              Share of episodes where every validator check passes (right calls, right arguments, safe ordering, right
              side effects, right final status and answer). Dots are means; whiskers are 95% bootstrap confidence
              intervals over tasks (and over training seeds for LoRA configurations).
            </p>
          </div>
          <Link href="/benchmarks" className="small">
            All metrics →
          </Link>
        </div>
        <div className="card">
          {results ? (
            <SuccessBySplit manifest={manifest} results={results} configs={configs} />
          ) : (
            <MissingData file="results.json" what="The headline chart" />
          )}
        </div>
        {withResults.length && withResults.every((c) => c.key === "oracle") ? (
          <p className="note note-warn" style={{ marginTop: 10 }}>
            Only the programmatic oracle has been evaluated so far. It reads the gold task graph, so its 100% is a
            check that every task is solvable and the validator is consistent — not a model result. Model rows will
            fill in as evaluation runs finish.
          </p>
        ) : null}
      </section>

      <section className="block" aria-labelledby="h-configs">
        <div className="block-head">
          <div>
            <h2 id="h-configs">Configurations compared</h2>
            <p>
              All student configurations share the same base weights, system prompt, tool-catalogue rendering, decoding
              settings and runtime limits. Only the LoRA adapter (or the prompt, for A+) differs.
            </p>
          </div>
        </div>
        <div className="stack" style={{ gap: 18 }}>
          {[...GROUPS, ...(ungrouped.length ? [{ title: "Other", keys: ungrouped.map((c) => c.key), blurb: "" }] : [])].map(
            (g) => {
              const items = configs.filter((c) => g.keys.includes(c.key));
              if (!items.length) return null;
              return (
                <div key={g.title}>
                  <h3 style={{ marginBottom: 2 }}>{g.title}</h3>
                  {g.blurb ? (
                    <p className="muted small" style={{ marginBottom: 8 }}>
                      {g.blurb}
                    </p>
                  ) : null}
                  <div className="grid grid-3">
                    {items.map((c) => {
                      const has = hasResults(results, c.rkey);
                      const seeds = seedCount(results, c.rkey);
                      return (
                        <article
                          key={c.key}
                          className="card cfg-card"
                          style={{ ["--cfg-color" as string]: c.color }}
                        >
                          <div className="row" style={{ justifyContent: "space-between" }}>
                            <h3>{c.label}</h3>
                          </div>
                          <p>{c.description}</p>
                          <div className="row" style={{ marginTop: "auto", paddingTop: 6 }}>
                            {has ? (
                              <span className="chip chip-good">
                                <span aria-hidden="true">✓</span> results
                                {seeds ? ` · ${seeds} seed${seeds > 1 ? "s" : ""}` : ""}
                              </span>
                            ) : (
                              <span className="chip chip-outline">no results yet</span>
                            )}
                            <span className="chip mono" title="evaluation run used for replayed traces">
                              {manifest?.configs[c.key]?.replay_run ?? c.rkey}
                            </span>
                          </div>
                        </article>
                      );
                    })}
                  </div>
                </div>
              );
            },
          )}
        </div>
      </section>

      <section className="block" aria-labelledby="h-protocol">
        <div className="block-head">
          <div>
            <h2 id="h-protocol">How an episode runs</h2>
            <p>
              The model sees a request, the user&apos;s role and a tool catalogue, and must reply with exactly one JSON
              action per turn. A runtime parses it, enforces permissions and human approval, executes calls on
              deterministic simulators and returns observations until the model finishes.
            </p>
          </div>
        </div>
        <div className="grid grid-3">
          <div className="card stack">
            <h3>1 · Call tools</h3>
            <pre className="codeblock">{`{"calls": [
  {"tool": "find_contact", "args": {"name": "…"}}
]}`}</pre>
            <p className="small muted" style={{ margin: 0 }}>
              Several calls in one list are independent and run in parallel. Unknown tools, bad arguments, missing
              roles and denied approvals come back as error observations.
            </p>
          </div>
          <div className="card stack">
            <h3>2 · Read observations</h3>
            <pre className="codeblock">{`{"tool": "find_contact",
 "result": {…}}
{"error": "approval_denied", …}`}</pre>
            <p className="small muted" style={{ margin: 0 }}>
              Later arguments must come from earlier observations — inventing an identifier counts as a dependency
              violation even if it happens to be right.
            </p>
          </div>
          <div className="card stack">
            <h3>3 · Finish</h3>
            <pre className="codeblock">{`{"final": "…",
 "status": "done | refused |
            blocked | failed"}`}</pre>
            <p className="small muted" style={{ margin: 0 }}>
              The status matters: refusing a forbidden action or reporting a blocked approval is the correct outcome
              for some tasks.
            </p>
          </div>
        </div>
      </section>

      {manifest ? (
        <section className="block" aria-labelledby="h-splits">
          <div className="block-head">
            <div>
              <h2 id="h-splits">Evaluation splits</h2>
              <p>Each split probes a different kind of generalisation beyond the training distribution.</p>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>split</th>
                  <th>what it tests</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(manifest.splits).map(([k, v]) => (
                  <tr key={k}>
                    <td className="mono nowrap">{k}</td>
                    <td>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <section className="block" aria-labelledby="h-caveats">
        <div className="block-head">
          <div>
            <h2 id="h-caveats">Read the numbers with these caveats</h2>
          </div>
        </div>
        <div className="card">
          <ul className="stack" style={{ margin: 0, paddingLeft: "1.2em" }}>
            <li>
              <strong>Synthetic, deterministic tools.</strong> Tasks run against simulators, not real APIs. That makes
              results exactly reproducible and every episode checkable, but real tools are messier.
            </li>
            <li>
              <strong>The oracle is a ceiling, not a model.</strong> It reads the gold task graph. Its role is to show
              that every task is solvable under the validator.
            </li>
            <li>
              <strong>The teacher is filtered.</strong> Configuration B only trains on teacher episodes that executed
              and validated, so its data is skewed towards tasks the teacher already solves.
            </li>
            <li>
              <strong>Intervals are about tasks and seeds, not the world.</strong> The 95% intervals resample the
              evaluation tasks (and training seeds where there are several). They do not cover other prompts, other
              base models or other tool ecosystems.
            </li>
            <li>
              <strong>Some slices are small.</strong> Breakdown cells (e.g. deep workflows or rare families) can contain
              only a handful of tasks — always check the <em>n</em> shown next to each number.
            </li>
            <li>
              <strong>One base model.</strong> Every student is Qwen2.5-0.5B-Instruct; conclusions may not transfer to
              other sizes or families.
            </li>
          </ul>
        </div>
      </section>
    </>
  );
}
