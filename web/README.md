# cognitive-core-lab — research showcase

A Next.js (App Router, TypeScript) site that presents the cognitive-core-lab experiments: whether LoRA
fine-tuning improves tool selection and multi-step orchestration in Qwen2.5-0.5B-Instruct.

| Page | What it shows |
|---|---|
| `/` Overview | Research question, configurations, headline success-by-split chart (95% CIs), caveats |
| `/traces` Trace explorer | Shared task list; request, role, tool catalogue, gold dependency graph; turn-by-turn trace (raw output, parsed calls, outcomes, observations, final answer, validator checks); side-by-side comparison; **Replay** and **Live** modes |
| `/benchmarks` | Success by split, diagnostics, paired comparisons (CI + p), breakdowns, catalogue-size ablation, efficiency, training curves |
| `/failures` | Failure-mode counts per configuration and example traces |
| `/methods` | `report.md` (when present), pipeline, dataset hashes, leakage audit, teacher manifest |

Every number and trace is read from `public/data/*.json`, which `scripts/export_web.py` writes from real run
outputs. Nothing is mocked or hard-coded. Configurations that have not been evaluated yet show "no results yet";
missing optional files (`teacher.json`, `report.md`, …) show an explanatory empty state.

## Run

Requires Node ≥ 20.9 (tested with Node 22).

```bash
cd web
npm ci            # or: npm install
npm run dev       # http://localhost:3000
```

Production build:

```bash
npm run build
npm start         # PORT=4000 npm start to change the port
```

`npm run typecheck` runs `tsc --noEmit`.

## Refreshing the data

From the repository root, after new evaluation / training runs (and `scripts/analyze.py`):

```bash
python scripts/analyze.py      # results/summary.json
python scripts/export_web.py   # -> web/public/data/
```

Pages read `public/data` from disk on every request, so a running `npm start` or `npm run dev` picks up the new
files on the next page load — no rebuild needed.

Files consumed: `manifest.json`, `results.json`, `showcase.json`, `failures.json`, `training.json`, and optionally
`teacher.json`, `datasets.json`, `leakage.json`, `report.md`. Python `NaN`/`Infinity` literals are tolerated and shown as "—".

## Replay vs live

* **Replay** (default) renders recorded traces from `showcase.json`. Each trace is badged
  **"REPLAY — recorded trace from evaluation run `<source_run>`"** (e.g. `B_s0`). No Python process is needed.
* **Live** runs a configuration on this machine. The browser calls the Next.js route handler `app/api/run/route.ts`,
  which proxies to the local Python server and badges results **"LIVE — local inference on this machine"**.
  If the server is not reachable the UI says so and shows how to start it — it never falls back to replay.

Start the inference server from the repository root (Apple silicon / MLX; one model resident at a time):

```bash
source .venv/bin/activate
python scripts/serve.py --port 8765
```

The proxy target is `process.env.CCL_SERVER_URL` (default `http://127.0.0.1:8765`):

```bash
CCL_SERVER_URL=http://127.0.0.1:9000 npm run dev
```

Proxy API (all errors are JSON `{error, message, hint?, server_url?}`):

| Request | Upstream | Notes |
|---|---|---|
| `GET /api/run` | `GET /health` | 503 `live_server_unreachable` when the server is down |
| `GET /api/run?tasks=eval_ood` | `GET /tasks?split=eval_ood` | task picker for "any evaluation task" |
| `POST /api/run` `{config, split, index, request?, approval}` | `POST /run` | `approval`: `task` \| `approve` \| `deny`; `request` ≤ 500 chars; 504 after 290 s |

A custom request keeps the task's catalogue and simulated world but clears the gold workflow, so the run is not scored.

## Dependencies

Pinned exactly (`.npmrc` sets `save-exact`): `next@16.4.3`, `react@19.3.1`, `react-dom@19.3.1`,
`typescript@5.9.3`, `@types/react@19.3.0`, `@types/react-dom@19.3.0`, `@types/node@22.20.5`.
Charts are hand-written SVG/CSS and the Markdown renderer is a small in-repo module (`lib/markdown.tsx`), so there
are no other runtime dependencies.

## Layout

```
app/                  routes (server components read public/data via lib/data.ts)
  api/run/route.ts    live-mode proxy to scripts/serve.py
components/           DotPlot, GoldDag, TraceView, TraceExplorer, LivePanel, FailureGallery, LossChart, …
lib/                  data loading, types, configuration identity/colours/formatting, markdown
public/data/          exported artefacts (written by scripts/export_web.py — do not edit by hand)
```
