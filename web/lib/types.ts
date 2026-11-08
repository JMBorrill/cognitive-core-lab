// Types for the files exported by scripts/export_web.py into public/data/.
// Every field that a partially-finished experiment might leave out is optional.

export type CI = {
  mean: number | null;
  lo: number | null;
  hi: number | null;
  n: number;
  seeds?: number;
  seed_means?: (number | null)[];
  seed_sd?: number | null;
};

export type ConfigMeta = {
  label: string;
  description: string;
  has_results: boolean;
  replay_run: string;
};

export type Manifest = {
  generated_at: string;
  git_commit: string | null;
  configs: Record<string, ConfigMeta>;
  splits: Record<string, string>;
};

/** results.configs[cfg] = { seeds: number[], <split>: { <metric>: CI } } */
export type ResultsConfig = { seeds?: number[] } & Record<string, unknown>;

export type Comparison = { diff: number | null; lo: number | null; hi: number | null; p: number | null; n: number };

export type BreakdownCell = { success: number | null; n_tasks: number };

export type Efficiency = {
  peak_mem_gb?: number | null;
  wall_time_s?: number | null;
  episodes?: number | null;
  mean_turns?: number | null;
  gen_tokens_per_s?: number | null;
  total_tokens_per_s?: number | null;
  mean_episode_latency_s?: number | null;
  mean_gen_tokens_per_episode?: number | null;
};

export type Results = {
  configs: Record<string, ResultsConfig>;
  comparisons: Record<string, Comparison>;
  breakdowns: Record<string, Record<string, Record<string, BreakdownCell>>>;
  ablations: { catalog_size?: Record<string, Record<string, CI>> } & Record<string, unknown>;
  efficiency: Record<string, Efficiency>;
};

export type GoldStep = { id: string; tool: string; deps: string[]; optional: boolean; execute: boolean };

export type ShowcaseTask = {
  key: string;
  split: string;
  index: number;
  task_id: string;
  family: string;
  template: string;
  role: string;
  request: string;
  depth: number;
  today: string;
  catalog: string[];
  expected_status: string;
  answer_facts: string[];
  graph_seen: boolean;
  gold: GoldStep[];
};

export type TraceTurn = {
  turn: number;
  raw: string;
  kind: string; // calls | final | invalid
  strict: boolean;
  error: string;
  gen_tokens: number;
  latency_s?: number;
};

export type TraceCall = {
  turn: number;
  tool: string;
  outcome: string; // ok | error | hallucinated | permission_denied | approval_denied | invalid_argument | limit
  args: unknown;
  result: unknown;
  approval: boolean | null;
  tool_key?: string | null;
  canonical_args?: unknown;
};

export type Trace = {
  turns: TraceTurn[];
  calls: TraceCall[];
  final: string | null;
  status: string | null;
  termination: string;
  messages?: { role: string; content: string }[];
};

export type ReplayTrace = Trace & {
  success: boolean;
  checks: Record<string, boolean>;
  failure_mode: string | null;
  source_run: string;
};

export type Showcase = {
  tasks: ShowcaseTask[];
  traces: Record<string, Record<string, ReplayTrace>>;
};

export type FailureExample = {
  split: string;
  index: number;
  task_id: string;
  family: string;
  request: string;
  role: string;
  expected_status: string;
  trace: Trace;
};

export type Failures = Record<
  string,
  { counts: Record<string, number>; total: number; examples: Record<string, FailureExample[]> }
>;

export type TrainingLogRow = {
  step: number;
  loss: number | null;
  val_loss: number | null;
  tok_per_s: number | null;
  peak_mem_gb: number | null;
};

export type Training = Record<string, { log: TrainingLogRow[]; summary: Record<string, unknown> | null }>;

export type DatasetFile = {
  sha256: string;
  n?: number;
  rows?: number;
  episodes?: number;
  families?: Record<string, number>;
  rejected?: Record<string, number>;
};

export type Datasets = {
  version?: string;
  files?: Record<string, DatasetFile>;
  templates?: Record<string, { family: string; split: string; depth: number }>;
};

export type LeakageRow = {
  n: number;
  request_overlap?: number;
  template_overlap?: number;
  graph_signature_overlap_frac?: number;
  unseen_tools_in_train?: string[];
  dropped_for_request_overlap?: number;
};

export type Leakage = Record<string, LeakageRow>;

// ---- live server (scripts/serve.py) ----

export type LiveHealth = { ok: boolean; configs: Record<string, { label: string; available: boolean }> };

export type LiveTaskSummary = {
  index: number;
  task_id: string;
  family: string;
  template: string;
  role: string;
  request: string;
  depth: number;
};

export type LiveRunResponse = {
  mode: "live";
  config: string;
  label: string;
  wall_time_s: number;
  custom_request: boolean;
  trace: Trace & { task_id?: string; model?: string };
  task: LiveTaskSummary & { catalog: unknown[]; expected_status: string };
  validation?: { success: boolean; checks: Record<string, boolean> };
  metrics?: Record<string, unknown>;
};

export type ApiError = { error: string; message: string; hint?: string; server_url?: string; detail?: unknown };
