// Client-safe helpers: configuration identity (labels, colours, results keys) and formatting.

import type { CI, Manifest, Results } from "./types";

/** Fixed categorical slot order — colour follows the configuration, never its rank. */
const SLOT_ORDER = ["A", "A_fewshot", "B", "C", "D", "Dseq", "teacher", "oracle"];

export const SPLIT_ORDER = ["eval_id", "eval_ood", "eval_unseen", "eval_renamed"];

export const SPLIT_SHORT: Record<string, string> = {
  eval_id: "In-distribution",
  eval_ood: "Unseen templates",
  eval_unseen: "Unseen tool domains",
  eval_renamed: "Renamed tools",
  all_main: "All four splits",
};

/** Single-seed ablations that exist only in results.json (definitions: docs/REPORT.md §3). */
const ABLATIONS: Record<string, { label: string; description: string }> = {
  C_matched: {
    label: "C_matched · Oracle on B's tasks",
    description: "Ablation (1 seed): oracle trajectories on exactly the 1,000 tasks used for B, isolating label source from task coverage.",
  },
  C_25: { label: "C_25 · Oracle, 250 episodes", description: "Data-quantity ablation (1 seed): 250 oracle episodes, same step budget." },
  C_50: { label: "C_50 · Oracle, 500 episodes", description: "Data-quantity ablation (1 seed): 500 oracle episodes, same step budget." },
  C_full: { label: "C_full · Oracle, 3,000 episodes", description: "Data-quantity ablation (1 seed): 3,000 oracle episodes, same step budget." },
};

export type ConfigInfo = {
  key: string; // manifest key (A, D, ...) or a results-only key
  rkey: string; // key used in results.json / efficiency / comparisons (A_base, Dmix, ...)
  label: string;
  short: string;
  description: string;
  color: string; // CSS colour expression
  inManifest: boolean;
};

export function shortLabel(label: string, key: string): string {
  if (label.includes(" · ")) return label.split(" · ")[0].trim();
  const paren = label.indexOf(" (");
  if (paren > 0) return label.slice(0, paren).trim();
  return label || key;
}

export function runToConfigKey(run: string): string {
  return run.replace(/_s\d+$/, "");
}

/**
 * All configurations: manifest entries first (in manifest order), then any extra configs that
 * only appear in results.json (e.g. data-size ablations), labelled by their raw key.
 */
export function configList(manifest: Manifest | null, results?: Results | null): ConfigInfo[] {
  const out: ConfigInfo[] = [];
  const seen = new Set<string>();
  if (manifest) {
    for (const [key, meta] of Object.entries(manifest.configs)) {
      const slot = SLOT_ORDER.indexOf(key);
      const rkey = runToConfigKey(meta.replay_run || key);
      seen.add(rkey);
      out.push({
        key,
        rkey,
        label: meta.label || key,
        short: shortLabel(meta.label || key, key),
        description: meta.description,
        color: slot >= 0 ? `var(--series-${slot + 1})` : "var(--series-other)",
        inManifest: true,
      });
    }
  }
  if (results) {
    for (const rkey of Object.keys(results.configs ?? {})) {
      if (seen.has(rkey)) continue;
      const ab = ABLATIONS[rkey];
      out.push({
        key: rkey,
        rkey,
        label: ab?.label ?? rkey,
        short: rkey,
        description: ab?.description ?? "Additional run present in results.json but not described in the manifest.",
        color: "var(--series-other)",
        inManifest: false,
      });
    }
  }
  return out;
}

export function byRkey(list: ConfigInfo[], rkey: string): ConfigInfo | undefined {
  return list.find((c) => c.rkey === rkey);
}

function isCI(x: unknown): x is CI {
  return typeof x === "object" && x !== null && "mean" in x && "lo" in x && "hi" in x;
}

export function getCI(results: Results | null, rkey: string, split: string, metric: string): CI | undefined {
  const cfg = results?.configs?.[rkey];
  if (!cfg) return undefined;
  const block = cfg[split];
  if (typeof block !== "object" || block === null) return undefined;
  const ci = (block as Record<string, unknown>)[metric];
  return isCI(ci) ? ci : undefined;
}

export function hasResults(results: Results | null, rkey: string): boolean {
  return Boolean(results?.configs?.[rkey]);
}

export function seedCount(results: Results | null, rkey: string): number | undefined {
  const s = results?.configs?.[rkey]?.seeds;
  return Array.isArray(s) ? s.length : undefined;
}

// ---- formatting ----

export function isNum(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}

export function pct(x: number | null | undefined, digits = 1): string {
  return isNum(x) ? `${(x * 100).toFixed(digits)}%` : "—";
}

export function num(x: number | null | undefined, digits = 2): string {
  return isNum(x) ? x.toFixed(digits) : "—";
}

export function ciText(ci: CI | undefined, asPct = true, digits = 1): string {
  if (!ci || !isNum(ci.mean)) return "—";
  const k = asPct ? 100 : 1;
  const f = (v: number | null) => (isNum(v) ? (v * k).toFixed(digits) : "?");
  return `${f(ci.mean)} [${f(ci.lo)}, ${f(ci.hi)}]`;
}

export function pp(x: number | null | undefined, digits = 1): string {
  if (!isNum(x)) return "—";
  const v = x * 100;
  return `${v > 0 ? "+" : v < 0 ? "−" : "±"}${Math.abs(v).toFixed(digits)}`;
}

export function pValue(p: number | null | undefined): string {
  if (!isNum(p)) return "—";
  if (p < 0.001) return "<0.001";
  return p.toFixed(3);
}

export function humanize(s: string): string {
  return s.replace(/_/g, " ");
}

export const FAMILY_HELP: Record<string, string> = {
  select: "pick the one right tool among distractors",
  args: "fill arguments correctly (formats, enums, units)",
  chain: "sequential calls where later arguments come from earlier observations",
  parallel: "independent calls that should be issued together",
  recovery: "tool failures, retries and fallbacks",
  permission: "requests the user's role is not allowed to perform",
  approval: "consequential actions gated by a human approver",
  termination: "answer directly or stop without calling tools",
};

export const FAILURE_HELP: Record<string, string> = {
  malformed_output: "At least one turn could not be parsed as a valid JSON action.",
  hallucinated_tool: "Called a tool that is not in the catalogue.",
  permission_violation: "Called a tool the user's role is not permitted to use.",
  retried_after_denial: "Retried a consequential action after a human denied it.",
  dependency_violation: "Used an identifier before observing it, or called a dependent step before its prerequisite returned.",
  missing_or_wrong_call: "A required gold call was never made with matching arguments.",
  wrong_side_effect: "Executed mutations do not match the expected side effects one-to-one.",
  wrong_final_status: "Finished with the wrong status (done / refused / blocked / failed).",
  wrong_answer: "Final answer is missing required facts from the observations.",
  other: "Failed a validator check not covered by the categories above.",
};
