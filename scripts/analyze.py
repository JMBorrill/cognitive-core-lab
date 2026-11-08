"""Aggregate evaluation runs into results/summary.json and results/tables.md.

Run names follow <config>_s<seed> for trained models (seeds are pooled with a
hierarchical bootstrap) and a bare <config> for single runs.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccl.stats import hierarchical_ci, paired_diff  # noqa: E402

RUNS = Path("results/runs")
MAIN_SPLITS = ["eval_id", "eval_ood", "eval_unseen", "eval_renamed"]
METRICS = ["success", "tool_selection", "arg_correct", "dependency_safe", "strict_schema",
           "hallucinated_tools", "unnecessary_calls", "invalid_args", "parse_errors", "turns", "n_calls"]
ORDER = ["A_base", "A_fewshot", "B", "C", "Dmix", "Dseq", "C_matched", "C_25", "C_50", "C_full", "teacher", "oracle"]


def load_runs() -> dict[str, dict[int, list[dict]]]:
    out: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    for d in sorted(RUNS.iterdir()):
        f = d / "metrics.jsonl"
        if not f.exists():
            continue
        m = re.fullmatch(r"(.+)_s(\d+)", d.name)
        cfg, seed = (m.group(1), int(m.group(2))) if m else (d.name, 0)
        out[cfg][seed] = [json.loads(l) for l in open(f)]
    return out


def aligned(rows_by_seed: dict[int, list[dict]], split: str, metric: str, where=None):
    """Per-seed lists of metric values over the same ordered task IDs."""
    per = {s: {r["task_id"]: r for r in rows if r["split"] == split and (where is None or where(r))}
           for s, rows in rows_by_seed.items()}
    common = sorted(set.intersection(*(set(v) for v in per.values()))) if per else []
    return common, [[float(per[s][t][metric]) for t in common] for s in sorted(per)]


def seed_mean(rows_by_seed, split, metric, where=None):
    ids, vals = aligned(rows_by_seed, split, metric, where)
    return ids, np.nanmean(np.array(vals), axis=0) if vals and ids else np.array([])


def main():
    runs = load_runs()
    cfgs = [c for c in ORDER if c in runs] + sorted(c for c in runs if c not in ORDER)
    summary = {"configs": {}, "comparisons": {}, "breakdowns": {}, "ablations": {}, "efficiency": {}}

    for c in cfgs:
        summary["configs"][c] = {"seeds": sorted(runs[c])}
        for split in MAIN_SPLITS + ["all_main"]:
            res = {}
            for met in METRICS:
                if split == "all_main":
                    vals = [sum((aligned(runs[c], s, met)[1][i] for s in MAIN_SPLITS), [])
                            for i in range(len(runs[c]))]
                else:
                    vals = aligned(runs[c], split, met)[1]
                if vals and len(vals[0]):
                    res[met] = hierarchical_ci(vals)
            summary["configs"][c][split] = res

    # Paired comparisons (seed-averaged per task) against the baseline and between trained configs.
    pairs = [("A_base", x) for x in cfgs if x != "A_base"] + [("B", "C"), ("B", "Dmix"), ("C", "Dmix"),
                                                               ("Dmix", "Dseq"), ("C", "C_matched"), ("B", "C_matched")]
    for a, b in pairs:
        if a not in runs or b not in runs:
            continue
        for split in MAIN_SPLITS:
            ia, va = seed_mean(runs[a], split, "success")
            ib, vb = seed_mean(runs[b], split, "success")
            common = sorted(set(ia) & set(ib))
            if not common:
                continue
            xa = np.array([va[ia.index(t)] for t in common])
            xb = np.array([vb[ib.index(t)] for t in common])
            summary["comparisons"][f"{b}_vs_{a}|{split}"] = paired_diff(xa, xb)

    # Breakdowns: family, depth, graph novelty, faults, renamed (pooled across main splits, seed-averaged).
    for c in cfgs:
        bd = defaultdict(dict)
        rows = [r for s in runs[c].values() for r in s if r["split"] in MAIN_SPLITS]
        nseeds = len(runs[c])
        for key in ["family", "depth", "graph_seen", "fault", "split"]:
            groups = defaultdict(list)
            for r in rows:
                groups[str(r[key])].append(r["success"])
            bd[key] = {g: {"success": float(np.mean(v)), "n_tasks": len(v) // nseeds} for g, v in groups.items()}
        summary["breakdowns"][c] = bd

    # Catalogue-size ablation.
    for c in cfgs:
        res = {}
        for k in (0, 5, 12):
            vals = aligned(runs[c], f"eval_catalog_{k}", "success")[1]
            if vals and len(vals[0]):
                res[str(k)] = hierarchical_ci(vals)
        if res:
            summary["ablations"].setdefault("catalog_size", {})[c] = res

    # Efficiency: latency/throughput from run.json.
    for d in sorted(RUNS.iterdir()):
        rj = d / "run.json"
        if not rj.exists():
            continue
        r = json.loads(rj.read_text())
        st = r.get("mlx_stats") or {}
        rows = [json.loads(l) for l in open(d / "metrics.jsonl")]
        summary["efficiency"][d.name] = {
            "peak_mem_gb": r.get("peak_mem_gb"), "wall_time_s": r.get("wall_time_s"),
            "episodes": len(rows), "mean_turns": float(np.mean([x["turns"] for x in rows])) if rows else None,
            "gen_tokens_per_s": st["gen_tokens"] / st["gen_time_s"] if st.get("gen_time_s") else None,
            "total_tokens_per_s": (st["gen_tokens"] + st["prompt_tokens"]) / st["gen_time_s"]
            if st.get("gen_time_s") else None,
            "mean_episode_latency_s": float(np.mean([x["latency_s"] for x in rows])) if rows else None,
            "mean_gen_tokens_per_episode": float(np.mean([x["gen_tokens"] for x in rows])) if rows else None,
        }

    Path("results").mkdir(exist_ok=True)
    Path("results/summary.json").write_text(json.dumps(summary, indent=1))
    Path("results/tables.md").write_text(render_tables(summary, cfgs))
    print(Path("results/tables.md").read_text())


def fmt(ci, pct=True):
    if not ci:
        return "-"
    k = 100 if pct else 1
    s = f"{ci['mean'] * k:.1f} [{ci['lo'] * k:.1f}, {ci['hi'] * k:.1f}]"
    return s


def render_tables(s, cfgs) -> str:
    out = ["# Results tables (auto-generated by scripts/analyze.py)\n",
           "Values: mean [95% CI]. Trained configs pool seeds via a hierarchical bootstrap "
           "(resample seeds, then tasks).\n"]
    out.append("## Workflow success (%) by split\n")
    out.append("| config | seeds | " + " | ".join(MAIN_SPLITS) + " | all |")
    out.append("|---|---|" + "---|" * (len(MAIN_SPLITS) + 1))
    for c in cfgs:
        row = s["configs"][c]
        out.append(f"| {c} | {len(row['seeds'])} | " + " | ".join(fmt(row[sp].get("success")) for sp in MAIN_SPLITS)
                   + f" | {fmt(row['all_main'].get('success'))} |")
    out.append("\n## Diagnostics (all main splits pooled)\n")
    cols = ["tool_selection", "arg_correct", "dependency_safe", "strict_schema"]
    cnt = ["hallucinated_tools", "unnecessary_calls", "invalid_args", "parse_errors", "turns"]
    out.append("| config | " + " | ".join(cols + cnt) + " |")
    out.append("|---|" + "---|" * (len(cols) + len(cnt)))
    for c in cfgs:
        r = s["configs"][c]["all_main"]
        out.append(f"| {c} | " + " | ".join(fmt(r.get(m)) for m in cols) + " | "
                   + " | ".join(fmt(r.get(m), pct=False) for m in cnt) + " |")
    out.append("\n## Paired differences in success (percentage points; seed-averaged per task)\n")
    out.append("| comparison | split | diff [95% CI] | p (sign-flip) | n |")
    out.append("|---|---|---|---|---|")
    for k, v in s["comparisons"].items():
        name, split = k.split("|")
        out.append(f"| {name} | {split} | {v['diff'] * 100:+.1f} [{v['lo'] * 100:+.1f}, {v['hi'] * 100:+.1f}] | "
                   f"{v['p']:.4f} | {v['n']} |")
    if "catalog_size" in s["ablations"]:
        out.append("\n## Catalogue-size ablation (success %, eval_catalog_k: k distractor tools)\n")
        out.append("| config | k=0 | k=5 | k=12 |")
        out.append("|---|---|---|---|")
        for c, r in s["ablations"]["catalog_size"].items():
            out.append(f"| {c} | " + " | ".join(fmt(r.get(k)) for k in ("0", "5", "12")) + " |")
    for key, title in [("family", "family"), ("depth", "gold workflow depth"), ("graph_seen", "graph seen in train"),
                       ("fault", "fault-injected / not-found")]:
        groups = sorted({g for c in cfgs for g in s["breakdowns"][c][key]})
        out.append(f"\n## Success (%) by {title} (pooled main splits, seed-averaged)\n")
        out.append("| config | " + " | ".join(groups) + " |")
        out.append("|---|" + "---|" * len(groups))
        for c in cfgs:
            b = s["breakdowns"][c][key]
            out.append(f"| {c} | " + " | ".join(
                f"{b[g]['success'] * 100:.1f} (n={b[g]['n_tasks']})" if g in b else "-" for g in groups) + " |")
    out.append("\n## Efficiency (per evaluation run)\n")
    out.append("| run | peak mem GB | wall s | episodes | mean turns | gen tok/s | mean latency/episode s |")
    out.append("|---|---|---|---|---|---|---|")
    for k, v in s["efficiency"].items():
        f = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"
        out.append(f"| {k} | {f(v['peak_mem_gb'])} | {f(v['wall_time_s'], 0)} | {v['episodes']} | "
                   f"{f(v['mean_turns'])} | {f(v['gen_tokens_per_s'], 0)} | {f(v['mean_episode_latency_s'])} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
