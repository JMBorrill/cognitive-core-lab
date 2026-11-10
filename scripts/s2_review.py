"""Study 02, Experiments 0 and 1: review checks.

    python scripts/s2_review.py            # results/study02/review/{review.json, review.md}
    python scripts/s2_review.py --quick    # skip the rebuild of the twenty training sets (about a minute)

Written for the review of the Experiment 0 and 1 results, after those results were known. It reads the
committed per-episode files and changes none of them. Two kinds of output, kept apart:

  Part A  verification. Quantities the plans pre-registered, derived a second time by code that shares
          nothing with scripts/s2_analyze.py beyond the input files, plus integrity checks on the data.
  Part B  post-hoc analyses. Chosen in review, after the results were seen. Exploratory, and labelled so
          wherever they are quoted.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccl import s2  # noqa: E402

RUNS = ROOT / "results/study02/runs"
OUT = ROOT / "results/study02/review"
STUDENTS = (("s05", "0.5B", (25, 100, 400, 1600), range(5)), ("s15", "1.5B", (25, 1600), range(3)))
SPLITS = ["own", "v2/dev", "v2/test_id", "v2/test_comp", "v2/test_depth", "v2/test_tool", "v2/test_domain",
          "v1/eval_id", "v1/eval_ood", "v1/eval_unseen", "v1/eval_renamed"]
T975 = {2: 4.303, 4: 2.776}  # two-sided 95% Student t quantile by degrees of freedom
# Deviations whose timing the plan states, and the commits that recorded them.
DEVIATIONS = {"trim of baseline and 1.5B scoring (E1 plan, deviation 3)": "a9dc1e0",
              "1.5B scoring set (E1 plan, deviation 5)": "47db539"}


def rows_of(name: str) -> list[dict]:
    with gzip.open(RUNS / "e1" / name / "metrics.jsonl.gz", "rt") as f:
        return [json.loads(l) for l in f]


def pc(x: float) -> str:
    return "-" if x is None or np.isnan(x) else f"{100 * x:.1f}"


def t_interval(v) -> dict:
    """Mean and 95% Student-t interval of replicate-level values (replicates are the independent units)."""
    v = np.asarray(v, dtype=float)
    half = T975[len(v) - 1] * v.std(ddof=1) / np.sqrt(len(v))
    return {"mean": float(v.mean()), "lo": float(v.mean() - half), "hi": float(v.mean() + half),
            "replicates": [float(x) for x in v]}


def ti(d: dict, signed: bool = True) -> str:
    f = "+.1f" if signed else ".1f"
    return f"{100 * d['mean']:{f}} [{100 * d['lo']:{f}}, {100 * d['hi']:{f}}]"


def cube(run_rows: list[list[dict]], split: str) -> np.ndarray:
    """(replicates, families, tasks) success array, aligned on family and task identifiers."""
    per = []
    for rows in run_rows:
        fam = defaultdict(dict)
        for r in rows:
            if r["split"] == split:
                fam[r["template"]][r["task_id"]] = r["success"]
        per.append(fam)
    fams = sorted(per[0])
    assert all(sorted(p) == fams and all(sorted(p[f]) == sorted(per[0][f]) for f in fams) for p in per)
    return np.array([[[p[f][t] for t in sorted(p[f])] for f in fams] for p in per])


def selections(task) -> tuple[str, int]:
    """Re-derive every list selection of a task from its request text and its world, without the generator.

    For each gold call that lists records, the selection rule is read back from the request (the phrase
    templates of ccl.compose), applied to the records that the list tool will return, and required to single
    out exactly one record that the rest of the gold graph, or the answer, then uses. Returns (verdict, number
    of selections checked); the verdict is "ok", "skipped" (the gold graph stops before the selections are
    made: not-found and refusal tasks) or a description of what went wrong."""
    from ccl.compose import MATCH_P, RANK_P, _ref
    from ccl.world2 import ENTITIES, TOOLS2
    lists = [g for g in task.gold if TOOLS2[g.tool].kind == "list"]
    if sum(t.startswith("rule=") for t in task.tags) != len(lists) or task.expected_status not in ("done", "blocked"):
        return "skipped", 0
    hits = {}
    for g in lists:
        t = TOOLS2[g.tool]
        child = ENTITIES[t.entity]
        words = dict(c=child.noun, cs=child.plural, p=_ref(child, t.ref_field).listed, x="")
        for a in child.attrs:
            if a.rank:
                for how, w in zip(("min", "max"), ("earliest", "latest") if a.kind[0] == "date" else ("lowest", "highest")):
                    for tpl in RANK_P:
                        text = tpl.format(w=w, a=a.phrase, **words)
                        for at in range(len(task.request)):
                            if task.request.startswith(text, at):
                                hits[at] = (child.name, how, a.name, None)
            if a.name == child.match:
                for v in a.kind[1]:
                    for tpl in MATCH_P:
                        text = tpl.format(v=v, a=a.phrase, **words)
                        for at in range(len(task.request)):
                            if task.request.startswith(text, at):
                                hits[at] = (child.name, "match", a.name, v)
    found = [hits[k] for k in sorted(hits)]
    if len(found) != len(lists):
        return f"{len(found)} selection phrases found for {len(lists)} list calls", 0
    # A request is right-branching, so within one walk the last phrase is the first selection made. A twin
    # request ("... of X and of Y") holds two walks one after the other.
    walks = 2 if "twin" in task.family else 1
    per = len(lists) // walks
    found = [r for w in range(walks) for r in reversed(found[w * per:(w + 1) * per])]
    fact = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
    for i, (g, (ent, how, attr, value)) in enumerate(zip(lists, found)):
        t = TOOLS2[g.tool]
        if ent != t.entity:
            return f"phrase about a {ent} where {g.tool} is called", 0
        (pid,) = g.args.values()
        listed = [r for r in task.world[t.entity].values() if r[t.ref_field] == pid]
        if how == "match":
            picked = [r for r in listed if r[attr] == value]
        else:
            best = (min if how == "min" else max)(r[attr] for r in listed) if listed else None
            picked = [r for r in listed if r[attr] == best]
        if len(picked) != 1:
            return f"{g.tool}: the rule singles out {len(picked)} of {len(listed)} records", 0
        later = {str(v) for h in task.gold[task.gold.index(g) + 1:] for v in h.args.values()}
        ids = {str(v) for k, v in picked[0].items() if k.endswith("_id")}
        if not (ids & later or {fact(v) for v in picked[0].values()} & set(task.answer_facts)):
            return f"{g.tool}: the record the rule singles out is not the one the gold graph uses", 0
    return "ok", len(lists)


def main() -> None:
    quick = "--quick" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    R: dict = {"A": {}, "B": {}}
    md = ["# Study 02, Experiments 0 and 1: review checks (generated by scripts/s2_review.py)", "",
          "Written in review, after the results were known. Part A re-derives pre-registered quantities and checks "
          "the data; Part B is post-hoc and exploratory. No result file of Experiments 0 and 1 is changed.", ""]
    fams, tools = s2.load_practice()
    by_id = {f["id"]: f for f in fams}
    part = json.loads((s2.V2 / "sealed/partition.json").read_text())
    sealed = s2.read_gz(s2.V2 / "sealed/families.jsonl.gz")
    dev = s2.read_gz(s2.V2 / "dev/families.jsonl.gz")
    sets = json.loads((ROOT / "results/study02/e1/training_sets.json").read_text())
    summary = json.loads((ROOT / "results/study02/e1/summary.json").read_text())
    order = {r: [f["id"] for f in s2.e1_order(fams, r)] for r in range(5)}
    data = {n.name: rows_of(n.name) for n in sorted((RUNS / "e1").iterdir())}
    name = lambda tag, d, r: f"e1_{tag}_D{d}_r{r}"  # noqa: E731

    # =================================================================== Part A: verification
    md += ["## Part A. Verification", ""]

    # ---- A1 hold-outs
    t_edges, d_edges = set(part["test_edges"]), set(part["dev_edges"])
    t_tools, d_tools, p_tools = set(part["test_tools"]), set(part["dev_tools"]), set(part["practice_tools"])
    pid = {f["id"] for f in fams}
    a1 = {"practice_families": len(fams),
          "practice_families_with_held_out_edge_or_tool": sum(bool(set(f["edges"]) & (t_edges | d_edges)
                                                                   or set(f["tools"]) - p_tools) for f in fams),
          "splits": {}}
    md += ["### A1. Hold-outs and leakage", "",
           "| split | families | hops | with a held-out test edge | with a held-out test tool | all tools practised | "
           "family also in the practice space |", "|---|---|---|---|---|---|---|"]
    for split in ("test_comp", "test_depth", "test_tool", "test_domain", "test_id"):
        v = [f for f in sealed if f["eval_split"] == split]
        rec = {"families": len(v), "hops": dict(sorted(Counter(f["hops"] for f in v).items())),
               "with_test_edge": sum(bool(set(f["edges"]) & t_edges) for f in v),
               "with_test_tool": sum(bool(set(f["tools"]) & t_tools) for f in v),
               "all_tools_practised": sum(set(f["tools"]) <= p_tools for f in v),
               "in_practice_space": sum(f["id"] in pid for f in v),
               "touching_dev_hold_outs": sum(bool(set(f["edges"]) & d_edges or set(f["tools"]) & d_tools) for f in v)}
        a1["splits"][split] = rec
        md.append(f"| {split} | {rec['families']} | {'/'.join(f'{h}:{n}' for h, n in rec['hops'].items())} | "
                  f"{rec['with_test_edge']} | {rec['with_test_tool']} | {rec['all_tools_practised']} | "
                  f"{rec['in_practice_space']} |")
    a1["dev_families_touching_test_hold_outs"] = sum(bool(set(f["edges"]) & t_edges or set(f["tools"]) & t_tools)
                                                     for f in dev)
    a1["dev_families_in_practice_space"] = sum(f["id"] in pid for f in dev)
    ok = (a1["practice_families_with_held_out_edge_or_tool"] == 0 and a1["dev_families_touching_test_hold_outs"] == 0
          and a1["dev_families_in_practice_space"] == 0
          and all(v["touching_dev_hold_outs"] == 0 for v in a1["splits"].values())
          and a1["splits"]["test_comp"]["with_test_edge"] == 600 and a1["splits"]["test_comp"]["all_tools_practised"] == 600
          and all(a1["splits"][s]["in_practice_space"] == 0 for s in ("test_comp", "test_depth", "test_tool", "test_domain")))
    a1["pass"] = bool(ok)
    md += ["", f"Practice families that contain a held-out edge or tool: {a1['practice_families_with_held_out_edge_or_tool']} "
               f"of {len(fams)}. `dev` families that touch a test hold-out: {a1['dev_families_touching_test_hold_outs']}. "
               f"**{'Pass' if ok else 'FAIL'}.** `test_id` is drawn from the practice space by design (see B1).", ""]
    R["A"]["hold_outs"] = a1

    # ---- A2 training subsets
    a2 = {"subsets_checked": 0, "recorded_coverage_matches": True, "nested": True, "sha256_checked": 0,
          "sha256_matches": None}
    for r in range(5):
        prev: set = set()
        for d in s2.E1_LEVELS:
            ids = order[r][:d]
            a2["nested"] &= prev <= set(ids)
            prev = set(ids)
            st = s2.subset_stats([by_id[i] for i in ids])
            rec = sets[f"r{r}_D{d}"]
            a2["recorded_coverage_matches"] &= (rec["family_ids"] == ids and rec["tools"] == st["tools"]
                                                and rec["edges"] == st["edges"])
            a2["subsets_checked"] += 1
    if not quick:
        good = True
        for r in range(5):
            for d in s2.E1_LEVELS:
                rows, stats = s2.oracle_rows(s2.e1_train_tasks(fams, tools, r, d))
                cfg = json.loads((ROOT / f"configs/study02/e1/e1_s05_D{d}_r{r}.json").read_text())
                got = json.loads((RUNS / "e1" / name("s05", d, r) / "train_run.json").read_text())["data"]["sha256"]
                good &= s2.rows_sha256(rows) == cfg["data"]["sha256"] == got and stats["episodes"] == 3200
                a2["sha256_checked"] += 1
        a2["sha256_matches"] = bool(good)
    overlap = [len(set(order[i][:1600]) & set(order[j][:1600])) for i in range(5) for j in range(i + 1, 5)]
    a2["families_shared_by_two_replicates_at_D1600"] = [min(overlap), max(overlap)]
    R["A"]["training_subsets"] = a2
    md += ["### A2. Training subsets", "",
           f"- {a2['subsets_checked']} subsets rebuilt from the practice space: nested within a replicate "
           f"({a2['nested']}), and identical in family list, tool coverage and edge coverage to "
           f"`results/study02/e1/training_sets.json` ({a2['recorded_coverage_matches']}).",
           ("- Training data not rebuilt (`--quick`)." if quick else
            f"- {a2['sha256_checked']} training sets rebuilt on CPU (3,200 oracle episodes each): SHA-256 equal to the "
            f"run config and to the hash each GPU job recorded ({a2['sha256_matches']})."),
           f"- Two replicates share {min(overlap)}-{max(overlap)} of their 1,600 families, so replicates are separate "
           f"draws from one practice space, not disjoint sets.", ""]

    # ---- A3 what was scored
    md += ["### A3. What was scored (episodes per run and split)", "",
           "| runs | " + " | ".join(s.split("/")[-1] for s in SPLITS) + " |", "|---|" + "---|" * len(SPLITS)]
    a3 = {}
    groups = [("0.5B, 20 trained runs", [n for n in data if n.startswith("e1_s05_D")]),
              ("1.5B, 6 trained runs", [n for n in data if n.startswith("e1_s15_D")]),
              ("4 untrained baselines", [n for n in data if n.endswith(("base", "fewshot"))])]
    for label, names in groups:
        counts = [Counter(r["split"] for r in data[n]) for n in names]
        cell = {s: sorted({c.get(s, 0) for c in counts}) for s in SPLITS}
        a3[label] = {s: v for s, v in cell.items()}
        md.append(f"| {label} | " + " | ".join("/".join(map(str, cell[s])) for s in SPLITS) + " |")
        assert all(len({(r["split"], r["task_id"]) for r in data[n]}) == len(data[n]) for n in names), "duplicate rows"
    R["A"]["scored"] = a3
    md += ["", "Every cell is either the full split or empty: no partial split entered a table. The 1.5B student has no "
               "`dev`, `test_depth`, `test_tool` or `test_domain` rows, and the baselines have no anchor or secondary-test "
               "rows. No run has a duplicated episode.", ""]

    # ---- A4 timing of the deviations against the access log
    access = [json.loads(l) for l in open(ROOT / "results/study02/test_access.jsonl")]
    first = lambda pred: min((a["time"] for a in access if a["split"] == "v2/test_comp" and pred(a["run"])),  # noqa: E731
                             default=None)
    a4 = {"first_test_comp_access_by_a_trained_run": first(lambda n: "_D" in n),
          "first_test_comp_access_by_a_trained_1.5B_run": first(lambda n: n.startswith("e1_s15_D")),
          "first_test_comp_access_by_a_baseline": first(lambda n: "_D" not in n), "commits": {}}
    for what, sha in DEVIATIONS.items():
        try:
            when = subprocess.run(["git", "log", "-1", "--format=%cI", sha], cwd=ROOT, capture_output=True, text=True,
                                  check=True, env={"TZ": "UTC", "PATH": "/usr/bin:/bin:/usr/local/bin"}).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            when = None
        a4["commits"][what] = {"commit": sha, "committed": when}
    R["A"]["deviation_timing"] = a4
    md += ["### A4. When the scoring cuts were recorded", "",
           "| event | time |", "|---|---|"]
    md += [f"| commit `{v['commit']}`: {k} | {v['committed'] or 'git history not available'} |"
           for k, v in a4["commits"].items()]
    md += [f"| first `test_comp` access by an untrained baseline | {a4['first_test_comp_access_by_a_baseline']} |",
           f"| first `test_comp` access by a trained run (0.5B) | {a4['first_test_comp_access_by_a_trained_run']} |",
           f"| first `test_comp` access by a trained 1.5B run | {a4['first_test_comp_access_by_a_trained_1.5B_run']} |", "",
           "The first cut was committed before any trained run began scoring `test_comp`. The 1.5B split list was "
           "committed before any 1.5B run began scoring `test_comp`, but after 0.5B `test_comp` scoring had started.", ""]

    # ---- A5 the confirmatory contrast by a second route
    md += ["### A5. The confirmatory contrast, derived a second time", "",
           "| student | D = 1,600 minus D = 25, per replicate | Student-t interval over replicates | bootstrap written "
           "here (replicates, families, tasks) | interval in `e1/summary.json` |", "|---|---|---|---|---|"]
    a5 = {}
    for tag, label, levels, reps in STUDENTS:
        lo = cube([data[name(tag, 25, r)] for r in reps], "v2/test_comp")
        hi = cube([data[name(tag, 1600, r)] for r in reps], "v2/test_comp")
        tint = t_interval(hi.mean((1, 2)) - lo.mean((1, 2)))
        rng = np.random.default_rng(20261110)
        n_r, n_f, n_t = lo.shape
        boots = np.empty(4000)
        row = np.arange(n_f)[:, None]
        for b in range(len(boots)):
            ri, fi, ti_ = rng.integers(0, n_r, n_r), rng.integers(0, n_f, n_f), rng.integers(0, n_t, (n_f, n_t))
            boots[b] = hi[ri][:, fi][:, row, ti_].mean() - lo[ri][:, fi][:, row, ti_].mean()
        ceil = t_interval(hi.mean((1, 2)))
        ref = summary["students"][tag]["primary"]["D1600_minus_D25"]
        a5[tag] = {"t_interval": tint, "bootstrap": {"lo": float(np.quantile(boots, .025)), "hi": float(np.quantile(boots, .975))},
                   "reported": ref, "ceiling_t_interval": ceil,
                   "supported_under_t_interval": bool(tint["mean"] >= 0.10 and tint["lo"] > 0)}
        md.append(f"| {label} | " + ", ".join(f"{100 * x:+.1f}" for x in tint["replicates"]) + f" | {ti(tint)} | "
                  f"[{100 * a5[tag]['bootstrap']['lo']:+.1f}, {100 * a5[tag]['bootstrap']['hi']:+.1f}] | "
                  f"{100 * ref['mean']:+.1f} [{100 * ref['lo']:+.1f}, {100 * ref['hi']:+.1f}] |")
    R["A"]["primary_contrast"] = a5
    md += ["", "H1 is supported for both students under either interval. With three replicates the percentile bootstrap "
               f"is too narrow for the 1.5B student; the Student-t interval ({ti(a5['s15']['t_interval'])}) is the one to "
               "quote. The 0.5B ceiling at D = 1,600 is "
               f"{ti(a5['s05']['ceiling_t_interval'], False)} by the same method: the point estimate is below 40% and the "
               "interval includes it.", ""]

    # ---- A6 cost
    jobs: dict = {}
    for e in (json.loads(l) for l in open(ROOT / "results/study02/ledger.jsonl")):
        j = jobs.setdefault(e.get("key"), {})
        if e["event"] == "reserve":
            j["reserved"] = e["reserved_usd"]
        elif e["event"] in ("settled", "provider_cost"):
            j["cost"] = e["cost_usd"]
            if e["event"] == "settled":
                j["stage"] = e["stage"]
    billed = {k: j for k, j in jobs.items() if k and "reserved" in j}
    over = {k: round(j["cost"] - j["reserved"], 4) for k, j in billed.items() if j.get("cost", 0.0) > j["reserved"] + 1e-9}
    a6 = {"jobs": len(billed), "total_usd": round(sum(j.get("cost", 0.0) for j in billed.values()), 2),
          "ended_in_error": sum(j.get("stage") == "ERROR" for j in billed.values()),
          "billed_above_their_reservation_usd": over}
    R["A"]["cost"] = a6
    md += ["### A6. Cost", "",
           f"{a6['jobs']} jobs in the ledger, {a6['ended_in_error']} ended in error, total billed {a6['total_usd']:.2f} USD "
           f"of the 60 USD cap.",
           f"{len(over)} jobs were billed more than the launcher had reserved for them (" +
           ", ".join(f"`{k}` by {v:.4f} USD" for k, v in over.items()) + "): both ran into their time limit, and the "
           "provider bills two minutes beyond a time limit where the launcher reserved one. The 3 USD reserve "
           "absorbed it and the cap held, but the launcher's claim that a job cannot cost more than its reservation "
           "was not true. Fixed in review (`reservation_usd`).", ""]

    # ---- A7 every list selection, re-derived from the task's world
    from ccl import compose as C
    md += ["### A7. List selections re-derived from the task worlds", "",
           "The reference-model gate vouched for families of at most two hops. This check covers the property it "
           "was meant to protect at every depth: for each gold call that lists records, the selection rule is read "
           "back from the request text and applied to the task's world, and must single out exactly one record, the "
           "one the gold graph or the answer then uses. It shares the phrase templates with the generator and nothing "
           "else.", "",
           "| tasks | checked | selections | not applicable (graph stops before selecting) | failures |", "|---|---|---|---|---|"]
    a7 = {}
    pools = {s_: s2.load_split(s_) for s_ in ("test_comp", "test_depth", "test_tool", "test_domain", "test_id", "dev")}
    pools["practice (2 training instances per family)"] = [C.make_task(f, i, "train", "practice", tools)
                                                           for f in fams for i in (0, 1)]
    for label, ts in pools.items():
        res = [selections(t) for t in ts]
        bad = sorted({v for v, _ in res if v not in ("ok", "skipped")})
        a7[label] = {"tasks": len(ts), "checked": sum(v == "ok" for v, _ in res), "selections": sum(n for _, n in res),
                     "not_applicable": sum(v == "skipped" for v, _ in res),
                     "failures": sum(v not in ("ok", "skipped") for v, _ in res), "failure_kinds": bad[:5]}
        md.append(f"| {label} | {a7[label]['checked']} | {a7[label]['selections']} | {a7[label]['not_applicable']} | "
                  f"{a7[label]['failures']} |")
    a7["pass"] = all(v["failures"] == 0 for v in a7.values())
    R["A"]["list_selections"] = a7
    md += ["", f"**{'Pass' if a7['pass'] else 'FAIL'}.** This does not test whether a request reads naturally or whether "
               "a model would find it clear; it tests that the stated rule has one answer in the world and that the gold "
               "graph follows it.", ""]

    # =================================================================== Part B: post-hoc
    md += ["## Part B. Post-hoc analyses (chosen in review; exploratory)", ""]

    # ---- B1 test_id is a mixture
    md += ["### B1. `test_id` mixes practised and unpractised families", "",
           "`test_id` holds 200 families of the practice space. A run's training subset contains some of them, and "
           "the share grows with D, so the rise of `test_id` with D mixes two things: generalisation to unpractised "
           "families built from practised edges, and more of the split being practised.", "",
           "| student | D | `test_id` families in the run's training subset | success, in subset | success, not in subset |",
           "|---|---|---|---|---|"]
    b1 = {}
    for tag, label, levels, reps in STUDENTS:
        for d in levels:
            ins, outs, share = [], [], []
            for r in reps:
                sub = set(order[r][:d])
                rows = [x for x in data[name(tag, d, r)] if x["split"] == "v2/test_id"]
                a = [x["success"] for x in rows if x["template"] in sub]
                ins.append(np.mean(a) if a else np.nan)
                outs.append(np.mean([x["success"] for x in rows if x["template"] not in sub]))
                share.append(len(a) / len(rows))
            b1[f"{tag}_D{d}"] = {"share_in_subset": float(np.mean(share)), "in_subset": float(np.nanmean(ins)),
                                 "not_in_subset": float(np.mean(outs))}
            md.append(f"| {label} | {d} | {100 * np.mean(share):.0f}% | {pc(float(np.nanmean(ins)))} | {pc(float(np.mean(outs)))} |")
    R["B"]["test_id_mixture"] = b1
    md += ["", "The two groups differ in depth (shallow families are over-represented among the practised ones), so "
               "the columns are not a like-for-like contrast; B2 gives that. Where the report calls `test_id` \"practised "
               "families\", read \"families of the practice space\".", ""]

    # ---- B2 own-family accuracy
    md += ["### B2. In-distribution accuracy and the number of families", "",
           "Own-family success per replicate, and differences paired on replicate (Student-t interval). The last "
           "column scores the same tasks at both levels: the own-family tasks that both runs were given, which are "
           "instances of the families the smaller training subset contains.", "",
           "| student | contrast | own-family success, all tasks | on identical tasks |", "|---|---|---|---|"]
    b2 = {}
    for tag, label, levels, reps in STUDENTS:
        own = {d: [{x["task_id"]: x["success"] for x in data[name(tag, d, r)] if x["split"] == "own"} for r in reps]
               for d in levels}
        mean = {d: [float(np.mean(list(o.values()))) for o in own[d]] for d in levels}
        b2[tag] = {"mean": {str(d): float(np.mean(mean[d])) for d in levels}, "contrasts": {}}
        pairs = [(25, 100), (100, 1600), (25, 1600)] if tag == "s05" else [(25, 1600)]
        for lo_d, hi_d in pairs:
            all_tasks = t_interval(np.array(mean[hi_d]) - np.array(mean[lo_d]))
            same, n_same = [], []
            for r in reps:
                common = sorted(set(own[lo_d][r]) & set(own[hi_d][r]))
                n_same.append(len(common))
                same.append(np.mean([own[hi_d][r][k] for k in common]) - np.mean([own[lo_d][r][k] for k in common]))
            same_i = t_interval(same)
            b2[tag]["contrasts"][f"D{hi_d}_minus_D{lo_d}"] = {"all_tasks": all_tasks, "identical_tasks": same_i,
                                                            "identical_tasks_per_replicate": n_same[0]}
            md.append(f"| {label} | D = {hi_d:,} minus D = {lo_d:,} | {ti(all_tasks)} | {ti(same_i)} "
                      f"({n_same[0]} tasks per replicate) |")
    by_hops = {}
    for tag, label, levels, reps in STUDENTS:  # the identical tasks of the widest contrast, by depth
        cell = defaultdict(lambda: [[], []])
        for r in reps:
            runs = [{x["task_id"]: x for x in data[name(tag, d, r)] if x["split"] == "own"} for d in (25, 1600)]
            for k in set(runs[0]) & set(runs[1]):
                for i in (0, 1):
                    cell[runs[0][k]["hops"]][i].append(runs[i][k]["success"])
        by_hops[tag] = {str(h): {"tasks": len(v[0]), "D25": float(np.mean(v[0])), "D1600": float(np.mean(v[1]))}
                        for h, v in sorted(cell.items())}
    b2["identical_tasks_by_hops_D25_vs_D1600"] = by_hops
    R["B"]["own_family"] = b2
    md += ["", "The identical tasks of the widest contrast by number of hops, replicates pooled (success % at 25 and at "
               "1,600 families; few tasks per cell): " + "; ".join(
                   f"{label}: " + ", ".join(f"{h} hops {100 * v['D25']:.0f} to {100 * v['D1600']:.0f} (n = {v['tasks']})"
                                            for h, v in by_hops[tag].items()) for tag, label, _, _ in STUDENTS) + "."]
    m = b2["s05"]["mean"]
    md += ["", f"For the 0.5B student the fall is real and is confined to the first step: {pc(m['25'])}% at 25 families "
               f"(128 instances each), {pc(m['100'])}% at 100 (32 each), {pc(m['400'])}% and {pc(m['1600'])}% at 400 and "
               "1,600 (8 and 2 each). What is lost is the benefit of seeing one family more than a hundred times, not "
               "accuracy that keeps eroding as families are added. The 1.5B student shows no reliable fall.", ""]

    # ---- B3 tasks no trained run ever solved
    tasks = {t.task_id: t for t in s2.load_split("test_comp")}
    wins, wins15 = Counter(), Counter()
    for n, rows in data.items():
        if "_D" in n:
            for x in rows:
                if x["split"] == "v2/test_comp":
                    wins[x["task_id"]] += int(x["success"])
                    if n.startswith("e1_s15_D1600"):
                        wins15[x["task_id"]] += int(x["success"])
    never = sorted(k for k in tasks if wins[k] == 0)
    rules = lambda t: sum(x.startswith("rule=") for x in t.tags)  # noqa: E731
    b3 = {"tasks": len(tasks), "never_solved_by_any_of_26_trained_runs": len(never),
          "never_solved_by_hops": dict(sorted(Counter(tasks[k].depth for k in never).items())),
          "never_solved_by_the_three_1.5B_D1600_runs": sum(wins15[k] == 0 for k in tasks),
          "by_selection_rules": {}, "ids": never}
    md += ["### B3. `test_comp` tasks that no trained run solved", "",
           f"The report counts families: 3 of 600 were never solved. Counting tasks, {len(never)} of {len(tasks)} were "
           f"never solved in 26 attempts (by hops: "
           + ", ".join(f"{h}: {n}" for h, n in b3["never_solved_by_hops"].items()) + "). They follow the number of "
           "list selections a request makes (\"the order with the lowest total among...\"), which the hop count does "
           "not capture:", "",
           "| list selections in the request | tasks | never solved | 0.5B, D = 1,600 | 1.5B, D = 1,600 |",
           "|---|---|---|---|---|"]
    for k in sorted({rules(t) for t in tasks.values()}):
        ids = {i for i, t in tasks.items() if rules(t) == k}
        acc = {tag: float(np.mean([x["success"] for r in reps for x in data[name(tag, 1600, r)]
                                   if x["split"] == "v2/test_comp" and x["task_id"] in ids]))
               for tag, _, _, reps in STUDENTS}
        b3["by_selection_rules"][str(k)] = {"tasks": len(ids), "never_solved": sum(i in ids for i in never), **acc}
        md.append(f"| {k} | {len(ids)} | {sum(i in ids for i in never)} | {pc(acc['s05'])} | {pc(acc['s15'])} |")
    R["B"]["never_solved_tasks"] = b3
    md += ["", "All 26 never-solved tasks of at most three hops were read in review with their oracle transcripts. "
               "Each has one defensible reading and a correct gold answer; they are hard because they chain two to four "
               "selections or need a weekday turned into a date. This is a check by reading, on a sample chosen by "
               "difficulty. It does not replace the reference-model gate for deep families.", ""]

    # ---- B4 the turn limit and test_depth
    depth = [x for r in range(5) for x in data[name("s05", 1600, r)] if x["split"] == "v2/test_depth"]
    need = Counter(x["min_turns"] for x in depth)
    b4 = {"turn_limit": 8, "share_of_tasks_by_turns_the_oracle_needs": {str(k): v / len(depth) for k, v in sorted(need.items())},
          "episodes_ended_by_the_turn_limit": {}}
    for split in ("v2/test_depth", "v2/test_comp"):
        for tag, label, levels, reps in STUDENTS:
            for d in levels:
                rows = [x for r in reps for x in data[name(tag, d, r)] if x["split"] == split]
                if rows:
                    b4["episodes_ended_by_the_turn_limit"][f"{split[3:]}|{tag}_D{d}"] = float(
                        np.mean([x["termination"] == "max_turns" for x in rows]))
    R["B"]["turn_limit"] = b4
    e = b4["episodes_ended_by_the_turn_limit"]
    md += ["### B4. The eight-turn limit and `test_depth`", "",
           "The runtime, reused unchanged from Study 01, ends an episode after eight turns. On `test_depth` the oracle "
           "itself needs " + ", ".join(f"{k} turns on {100 * v:.0f}% of tasks" for k, v in
                                       b4["share_of_tasks_by_turns_the_oracle_needs"].items()) +
           ". A six-hop task with a retry therefore fails after a single wasted turn.", "",
           "| split | student | " + " | ".join(f"D = {d}" for d in (25, 100, 400, 1600)) + " |", "|---|---|---|---|---|---|"]
    for split in ("test_depth", "test_comp"):
        for tag, label, levels, reps in STUDENTS:
            if any(f"{split}|{tag}_D{d}" in e for d in levels):
                md.append(f"| {split}, episodes ended by the limit (%) | {label} | " + " | ".join(
                    pc(e[f"{split}|{tag}_D{d}"]) if f"{split}|{tag}_D{d}" in e else "-" for d in (25, 100, 400, 1600)) + " |")
    md += ["", "`test_depth` is near the floor partly for this reason, and the report should say so. The limit does not "
               "explain the four-hop results on the primary endpoint, where the oracle needs at most six turns.", ""]

    # ---- B5 how four-hop episodes fail
    md += ["### B5. How four-hop `test_comp` episodes fail at D = 1,600", "",
           "| student | success | answered before the minimum number of turns | failures missing a required call | "
           "failures with an invented tool | failures ended by a limit |", "|---|---|---|---|---|---|"]
    b5 = {}
    for tag, label, levels, reps in STUDENTS:
        rows = [x for r in reps for x in data[name(tag, 1600, r)] if x["split"] == "v2/test_comp" and x["hops"] == 4]
        bad = [x for x in rows if not x["success"]]
        b5[tag] = {"success": float(np.mean([x["success"] for x in rows])),
                   "early_final": float(np.mean([x["termination"] == "final" and x["turns"] < x["min_turns"] for x in rows])),
                   "failures_missing_required_call": float(np.mean([1 - x["check_required_calls"] for x in bad])),
                   "failures_with_invented_tool": float(np.mean([x["hallucinated_tools"] > 0 for x in bad])),
                   "failures_ended_by_limit": float(np.mean([x["termination"] != "final" for x in bad])),
                   "longest_mean_tokens_per_turn": float(max(x["gen_tokens"] / max(1, x["turns"]) for x in rows))}
        v = b5[tag]
        md.append(f"| {label} | {pc(v['success'])} | {pc(v['early_final'])} | {pc(v['failures_missing_required_call'])} | "
                  f"{pc(v['failures_with_invented_tool'])} | {pc(v['failures_ended_by_limit'])} |")
    R["B"]["four_hop_failures"] = b5
    md += ["", "Failures are wrong or missing calls and early answers, the errors of a model that loses the chain. No "
               f"episode comes near the 256-token limit per turn (longest mean: "
               f"{max(v['longest_mean_tokens_per_turn'] for v in b5.values()):.0f} tokens), so truncation is not a cause.", ""]

    # ---- B6 the reference-model gate by depth
    gate = [json.loads(l) for p in ("gate", "gate_shallow") for l in open(RUNS / p / "metrics.jsonl")]
    b6 = {}
    md += ["### B6. The reference model by depth (first attempts, `Qwen2.5-7B-Instruct`, few-shot)", "",
           "| hops | tasks | solved | mean turns used | minimum turns needed |", "|---|---|---|---|---|"]
    for h in sorted({x["hops"] for x in gate}):
        v = [x for x in gate if x["hops"] == h and x["split"].endswith("#1")]
        b6[str(h)] = {"tasks": len(v), "success": float(np.mean([x["success"] for x in v])),
                      "turns": float(np.mean([x["turns"] for x in v])), "min_turns": float(np.mean([x["min_turns"] for x in v]))}
        md.append(f"| {h} | {len(v)} | {pc(b6[str(h)]['success'])} | {b6[str(h)]['turns']:.1f} | {b6[str(h)]['min_turns']:.1f} |")
    R["B"]["reference_gate"] = b6
    md += ["", "The reference model is competent at one hop only. At four hops it stops after three turns on average "
               "where five are the minimum, so its failures say nothing about the tasks. Three-, five- and six-hop "
               "test families were not attempted by it at all, and four-hop ones only in the run that was stopped. A "
               "trained 1.5B student solves "
               f"{pc(b5['s15']['success'])}% of four-hop `test_comp` tasks and is the better reference (see the "
               "Experiment 2 plan).", ""]

    (OUT / "review.json").write_text(json.dumps(R, indent=1) + "\n")
    (OUT / "review.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
