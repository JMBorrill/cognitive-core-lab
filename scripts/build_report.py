"""Fill the artifact-derived sections of docs/REPORT.md (between <!-- BEGIN:x --> / <!-- END:x --> markers).

Sections: tables (results/tables.md), teacher (data/v1/teacher/manifest.json), data (variants +
manifests), training (runs/*/summary.json), queue (logs/queue.jsonl wall-clock). Narrative text
outside the markers is written by hand and must only cite numbers that appear in these sections.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def section_tables() -> str:
    t = (ROOT / "results/tables.md").read_text()
    return re.sub(r"^# .*\n", "", t).replace("\n## ", "\n### ")


def section_teacher() -> str:
    m = json.loads((ROOT / "data/v1/teacher/manifest.json").read_text())
    rows = ["| pass | decoding | episodes | accepted | acceptance | rejection reasons (an episode can have several) |",
            "|---|---|---|---|---|---|"]
    dec = {"pass1": "greedy", "pass2": "T=0.7, top-p 0.95 (rejected tasks only)"}
    for p, s in m["summary"].items():
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(s["rejection_reasons"].items(), key=lambda x: -x[1]))
        rows.append(f"| {p} | {dec[p]} | {s['episodes']} | {s['accepted']} | {s['acceptance_rate'] * 100:.1f}% | {reasons} |")
    t = m["timings_s"]
    fam = ", ".join(f"{k} {v}" for k, v in sorted(m["accepted_by_family"].items()))
    return "\n".join(rows) + (
        f"\n\nTeacher: `{m['teacher']['repo']}` (upstream `{m['teacher']['upstream']}`, {m['teacher']['licence']}). "
        f"Tasks offered: {m['tasks_offered']}; accepted episodes: {m['accepted_episodes']} "
        f"({m['accepted_episodes'] / m['tasks_offered'] * 100:.1f}% of tasks). "
        f"Accepted episodes by family: {fam}. "
        f"Wall-clock: pass 1 {t['pass1_s'] / 3600:.2f} h, pass 2 {t['pass2_s'] / 3600:.2f} h. API cost: $0 (local).\n")


def section_data() -> str:
    v = json.loads((ROOT / "data/v1/sft/variants.json").read_text())
    rows = ["| variant | episodes | SFT rows (assistant turns) | sha256 (first 12) | families (episodes) |",
            "|---|---|---|---|---|"]
    for k, x in v.items():
        fam = ", ".join(f"{a} {b}" for a, b in sorted(x["families"].items()))
        rows.append(f"| {k} | {x['episodes']} | {x['rows']} | `{x['sha256'][:12]}` | {fam} |")
    man = json.loads((ROOT / "data/v1/manifest.json").read_text())
    rows += ["", "| task file | n | sha256 (first 12) |", "|---|---|---|"]
    for k, x in man["files"].items():
        if k.startswith("tasks/"):
            rows.append(f"| {k} | {x['n']} | `{x['sha256'][:12]}` |")
    audit = json.loads((ROOT / "data/v1/leakage_audit.json").read_text())
    rows += ["", "| eval split | n | request overlap | template overlap | graph-signature overlap | "
                 "held-out tools seen in train | dropped (verbatim request in train) |", "|---|---|---|---|---|---|---|"]
    for k, a in audit.items():
        rows.append(f"| {k} | {a['n']} | {a['request_overlap']} | {a['template_overlap']} | "
                    f"{a['graph_signature_overlap_frac'] * 100:.0f}% | {len(a['unseen_tools_in_train'])} | "
                    f"{a.get('dropped_for_request_overlap', '-')} |")
    return "\n".join(rows) + "\n"


def section_training() -> str:
    rows = ["| run | steps | train time (min) | peak mem (GB) | final val loss (oracle val) | trainable params | adapter sha256 (first 12) |",
            "|---|---|---|---|---|---|---|"]
    total = 0.0
    for d in sorted((ROOT / "runs").glob("*_s[0-9]")):
        s = d / "summary.json"
        if not s.exists():
            rows.append(f"| {d.name} | — | not completed | | | | |")
            continue
        x = json.loads(s.read_text())
        total += x["train_time_s"]
        rows.append(f"| {d.name} | {x['steps']} | {x['train_time_s'] / 60:.1f} | {x['peak_mem_gb']:.2f} | "
                    f"{x['final_val_loss']:.4f} | {x['trainable_params']:,} | `{x['adapter_sha256'][:12]}` |")
    return "\n".join(rows) + f"\n\nTotal LoRA training wall-clock: {total / 3600:.2f} h on one Apple M4.\n"


def section_queue() -> str:
    p = ROOT / "logs/queue.jsonl"
    if not p.exists():
        return "_no queue log_\n"
    recs = [json.loads(l) for l in open(p)]
    ok = [r for r in recs if r["rc"] == 0]
    bad = [r for r in recs if r["rc"] != 0]
    ev = sum(r["seconds"] for r in ok if r["step"].startswith("eval"))
    tr = sum(r["seconds"] for r in ok if r["step"].startswith("train"))
    out = f"Successful queue steps: {len(ok)} (training {tr / 3600:.2f} h, evaluation {ev / 3600:.2f} h wall-clock).\n"
    if bad:
        out += "\nFailed or interrupted steps (re-run by the resumable queue):\n\n" + "\n".join(
            f"- `{r['step']}` exit {r['rc']} after {r['seconds']} s ({r['ended']})" for r in bad) + "\n"
    return out


SECTIONS = {"tables": section_tables, "teacher": section_teacher, "data": section_data,
            "training": section_training, "queue": section_queue}


def main():
    path = ROOT / "docs/REPORT.md"
    text = path.read_text()
    for name, fn in SECTIONS.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->\n).*?(<!-- END:{name} -->)", re.S)
        if pat.search(text):
            try:
                body = fn()
            except FileNotFoundError as e:
                body = f"_artifact missing: {e.filename}_\n"
            text = pat.sub(lambda m: m.group(1) + body + m.group(2), text)
    path.write_text(text)
    print("report sections refreshed:", [n for n in SECTIONS if f"<!-- BEGIN:{n} -->" in text])


if __name__ == "__main__":
    main()
