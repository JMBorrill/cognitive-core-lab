"""Apply the pre-registered stability rule (experiments/grad_diag/PLAN.md) to every finished run -> logs/stability.jsonl."""
import json, pathlib, statistics as st
root = pathlib.Path(__file__).resolve().parents[1]
with open(root / "logs/stability.jsonl", "w") as out:
    for log in sorted(root.glob("runs/*/train_log.jsonl")):
        L = [json.loads(x) for x in open(log) if x.strip()]
        if len(L) < 30:
            continue
        loss = [r["loss"] for r in L]; gn = [r["grad_norm"] for r in L]
        ma = [sum(loss[max(0, i - 4): i + 1]) / len(loss[max(0, i - 4): i + 1]) for i in range(len(loss))]
        rec = {"run": log.parent.name, "steps": len(L), "loss_ok": max(ma[20:]) <= 1.5 * ma[9],
               "gradnorm_ok": st.median(gn[-20:]) < 10 * st.median(gn[:10]),
               "ma10": ma[9], "max_ma_after20": max(ma[20:]), "gn_first10": st.median(gn[:10]), "gn_last20": st.median(gn[-20:])}
        rec["stable"] = rec["loss_ok"] and rec["gradnorm_ok"]
        out.write(json.dumps(rec) + "\n")
