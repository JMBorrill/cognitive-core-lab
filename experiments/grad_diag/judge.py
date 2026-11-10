import json, statistics as st, sys
for d in sys.argv[1:]:
    L = [json.loads(x) for x in open(f"{d}/train_log.jsonl")]
    loss = [r["loss"] for r in L]; gn = [r["grad_norm"] for r in L]
    ma = [sum(loss[max(0, i - 4): i + 1]) / len(loss[max(0, i - 4): i + 1]) for i in range(len(loss))]
    i_ok = max(ma[20:]) <= 1.5 * ma[9]
    g0, g1 = st.median(gn[:10]), st.median(gn[80:100])
    ii_ok = g1 < 10 * g0
    val = {r["step"]: round(r["val_loss"], 4) for r in L if "val_loss" in r}
    print(f"{d}: ma10 {ma[9]:.3f} max_ma>20 {max(ma[20:]):.3f} (i {i_ok}) | gn med 1-10 {g0:.1f} 81-100 {g1:.1f} (ii {ii_ok}) "
          f"| stable {i_ok and ii_ok} | val {val} | loss@100 {ma[-1]:.3f}")
