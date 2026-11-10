"""Gradient norms at initialisation (no updates) on B_s0's first micro-batches, per module type."""
import json, sys, collections
import mlx.core as mx, numpy as np
from mlx.utils import tree_flatten
sys.path.insert(0, ".")
from ccl.train import load_examples, batch_schedule, collate, loss_fn, DEFAULTS
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
cfg = {**DEFAULTS, **json.load(open("configs/train/B.json"))}
model, tok = load(cfg["model"]); model.freeze()
NL = int(sys.argv[1]) if len(sys.argv) > 1 else len(model.layers); linear_to_lora_layers(model, NL, cfg["lora"]); print("lora layers", NL)
print("dtypes", collections.Counter(str(v.dtype) for _, v in tree_flatten(model.trainable_parameters())),
      "base", model.layers[-1].mlp.down_proj.linear.weight.dtype)
exs, _ = load_examples(cfg["train_files"][0], tok, cfg["max_seq_len"])
micro = batch_schedule(len(exs), cfg["batch_size"], 12, cfg["seed"])
lg = nn.value_and_grad(model, loss_fn)
for i, idx in enumerate(micro):
    b, m = collate([exs[j] for j in idx], cfg["max_seq_len"])
    (lv, nt), g = lg(model, b, m)
    flat = tree_flatten(g)
    per = collections.defaultdict(float)
    for k, v in flat:
        per[k.split(".")[-3] if "lora" in k else k] += (v.astype(mx.float32) ** 2).sum().item()
    tot = sum(per.values()) ** .5
    top = sorted(per.items(), key=lambda x: -x[1])[:2]
    worst = max(flat, key=lambda kv: (kv[1].astype(mx.float32) ** 2).sum().item())[0]
    print(f"mb{i:02d} loss {lv.item():.3f} sup {int(nt.item()):5d} width {b.shape[1]:5d} gnorm {tot:12.1f} "
          f"top {[(k, round(v**.5,1)) for k,v in top]} worst {worst} nonfinite {any(not np.isfinite(np.array(v.astype(mx.float32))).all() for _,v in flat)}", flush=True)

# per-layer breakdown on the last micro-batch
per_layer = collections.defaultdict(float)
for k, v in flat:
    per_layer[int(k.split(".")[2])] += (v.astype(mx.float32) ** 2).sum().item()
print("per-layer grad norm:", {l: round(x ** .5, 1) for l, x in sorted(per_layer.items())})

