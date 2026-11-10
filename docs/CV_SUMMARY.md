# CV summary

**Cognitive Core Lab — fine-tuning a small language model for tool use and multi-step workflows**
*Personal research project · Python, Apple MLX, LoRA, Next.js/TypeScript · Nov 2026*

- Built an end-to-end pipeline to fine-tune and evaluate a 0.5B open-weight model (Qwen2.5-0.5B) for
  tool selection and multi-step orchestration: 21 simulated tools, 44 task types with automatic
  checking, and a 1,008-task held-out test set with a leakage audit.
- Compared LoRA training on teacher-model examples (4B model, filtered to 1,011 verified episodes)
  against programmatically generated examples, under matched training budgets with 3 seeds and
  confidence intervals.
- Raised task success on held-out workflows from 2% (untrained) and 13% (prompted) to 53–59%, and to
  86–91% on familiar task types; showed the gains do not extend to unseen tool families (no better than
  prompting).
- Diagnosed and fixed training divergence with gradient analysis, and documented failed runs and a
  flawed test as negative results.
- Shipped a local agent runtime with permission and human-approval controls, and a Next.js showcase
  with recorded-replay and live-inference modes.

**One-line version:** Fine-tuned a 0.5B model with LoRA (Apple MLX) for multi-step tool use; raised
held-out task success from 2% to 53–59% and reported where it fails to generalise.
