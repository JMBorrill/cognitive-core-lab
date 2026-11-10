# Case study: teaching a small model to run multi-step tool workflows

## The question
Can a 0.5-billion-parameter open model, small enough to run on a laptop, be trained cheaply to pick the
right tools, fill in their inputs correctly and carry a multi-step task through to the end? And does it
matter whether the training examples come from a bigger "teacher" model or from a program that is
correct by construction?

## What I built
- **A test environment.** 21 simulated business tools (contacts, calendar, email, orders, refunds,
  files, flights, currency and others) with typed inputs, role-based permissions, human-approval gates
  and injected failures (timeouts, outages with a backup tool, missing records).
- **A task generator.** 44 scenario types, each with a known-correct dependency graph and an automatic
  checker. A task only counts as solved if the final answer contains the required facts, the right
  actions happened exactly once, no permission was broken and no refused action was retried.
- **Held-out tests that resist memorisation.** 1,008 test tasks kept apart from training: unseen
  scenario types, tool families never used in training, and catalogues where every tool and parameter
  is renamed. A leakage audit checks requests, scenario types and graph shapes.
- **Two sources of training data.** 1,011 examples from a 4B teacher model, kept only if they passed
  every automatic check (63% of tasks), and examples from a programmatic solver.
- **Training and evaluation.** A resumable LoRA trainer in Apple MLX, four budget-matched setups with
  three random seeds each, four ablations, and statistics with confidence intervals and
  multiple-comparison correction.
- **A runtime and a showcase.** A command-line agent that asks a human before consequential actions, a
  local inference server, and a Next.js site that replays recorded runs or runs the model live, labelled
  as such.

## What happened
- **Training works, inside its comfort zone.** Task success went from 2% (untrained) and 13% (with a
  few-shot prompt) to 53–59% across all held-out tests, and 86–91% on new instances of familiar
  scenarios, after about 38 minutes of training on a laptop.
- **It doesn't generalise as far.** On unseen scenario types success is 30–36% (the 4B teacher: 56%).
  On unseen tool families, training adds nothing over a good prompt. The model learns patterns more
  than principles.
- **Data source matters less than data coverage.** Teacher-written and program-written examples gave the
  same overall score, but each model inherited the gaps of its data: the teacher almost never handled
  approval requests correctly, so neither did its student (1% vs 44%).
- **How you combine data matters.** Mixing both sources was best on average, though not significantly.
  Training on them one after the other collapsed on unseen tools, from 36% to 9%, consistently across
  seeds.

## What went wrong, and what I did about it
- **The first two training setups diverged.** Nearly every model learned for about ten steps, then
  collapsed. Instead of trying learning rates until something worked, I measured gradients before any
  training. One small set of weights in the model's earliest layers dominated them, and the LoRA
  settings amplified those updates. Moving the adapters to the later layers fixed it. Each selection
  rule was written down before the pilot ran, and the failed runs are kept and reported.
- **A flawed test.** One scenario could be answered from general knowledge, so every model, including
  the teacher, "failed" it by answering correctly without the expected lookup. I report the original
  numbers unchanged, plus a clearly labelled corrected view, rather than quietly editing the test after
  seeing the results.
- **A crash in the middle of a 6-hour run.** Unexpected model output crashed the tool simulator. I fixed
  it so malformed input becomes an error the model sees, added a test, and resumed without losing the
  work already done.

## What I'd do next
Repeat the single-seed ablations, fix the flawed scenario in a v2 test split, and try a stronger hosted
teacher (estimated at about 3M input tokens). The open question is whether better coverage of
unfamiliar workflows, not more examples of familiar ones, closes the generalisation gap.

Full method, numbers and limitations: [REPORT.md](REPORT.md).
