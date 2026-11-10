# Study 01 audit

*Prepared 2026-11-10 as groundwork for Study 02. Based on a direct read of `ccl/`, `scripts/`, `tests/`,
the 18 evaluation runs in `results/runs/`, the datasets in `data/v1/`, and `archive/` and `experiments/`.
New numbers come from [`scripts/study02_audit.py`](../../scripts/study02_audit.py); its full output is
[audit_tables.md](audit_tables.md). Every analysis in this document is post-hoc.*

## Summary

Study 01 is sound and its headline numbers hold up. Three properties of its design limit what it can
say about generalisation, and each one shapes Study 02:

1. **Training covered 23 task structures.** The 3,000-task pool comes from 24 templates with 23 distinct
   dependency graphs. Tripling the episodes drawn from those same structures did not help. The evidence
   is consistent with the gap being a shortage of *distinct structures*, which no arm of Study 01 varied.
2. **The unseen-template result rests on 16 templates.** With the template as the unit of resampling, the
   95% interval for Dmix on `eval_ood` is 19–53%, against the reported 29–43%. The design could
   detect a difference between two training arms only if it was about 17–24 points.
3. **"Unseen" mixes several kinds of novelty.** New compositions, deeper chains, tools never practised and
   new behaviours share one split, and they fail for different reasons.

The environment, runtime, validator, oracle and statistics are pure Python and reproduce exactly on
Linux. Only the trainer and the inference backend depend on MLX.

## What Study 01 demonstrated

| claim | evidence | status |
|---|---|---|
| LoRA training lifts a 0.5B model from 2% to 86–91% on new instances of training templates | 3 seeds per arm; per-seed range 84.0–92.0% | demonstrated |
| Models read the catalogue instead of memorising tool names | full renaming costs 13–21 points | demonstrated |
| Teacher demonstrations and oracle trajectories give the same aggregate result | C − B within ±4.2 points on every split | demonstrated at this power; the two arms differ strongly per template (below) |
| Success on unseen templates is far lower (30–36%) | `eval_ood`, per-seed range 25.4–40.4% | direction demonstrated; magnitude uncertain (16 templates) |
| Sequential training on two sources collapses on unseen tool domains | Dseq 9.2% vs Dmix 36.4%, all three seeds | demonstrated on 4 templates; cause unknown |
| More data from the same templates does not help | C_25 55.0, C 55.9, C_50 62.5, C_full 52.2 pooled | suggestive; one seed per point |

## Findings that shape Study 02

### 1. The unit of generalisation is the template, and there are 16

`eval_ood` has 240 tasks, but they are 15 instances each of 16 templates, and instances of one template
succeed or fail together. Template means for Dmix range from 0% to 100%.

| config | `eval_ood` | reported CI (seeds, tasks) | CI resampling seeds and templates |
|---|---|---|---|
| B | 30.3 | [23.9, 37.5] | [16.8, 44.9] |
| C | 30.3 | [23.9, 36.8] | [13.9, 48.3] |
| Dmix | 35.6 | [29.2, 42.8] | [19.4, 52.9] |
| teacher | 56.2 | [50.0, 62.5] | [38.3, 73.3] |

The variance components explain why. A template's success moves by about 16 points (sd) from one
training seed to the next, and the difference between two arms varies across templates with an sd of
18–30 points. With 16 templates and 3 seeds, the smallest difference between two arms detectable with
80% power is 17–24 points. Study 01's conclusion that B, C and Dmix are indistinguishable on unseen
templates is correct, but the study could not have told them apart unless they differed hugely.

**Consequence.** Study 02 needs hundreds of held-out families and resamples them. With 300 families,
5 seeds and 4 tasks per family, the detectable difference falls to 5–6 points
([audit_tables.md §5](audit_tables.md)). Hundreds of families cannot be written by hand, so the
environment needs a task generator.

### 2. Training covered 23 structures

| | count |
|---|---|
| training templates | 24 |
| distinct gold-graph signatures in the 3,000-task pool | 23 |
| tools that are gold in at least one training task | 13 of 17 exposed |
| `eval_ood` signatures absent from training | 12 of 16 |

Success on graphs seen in training is 72–74%, and 26–38% on novel graphs. C_full used all 3,000 tasks
and scored no better than C_25 with 250. Teacher and oracle data, which cover the same 24 templates,
give the same aggregate. Every arm varied the labels or the number of episodes. None varied the number
of structures.

**Consequence.** The first question for Study 02 is whether structure diversity is the lever. This is
cheap to test with oracle labels and it bounds what any learning loop over the same task space can
achieve.

### 3. `eval_ood` mixes at least four kinds of novelty

| kind of novelty | templates | trained students | 4B teacher |
|---|---|---|---|
| needs a tool that was never gold in training (flights, refunds) | 4 | 0–42% | 67–93% |
| deeper chain than trained (depth 4, 5) | 2 | depth 4: 13–82%; depth 5: 0–4% | 27–33% |
| new composition of practised tools | 5 | 2–100%, strongly arm-dependent | 13–100% |
| new behaviour or phrasing (capability question, symptom-to-priority, refusal) | 5 | 0–76% | 0–100% |

Per-template results are in [audit_tables.md §1](audit_tables.md). Two examples of how much the arms
differ underneath an identical average: `ch_deep4` is 82% for B and 13% for C, and `rec_fx_timeout` is
38% for B and 98% for C.

**Consequence.** Study 02 reports each kind of novelty as its own split, and makes "new composition of
practised tools" the primary endpoint because it isolates combination from exposure.

### 4. Some held-out tasks may penalise defensible behaviour

On two held-out templates the 4B teacher fails the required-calls check in every episode:
`un_timezone` (already documented as flawed) and `rec_contact_missing_email`, where it makes 0.2
calls per episode on average. The second has not been reviewed. A stronger model declining to make a
call that the gold graph demands is a signal that the task specification may be ambiguous.

**Consequence.** Study 02 adds a quality gate before the evaluation set is frozen: every generated task
must be solved by the oracle, and families that a strong reference model fails are reviewed before any
student result exists.

### 5. Training settings were chosen for stability on MLX

The final configuration (rank 16, scale 20, last 16 of 24 blocks, lr 1e-5) was the third tried after two
divergences. MLX-LM's `scale` multiplies the low-rank update directly, so scale 20 at rank 16
corresponds to `lora_alpha = 320` in PEFT terms, far from the common `alpha = 2 × rank`. The gap may
depend partly on these settings.

**Consequence.** The PyTorch port starts with a bridge run on the v1 data: reproduce B and C, then check
that the gap survives conventional LoRA settings.

### 6. The data filter uses privileged information

An episode enters the training set only if it passes the validator *and* has no unnecessary or
dependency-unsafe calls. Both extra criteria are computed from the gold graph. That suits teacher and
oracle data, but a learner that is supposed to receive only pass/fail feedback must not be filtered by
criteria that encode the solution.

**Consequence.** Study 02 uses one acceptance rule for every data source, based on the outcome and on
what the runtime itself observes.

### 7. Other observations

- **`eval_id` is unbalanced.** The verbatim-overlap filter left between 1 and 13 tasks per template, so
  a few templates dominate the in-distribution number.
- **Seed effects are large where families are few.** On `eval_unseen` (4 templates) per-seed success
  ranges from 16.7% to 45.8% for B.
- **Per-episode traces are not in the repository** (`traces.jsonl.gz` is git-ignored), so failure
  analysis beyond the validator checks needs a re-run.

## Reuse

| component | Study 02 |
|---|---|
| `ccl/tools.py`, `tasks.py` (data model, matching rules), `runtime.py`, `oracle.py`, `evaluate.py`, `data.py` | reused unchanged; verified on Linux |
| `ccl/stats.py` | extended with family-level resampling |
| `ccl/train.py`, `ccl/models.py` (MLX) | kept for Study 01; a PyTorch backend is added alongside |
| `data/v1` evaluation splits | kept frozen and used as an external anchor: Study 02 never trains on v1 |
| `scripts/analyze.py` conventions (JSONL per run, generated tables) | kept |

## Validation performed in this session (Linux, CPU, Python 3.11)

- `tests/test_core.py`: 28 of 28 pass.
- `scripts/build_datasets.py` rebuilds all 11 task and oracle files with SHA-256 identical to
  `data/v1/manifest.json`. This is now pinned by `tests/test_study01_frozen.py`.
- `tests/test_train.py` cannot be collected without MLX. It is left untouched; the PyTorch backend gets
  its own tests.
