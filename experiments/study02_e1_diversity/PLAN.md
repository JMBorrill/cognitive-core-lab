# Study 02, Experiment 1: diversity dose-response (plan written 2026-11-10, before the environment exists)

Status: draft for review. It becomes binding when the `data/v2` manifest hash is recorded below, which
must happen before the first training run. Later changes are logged under "Deviations".

## Question

With the number of training episodes and optimizer steps fixed, does success on held-out task
structures rise with the number of distinct structures in the training set? And how high can a 0.5B
model get on this task space with perfect labels?

Background: Study 01 trained on 23 structures and reached 30–36% on unseen templates; tripling
episodes from the same structures did not help ([audit](../../docs/study02/STUDY01_AUDIT.md)).

## Design

| | |
|---|---|
| Factor | number of distinct training families D: 25, 100, 400, 1,600 |
| Family subsets | drawn from the practice space, spread as evenly as possible over 1, 2, 3 and 4 hops; nested within a replicate (the 25 are among the 100, and so on) |
| Episodes | 3,200 at every level: 128, 32, 8 and 2 instances per family |
| Labels | oracle trajectories, kept under the common acceptance rule |
| Training | settings fixed by Experiment 0; 800 steps × 8 episodes, from the base model |
| Replicates | 5 per level; replicate *r* uses seed *r* for both the family subset and training |
| Students | `Qwen2.5-0.5B-Instruct` at all four levels; `Qwen2.5-1.5B-Instruct` at D = 25 and 1,600, replicates 0–2 |
| Baselines | base and few-shot prompted, both sizes |
| Evaluation | greedy; `test_comp`, `test_depth`, `test_tool`, `test_domain`, the v1 anchor, and new instances of the run's own training families |

26 training runs. The nested subsets mean that a higher level adds families and never swaps them.

## Outcomes

**Primary.** Success on `test_comp` (600 families × 2 tasks; every family contains at least one edge
absent from the whole practice space).

**Secondary.** The other splits; the gap between a run's own-family success and its `test_comp`
success; `test_comp` restricted to families whose tools all occur in that run's training set; strict
success; the Study 01 diagnostics.

## Hypothesis and criteria

**H1.** For the 0.5B student, mean `test_comp` success at D = 1,600 exceeds that at D = 25.

- Supported: the difference is ≥ 10 points and its 95% interval excludes zero.
- Refuted: the 95% interval lies entirely below +10 points.
- Otherwise inconclusive.

The same contrast is computed for the 1.5B student as a secondary test.

## Analysis

- Intervals: hierarchical bootstrap, 10,000 resamples of replicates, then families, then tasks. The
  D = 1,600 versus D = 25 contrast is paired on families and tasks.
- Trend: slope of success on log₂ D across the four levels, with the same bootstrap.
- Multiplicity: H1 for 0.5B is the single confirmatory test. Everything else is descriptive.
- Variance components on `test_comp` (seed, seed × family, level × family, within) are estimated as in
  `scripts/study02_audit.py` and reported. They set the number of seeds for Experiment 2.
- A run that fails the stability rule is rerun once with the learning rate halved and logged as a
  deviation. Runs are never dropped.

## Decisions that follow

| result | decision |
|---|---|
| H1 supported for 0.5B and `test_comp` at D = 1,600 is ≥ 40% | Experiment 2 on the 0.5B student |
| H1 refuted or inconclusive for 0.5B, supported for 1.5B | Experiment 2 on the 1.5B student |
| H1 supported but `test_comp` at D = 1,600 is < 40% for both sizes | task space too hard at this size: revise the held-out definition or the student before Experiment 2 |
| H1 refuted for both sizes | stop the loop programme, report the negative result, investigate task representation |
| estimated seed sd on `test_comp` > 0.03 | Experiment 2 uses 8 seeds per main arm |

## Not to be changed after results are seen

The split definitions, the primary endpoint, the D levels, the thresholds above, and the training
settings.

## Budget

About 28 GPU-hours on a 24 GB GPU. Shares a $60 cap with Experiment 0.

## Frozen inputs

- `data/v2/manifest.json` SHA-256: *to be recorded before the first run*
- commit: *to be recorded before the first run*
- training settings from Experiment 0: *to be recorded before the first run*

## Deviations

None yet.
