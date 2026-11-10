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

## Clarifications recorded before the first run (2026-11-10)

Written after the environment was built and tested on CPU and before any model was trained or evaluated on it.

1. **Family.** A family is one gold dependency graph: role, expected status, the calls and their dependencies,
   and the faults. Study 01's `graph_signature` is used with one refinement: a tool that fails once (retry) and
   a tool that is down (fallback) are different families, because they demand different behaviour. Wording,
   concrete values, the attribute asked for, the rule that picks from a list, distractor tools and the order of
   independent calls never make a new family. Each base graph has at most one retry and one fallback variant, and
   "the same question about two records" is a family only for paths of at most two calls
   (`ccl/compose.py`).
2. **Drawing the subsets.** Replicate *r* draws one seeded order of the practice space
   (`ccl.s2.e1_order`). The subset for *D* families is the first *D* entries, so the levels are nested. Every
   prefix has the same composition: modifier classes in fixed proportions (55% none, 15% retry, 10% fallback,
   10% denial, 10% not-found, refusal and no-tool, renormalised where a class does not exist), and hops as even
   as that mix allows. With 414 one-hop families in the practice space, 188 of them single-call failure or
   refusal families, this gives 18% / 25% / 28% / 29% for 1 to 4 hops at every level, and 2.8 to 2.9 gold calls
   per task at every level. Realised compositions, tool coverage and edge coverage of every subset are in
   `results/study02/e1/training_sets.json`.
3. **What a higher level changes besides D.** A subset of 25 families shows 39 to 50 of the 80 gold tools and
   about 40 of the 235 practised edges; a subset of 1,600 shows all tools and nearly all edges. D is therefore
   not separable from tool and edge coverage by design. Both are recorded per run, and the pre-registered
   secondary endpoint (families whose tools all occur in the run's training set) is reported next to the primary
   one, together with the same restriction on edges.
4. **Own-family split.** 400 new instances of the run's own training families, from the evaluation value pool.
   Replicate *r* uses the same 400 families at D = 400 and D = 1,600.
5. **Reference split.** `test_id` (200 practice families, 2 new instances each) is scored for every run as well.
6. **Pairing in the bootstrap.** Replicate *r* shares its seed and its nested subsets across levels, so the
   bootstrap resamples replicate indices once per draw and applies them to every level, then families, then
   tasks within families (`ccl.stats.cluster_bootstrap`).
7. **Stability rule.** As in Experiment 0, applied to steps 1-800.
8. **Validation loss** during training is computed on oracle trajectories for the first 80 `dev` tasks. It is
   monitored only. No setting is tuned in Experiment 1.
9. **Evaluation batch and length limit.** 128 episodes in lock-step for the 0.5B student, 64 for the 1.5B
   student, scored in a fixed order grouped by depth and prompt length. Greedy decoding uses a minimal loop
   that gives the same outputs as the library's `generate` (tested). The training length limit is 4,096 tokens
   instead of 2,048 so that no episode is dropped: the longest oracle episode has 2,074 tokens.
10. **Quality gate, step 3.** `Qwen/Qwen2.5-7B-Instruct` with the few-shot prompt attempts the first task of
    every test family and the second task of every family whose first attempt failed. A family is removed only
    if reading its failures shows a defect in the task (an inconsistent world, an ambiguous request, a gold graph
    that penalises a defensible solution). Failing because the task is hard is not a reason. A defect that
    comes from the generator is fixed in the generator, and the data are rebuilt and re-checked. Every decision
    is logged in `data/v2/gate/`.
11. **Budget.** Hard cap of 60 USD for Experiments 0 and 1 together, enforced by `scripts/s2_launch.py`.

## Frozen inputs (recorded 2026-11-10, before the first Experiment 1 run)

- `data/v2/manifest.json` SHA-256: `eec8eb4398ec254eb77e33f30b097ffd13d5f21661ae88f551dd5dcb4031d005`
- commit that last changed `data/v2`: `66b8b0b4d4fefbe4cf5d0ec890b25d0c28627e08`
- census: 5,364 practice families (2,103 base graphs, 79 gold tools, 229 practised edges); `test_comp` 600
  families, `test_depth` 200, `test_tool` 200, `test_domain` 200, `test_id` 200, `dev` 300, two tasks each
- training settings from Experiment 0 (`results/study02/e0/decision.json`): LoRA rank 16,
  alpha 32, all blocks, bias-corrected AdamW, lr 0.0001, 20 warm-up steps,
  cosine decay to 10%, gradient clipping 1.0, 800 steps x 8 episodes
- run configs: `configs/study02/e1/` (each names the SHA-256 its training set must have when rebuilt in the job)
- the code commit of every job is recorded in `results/study02/ledger.jsonl` and in each run's `run.json`
- quality gate at the time of freezing: oracle 100% on every task; both degenerate policies fail every task that
  needs a tool; reference-model reading of the two-hop `test_comp` families found no task defect and no family
  was removed (`data/v2/gate/`)

## Deviations

1. **Quality gate, step 3.** The reference model (`Qwen2.5-7B-Instruct`, few-shot prompt) solved 3% of the
   first 200 four-hop `test_comp` tasks through its own errors, so the run over all test families was stopped and
   replaced by one over families of at most two hops. Reading its failures found no task defect and no family was
   removed. The gate therefore vouches for shallow families only (`data/v2/gate/README.md`). Recorded before any
   student was scored on a sealed split.
2. **Gradient checkpointing for the 1.5B student.** Its six runs train with gradient checkpointing so that the
   longest episodes fit on a 24 GB card. Gradients are the same (tested); only memory and speed change. Recorded
   before those runs were submitted.
3. **Scoring cost and what was trimmed to stay under the cap.** Scoring a trained 0.5B run on every split takes
   about 54 minutes on an A10G (0.72 s per episode), three times the plan's estimate, and the untrained
   baselines are slower still because they emit long malformed output. To keep the total under 60 USD without
   touching the confirmatory test: (a) the four baselines are scored on `dev`, `test_id` and `test_comp` only,
   and stopped once `test_comp` is complete; their v1-anchor values for the 0.5B student are taken from Study 01,
   where they were measured (2.2% and 12.8%); (b) the six 1.5B runs keep their job limit of 160 minutes and are
   scored in the fixed order own, `dev`, `test_id`, `test_comp`, `test_depth`, `test_tool`, `test_domain`, v1
   anchor, so later splits may be missing for them. All twenty 0.5B runs are scored on every split. Decided at
   14:36 UTC on 2026-11-10, when no `test_comp` result of any trained run had been looked at; the only trained
   results seen were own-family and partial `dev` scores of the five D = 1,600 runs.
4. **How the jobs were packed.** Runs were first submitted several to a job and then, to finish sooner, one to a
   job, resuming from the checkpoints and partial results in the storage bucket. A run's results do not depend
   on which job produced them; the code commit of each job is in the ledger and in each `run.json`.
5. **Out-of-memory failures and the 1.5B scoring set.** Four jobs failed with a GPU out-of-memory error during
   scoring (the first quality-gate job, two untrained baselines, one 1.5B run). The fallback that halves a batch
   retried inside the exception handler, where the failed call's memory cannot be released; it was fixed and
   tested (`ccl/hf.py`). No result was lost: adapters and partial scores were in the storage bucket. The six 1.5B
   runs had all finished training; their scoring was resubmitted on the fixed code with 32 episodes in lock-step
   instead of 64 and, replacing point 3(b), on a fixed list of splits chosen for the budget before any 1.5B
   `test_comp` result existed: own families, `test_comp`, `test_id` and the v1 anchor. `dev`, `test_depth`,
   `test_tool` and `test_domain` are not scored for the 1.5B student, except for partial `dev` rows already
   written before the failures.

