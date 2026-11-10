# Study 02 proposal: can a small model close its generalisation gap by practising in a verifiable environment?

*Draft for review, 2026-11-10. Nothing in this proposal has been run on paid compute.
Companion documents: [Study 01 audit](STUDY01_AUDIT.md), [literature positioning](LITERATURE.md),
[implementation plan](IMPLEMENTATION_PLAN.md), and the pre-registration of the first experiment in
[`experiments/study02_e1_diversity/PLAN.md`](../../experiments/study02_e1_diversity/PLAN.md).*

## 1. Summary

**Question.** Study 01 trained a 0.5B model to 86–91% on new instances of its training templates and
30–36% on templates it had not seen. Study 02 asks what closes that gap, and whether the model can
close it by itself: attempting tasks, receiving only a pass or fail from an executable verifier, and
retraining on its own verified successes over successive cycles.

**What the audit adds.** Study 01 trained on 23 distinct task structures, and no arm varied that
number. Its unseen-template split has 16 templates, so it could only have detected a difference of
17–24 points between training arms ([audit](STUDY01_AUDIT.md)).

**Thesis.** The gap is limited by how many distinct task structures the student practises, not by who
supplies the solutions. If that is right, three things follow, and each can fail:

1. With episodes and optimizer steps fixed, success on held-out structures rises with the number of
   distinct training structures.
2. On a diverse task space, verified self-practice reaches about the same held-out success as teacher
   demonstrations, using no teacher output after a small seed.
3. Practice without the verifier does not.

**Programme.**

| | experiment | question | estimated cost |
|---|---|---|---|
| 0 | Bridge | Does the PyTorch stack reproduce Study 01, and does the gap survive conventional LoRA settings? | $4 |
| 1 | Diversity dose-response | Is structure diversity the lever, and how far can this task space take a 0.5B model? | $23 |
| 2 | Signal source × curriculum | Teacher demonstrations, oracle trajectories, verified self-practice and weakness-targeted self-practice at matched budgets | $110 |
| 3 | Retention and combination | Does the learner keep earlier skills while adding new ones, and do separate specialists combine? | $50 |
| 4 | Teacher dependence | How small can the seed be, and can the task space grow? | $70 |

**Recommended first experiment: Experiment 1**, entered through Experiment 0. It needs no teacher and
no loop, costs about $27 together with the bridge, and its result decides whether the later
experiments are worth running and on which student (§7.2, §11).

**Decision needed before a teacher is used.** Anthropic's Usage Policy prohibits using Claude outputs
to train a model without prior authorisation. The default design therefore uses an open-weight
teacher, and Fable 5.2 stays in the research-assistant role (§13).

## 2. Terms

- **Family**: all tasks that share one gold dependency graph, ignoring the concrete values (the
  `graph_signature` of Study 01). A family is the unit that generalisation is claimed over.
- **Edge**: one tool's output feeding another tool's argument.
- **Practice space**: the families a learner is allowed to train on.
- **Sealed test set**: families that no training, tuning or curriculum code can read.
- **Seed adapter**: a LoRA adapter trained on a small, narrow set of demonstrations, shared by every
  arm of Experiment 2.
- **Verified success**: an episode that passes the executable validator.

## 3. Hypotheses

Each hypothesis names the experiment that tests it, the criterion for support and the result that
refutes it. Success is the Study 01 validator, on the primary endpoint defined in §8.

| | hypothesis | supported if | refuted if |
|---|---|---|---|
| **H1 diversity** (Exp. 1) | At fixed episodes and steps, held-out success rises with the number of distinct training families D | success at D = 1,600 exceeds success at D = 25 by ≥ 10 points, 95% CI above zero | the 95% CI of that difference lies entirely below +10 |
| **H2 autonomy** (Exp. 2, primary) | Verified self-practice with a weakness-targeted curriculum closes the gap without further teacher output | it gains ≥ 10 points over the seed adapter **and** is non-inferior to static teacher distillation at a margin of 7.5 points, at equal training episodes and steps | either part fails: no 10-point gain, or the 95% CI of (practice − teacher) lies entirely below −7.5 |
| **H3 curriculum** (Exp. 2) | Weakness-targeted task selection beats uniform selection | ≥ 5 points, CI above zero | CI entirely below +5 |
| **H4 verification** (Exp. 2) | The verifier is what makes practice work | verified practice beats unverified practice by ≥ 5 points, CI above zero | CI entirely below +5 |
| **H5 retention** (Exp. 3) | Earlier skills survive new learning when earlier data is replayed, and do not when it is not | without replay, earlier groups lose ≥ 10 points; with ≤ 10% replay the loss against joint training is ≤ 3 points | replay at 10% still loses > 3 points, or no forgetting occurs without replay |

Results that meet neither criterion are reported as inconclusive. My prior for H3 is below even: two
controlled studies found no gain from curricula at a fixed budget ([literature](LITERATURE.md) §1).

**What each refutation would mean.**

- H1 refuted at 0.5B but not at 1.5B: the gap is a capacity limit at this size.
- H1 refuted at both sizes: neither more structures nor more parameters helps, and the programme pivots
  to how the task is represented before building any loop.
- H2 refuted: autonomy has a price, and Experiment 2 measures it.
- H4 refuted: any gain from practice is self-distillation or extra optimisation, not learning from the
  environment.

## 4. Position in the literature

The loop itself is established: expert iteration on verified self-generated data works for code and
mathematics, and Tool-R0 (2026) applies self-play to tool calling at 0.5B and 1.5B. What has not been
shown, as far as an abstract-level review can tell:

- a dose-response for the number of plan structures in a small tool-using model;
- a comparison of teacher demonstrations, programmatic supervision and verified self-practice on one
  task space at matched training budgets;
- held-out families that are sealed from the curriculum and resampled in the statistics, with
  retention measured every cycle.

That is the contribution claimed here. The review was limited to search-indexed abstracts because
arxiv.org was unreachable from this session; [LITERATURE.md](LITERATURE.md) marks what needs
re-checking.

## 5. Environment

Study 01's runtime, validator, oracle and task data model are reused unchanged. What changes is where
tasks come from.

### 5.1 A task generator in place of hand-written templates

Hundreds of held-out families cannot be written by hand. The v2 environment derives everything from
one declarative schema:

- **World.** About 12 domains (for example library, clinic, fleet, procurement, billing) with about 36
  entity types, their fields, and the references between them. No domain or tool is shared with v1.
- **Tools.** Derived from the schema: look up by ID, find by attribute, list by reference, and
  actions that change state. Some actions need approval, some are role-gated, some lookups have a
  fallback. About 80 tools, each simulated by one of a handful of generic functions.
- **Families.** A family is a typed path through the schema: a starting entity given by ID or by name,
  one to six hops along references, optional parallel fan-out, fan-in, or a choice over a list
  (cheapest, latest, matching), and an ending that is either an answer or an action. Modifiers add
  the behaviours of Study 01: transient failure, fallback, not-found, approval denied, role refusal,
  argument normalisation, and requests that need no tool.
- **Requests.** Composed from phrase templates attached to each reference, with several variants; a
  subset of variants is reserved for evaluation.
- **Gold graph and checks.** Emitted with the task, in the Study 01 format.

The schema gives on the order of 10⁴ families up to four hops. The generator writes an exact census
into the data manifest, and the build fails if the practice space has fewer than 5,000.

### 5.2 Splits

| split | definition | size | role |
|---|---|---|---|
| practice space | families of 1–4 hops over practised tools and edges | ≥ 5,000 families | training, practice |
| `dev` | its own held-out edges, held-out tools and one held-out domain | 300 families × 2 tasks | tuning, per-cycle monitoring |
| `test_comp` | families that contain at least one **edge never practised**, built only from practised tools | 600 × 2 | **primary endpoint** |
| `test_depth` | 5–6 hops over practised edges | 200 × 2 | secondary |
| `test_tool` | families needing a tool absent from every practice catalogue | 200 × 2 | secondary |
| `test_domain` | two domains never practised | 200 × 2 | secondary |
| `test_id` | new instances of practised families | 200 × 2 | reference for the gap |
| v1 anchor | the four Study 01 evaluation splits, 720 tasks, frozen | 720 | external check |

`dev` and the test splits hold out different edges, tools and domains, so tuning on `dev` does not
tune on test families.

The **v1 anchor** is the independent check on the whole approach. Study 02 never trains on a v1
template or a v1 tool, and v1 was written by hand before the generator existed. Reference points
already exist: 12.8% for the few-shot base model and 59.1% for the best Study 01 arm, which was
trained on v1.

### 5.3 Quality gate before the test set is frozen

1. Every generated task is run by the oracle and must pass the validator.
2. Degenerate policies must fail it: answering without calls, and calling every tool.
3. A strong open-weight reference model attempts two tasks from every test family. Families it fails
   twice are read for ambiguity and fixed or dropped, with the decision logged.

The gate runs before any student is evaluated. The frozen files are hashed in the manifest.

## 6. Protocol common to all experiments

| | choice | reason |
|---|---|---|
| Student | `Qwen/Qwen2.5-0.5B-Instruct` at the Study 01 revision | continuity with Study 01; the closest published work uses the same model |
| Scale check | `Qwen/Qwen2.5-1.5B-Instruct` | separates capacity from data |
| Second family | one non-Qwen model of about 1B, for the final confirmation of Experiment 2 | guards against a result specific to one model family |
| Learning method | LoRA supervised fine-tuning on verified episodes | simplest method that worked in Study 01; RL is a later option (§7.3) |
| Stack | PyTorch, `transformers`, PEFT on Linux/CUDA | portable; Study 01's MLX trainer stays for Study 01 |
| Decoding | greedy for evaluation; sampled for practice | as Study 01 |

**One acceptance rule for every data source.** An episode enters a training set only if it passes the
validator and the runtime recorded no parse error, unknown tool or invalid argument. When several
attempts at a task pass, the one with the fewest calls is kept. Criteria derived from the gold graph
(unnecessary or dependency-unsafe calls) are measured and reported, never used for filtering.

**Budget matching.** Arms are compared at the same number of accepted training episodes and the same
number of optimizer steps, each trained from the base model. Three costs are reported per arm and
never hidden inside the match: GPU-seconds including rollouts and rejected attempts, teacher tokens,
and oracle calls.

**Hyperparameters** are fixed once in Experiment 0, on validation tasks from the training
distribution, and then used for every arm.

## 7. Experiments

### 7.1 Experiment 0: bridge

Purpose: trust the new stack, and check that the Study 01 gap is not an artefact of its training
settings.

- Reproduce arms B and C on the v1 data with three seeds each, using the Study 01 settings expressed
  in PEFT. Pass: `eval_id` and `eval_ood` means within 5 points of Study 01.
- Train C with conventional settings (all blocks, `alpha = 2 × rank`, three learning rates, two
  lengths). A conventional default is adopted for Study 02 unless it is unstable or clearly worse on
  validation tasks than the best of the six. The `eval_ood` scores of this grid are reported, not
  used for selection.
- Measure training and evaluation throughput on two GPU types and choose the cheaper per token.
- Evaluate one adapter twice to measure how many episodes flip between identical runs.

### 7.2 Experiment 1: diversity dose-response (recommended first)

Purpose: test H1 and measure the ceiling of the task space before building a loop.

| factor | levels |
|---|---|
| distinct training families D | 25, 100, 400, 1,600 (nested subsets, equal mix of 1–4 hops) |
| training episodes | 3,200 at every level |
| optimizer steps | 800 at every level (two passes) |
| labels | oracle |
| replicates | 5 per level; each draws a new family subset and training seed |
| student size | 0.5B at all four levels; 1.5B at D = 25 and 1,600 with 3 replicates |

Twenty-six training runs plus base and few-shot baselines for both sizes. Every run is evaluated on
all splits of §5.2.

What it delivers beyond H1:

- the **ceiling**: held-out success with perfect labels and maximal diversity, which no learner over
  the same task space is expected to exceed;
- the choice of student for Experiment 2 (rule in §11);
- variance components on the new environment, which set the number of seeds for Experiment 2;
- measured cost per run.

### 7.3 Experiment 2: signal source × curriculum

Purpose: the comparison the study is named for. All arms start from the same position Study 01 ended
in: a narrow supervised start and a generalisation gap.

**Seed.** 400 verified teacher demonstrations from 50 families of one or two hops. The seed adapter is
trained on them. If the teacher is accepted on fewer than 80% of seed tasks, oracle trajectories are
used for the seed and this is reported.

**Arms.** Each adds 2,880 accepted episodes to the seed in 6 cycles of 480.

| arm | where the 480 episodes per cycle come from | information the learner receives |
|---|---|---|
| `teacher` | teacher demonstrations on tasks drawn uniformly from the practice space; one greedy attempt and one sampled retry, as in Study 01 | full solutions from a larger model |
| `oracle` | oracle trajectories on tasks drawn uniformly | full solutions from a privileged solver |
| `practice` | the student's own attempts on tasks drawn uniformly; 4 sampled attempts per task | pass or fail |
| `practice+curriculum` | as `practice`, with tasks drawn towards measured weakness | pass or fail |
| `seed` (control) | nothing | nothing beyond the seed |
| `practice-unverified` (control) | as `practice`, but attempts are accepted at random among well-formed ones | no verification signal |

`teacher` and `oracle` are static: their data does not depend on the student. They are the first two
arms of the brief; `practice` is autonomous verified learning and `practice+curriculum` adds the
adaptive curriculum.

**One cycle of a practice arm.**

1. Draw tasks from the practice space (uniformly, or by the curriculum).
2. The current adapter attempts each task four times at temperature 0.8.
3. The verifier returns pass or fail for each attempt. The learner never sees the gold graph.
4. Accepted episodes, at most one per task, join the buffer. The cycle ends at 480 accepted episodes
   or after 2,400 tasks.
5. A new adapter is trained from the base model on seed plus buffer, two passes.
6. The new adapter is evaluated on `dev`.

Retraining from the base model on accumulated data follows ReST-EM and avoids the drift that stacked
low-rank updates accumulate. The final adapters of all arms are therefore trained identically on
3,280 episodes and differ only in which episodes those are.

**Curriculum.** Practice tasks are grouped into about 50 buckets by number of hops, shape and
modifier. A bucket's weight is `p(1 − p) + 0.05`, where `p` is the student's single-attempt pass rate
on that bucket over the last two cycles, smoothed so that an untried bucket counts as 0.5. A fifth of
tasks are always drawn uniformly. This favours tasks the student solves sometimes: mastered buckets
and currently impossible ones both get little weight. The curriculum reads only practice outcomes. It
has no access to `dev` or to any test split.

**Replicates.** Five seeds for the four main arms, three for each control. The four main arms are then
repeated with three seeds on the second model family.

**Diagnostics reported with the result.**

- Learning curves per cycle on `dev`, and on the test splits computed afterwards from saved adapters.
- Yield: accepted episodes per attempted task, per bucket and cycle.
- Sharpening versus new capability: on 300 held-out tasks, how many the final adapter solves greedily
  that the seed adapter failed in 16 sampled attempts.
- Verifier audit: the false-accept rate in a random sample of 200 accepted practice episodes, checked
  against the oracle's calls and read where they differ.
- Whether practice-stream weakness predicts held-out failure.

**Follow-ups, only if indicated.** If H3 is supported, an `oracle+curriculum` arm tests whether
adaptivity also helps supervised data. If practice plateaus while most buckets stay near a zero pass
rate, policy-gradient RL on the same reward is the next method to try; it is excluded from the main
design because it adds instability at this model size for no gain in clarity.

### 7.4 Experiment 3: retention, combination and specialists

Purpose: test H5 and the modular-specialist idea on the best autonomous arm of Experiment 2.

The practice domains are divided into three groups, introduced one per stage. Cross-group edges are
held out from all training.

| arm | how stage *i* is learned |
|---|---|
| `sequential` | continue the adapter on group *i* only |
| `replay-1`, `replay-10`, `replay-25` | continue on group *i* mixed with 1%, 10% or 25% earlier episodes |
| `cumulative` | retrain from base on all groups so far (reference for sequential arms) |
| `specialists+router` | one adapter per group, selected by which domains appear in the task's catalogue |
| `specialists+merge` | the same adapters combined by averaging and by learned concatenation |
| `joint` | one adapter trained on all groups at once (reference for specialists) |

Measures: the accuracy matrix R[i, j] (success on group *j* after stage *i*), backward and forward
transfer, the v1 anchor at every stage, and success on **cross-group compositions** that no arm
trained on. The published evidence predicts that routed specialists will match the joint adapter
within groups and fall short on cross-group compositions.

### 7.5 Experiment 4: teacher dependence and a growing task space

Run only if H2 is at least partly supported.

- **Seed size.** Repeat the best practice arm from seeds of 0, 100, 400 and 1,600 teacher
  demonstrations. The zero-seed run starts from the few-shot prompted base model. Outcome: held-out
  success as a function of teacher episodes consumed.
- **Teacher on demand.** Tasks the student fails in two consecutive cycles receive a teacher
  demonstration. Outcome: teacher calls per cycle, which should fall if dependence is decreasing.
- **Growing the task space.** New domains are added to the schema between cycles and admitted through
  the quality gate of §5.3. This is the first step towards an open-ended loop and is exploratory.

## 8. Measurement and statistics

**Primary endpoint.** Success on `test_comp`: families with at least one edge the learner never
practised, built from tools it did practise. This isolates combining known capabilities from exposure
to new ones.

**Secondary endpoints.** The other test splits; the gap (`test_id` minus `test_comp`); the v1 anchor;
strict success (success with no unnecessary call and no dependency-unsafe call); the Study 01
diagnostics; GPU-seconds and teacher tokens per point of held-out gain.

**Keeping evaluation isolated.**

- Test files are built by a separate script and stored apart from practice data. Training, practice
  and curriculum code take the practice space and `dev` as their only inputs, and a unit test asserts
  that no test family, held-out edge or held-out tool appears in any training file.
- Test splits are scored once per final adapter by a separate script. Every access is appended to a
  committed log with the configuration hash.
- No setting is changed after a test result is seen. Later analyses are labelled post-hoc.
- No model judges anything. The teacher is used for demonstrations and for the pre-freeze quality
  gate only.

**Uncertainty.** A hierarchical bootstrap resamples training seeds, then families, then tasks within
families; arm comparisons are paired on families and tasks. A mixed-effects logistic regression with
family and seed as random effects is reported as a sensitivity check.

**Multiplicity.** Each experiment has a short pre-registered list of confirmatory contrasts (H1; H2
both parts, H3, H4; H5), Holm-adjusted within the experiment. Everything else is exploratory.

**Power.** Using the variance components measured in Study 01
([audit tables §5](audit_tables.md)):

| design | detectable difference between two arms (80% power) |
|---|---|
| Study 01: 16 templates, 3 seeds | 17–24 points |
| Study 02: 600 families × 2 tasks, 5 seeds | 4.6–5.3 points if the seed effect is small (sd 0.02); 9–10 points if it is large (sd 0.05) |

Experiment 1 measures the seed effect on the new environment. If its standard deviation exceeds 0.03,
Experiment 2 runs 8 seeds per main arm.

## 9. Threats and controls

| threat | control |
|---|---|
| Practice gains come from extra optimisation or self-distillation, not from the environment | `practice-unverified` control; identical final training for all arms |
| The verifier has loopholes the learner exploits | strict success reported beside success; a widening gap between them across cycles is the pre-specified warning sign; audit of accepted episodes each cycle |
| Test families leak into training or curriculum | sealed splits, hash-based split assignment, unit test, access log |
| The generator's own style is what is learned | v1 anchor: hand-written tasks and tools never used in training; reserved phrase variants |
| A weak teacher makes distillation look bad | teacher chosen by measured acceptance rate; `oracle` arm bounds what perfect demonstrations give |
| Result specific to one model | 1.5B check; second model family |
| Seed noise mistaken for an effect | 5 seeds, hierarchical intervals, seed count raised if Experiment 1 shows a large seed effect |
| Hyperparameters favour one arm | chosen once on oracle data in Experiment 0 and fixed |
| Flawed evaluation tasks | quality gate before freezing, with decisions logged |
| Synthetic environment | stated as a limit; an external function-calling benchmark can be scored at the end as an exploratory check |

## 10. Compute, cost and tracking

Details and the execution path are in the [implementation plan](IMPLEMENTATION_PLAN.md).

- **Where.** Hugging Face Jobs: one command runs a script on one GPU and the job ends with the script.
  It has a built-in timeout, prepaid credits act as a hard spending cap, and the same account stores
  models, adapters and checkpoints. Modal is the fallback.
- **GPU.** One 24 GB GPU is enough for every run. Planning price: $0.80 per hour (L4).
- **Budget.** About 300 GPU-hours and $260 for the whole programme including a 1.5× contingency;
  $27 for Experiments 0 and 1. Throughput figures behind these numbers are extrapolated, so
  Experiment 0 starts with a ten-minute calibration and the estimates are replaced by measurements
  before Experiment 1 is queued.
- **Tracking.** As in Study 01: JSONL per run, committed to the repository, with tables generated by
  script. Adapters and traces go to private Hub storage with their hashes in the repository. A live
  dashboard is optional.

## 11. Stop/go decisions

| after | condition | decision |
|---|---|---|
| Exp. 0 | reproduction within 5 points of Study 01 | proceed |
| | reproduction fails | fix the stack; nothing else runs |
| | `eval_ood` ≥ 60% under conventional settings | the Study 01 gap was largely a training artefact; re-baseline, then proceed |
| | measured cost per run > 2× estimate | re-plan the budget before Experiment 1 |
| Exp. 1 | H1 supported for 0.5B | Experiment 2 on 0.5B |
| | H1 refuted for 0.5B, supported for 1.5B | Experiment 2 on 1.5B |
| | H1 refuted for both | stop the loop programme; report; investigate task representation |
| | held-out success at D = 1,600 below 40% for the chosen student | the task space is too hard for this size; simplify the held-out definition or raise model size before Experiment 2 |
| Exp. 2 pilot (one seed, two cycles) | yield ≥ 15% of attempted tasks; false-accept rate ≤ 2%; cost within 2× estimate | run the full experiment |
| Exp. 2 | H2 supported, or gap closed by at least half of what `oracle` achieves | Experiments 3 and 4 on the practice arm |
| | otherwise | Experiment 3 on oracle data; report the price of autonomy; Experiment 4 dropped |

Every experiment has a spending cap (§10 of the implementation plan) and needs explicit approval
before it is queued.

## 12. Towards recursive self-improvement and modular specialists

**Demonstrated so far (Study 01).** A small model can be specialised cheaply. What it learns is tied
to the structures it saw.

**Tested by Study 02.** Whether diversity closes the gap (H1), whether the model can supply its own
training signal from a verifier (H2, H4), whether it can direct its own practice (H3), and whether it
keeps what it learned (H5).

**If those hold, the next steps are concrete.**

- *A loop whose task space grows.* The limit on self-improvement in this design is the environment: a
  learner can only improve at what the verifier can check. Experiment 4 adds domains through an
  admission gate. A system that proposes new verifiable tasks needs either a larger model as
  environment designer or a student capable of writing checked environments, which a 0.5B model is
  not.
- *Specialists.* Experiment 3 measures whether per-domain adapters with a programmatic router keep
  skills isolated and updatable. Published results suggest specialists will not combine into skills
  none of them was trained on, so a jointly trained core plus routed specialists is the more
  plausible architecture than specialists alone.

**Speculative.** Recursive self-improvement in the strong sense, a system that improves its own
learning procedure without bound, is not what these experiments test. They test bounded
self-training inside a task space that people define and a verifier that people control. Nothing here
would show that the bound can be removed.

**Safeguards kept throughout.** The learner acts only on simulators and has no network or real side
effects. Verifier, evaluation code and test data are outside anything the loop can write. Each stage
needs approval and has a spending cap. A drop of more than 5 points on the v1 anchor or on `test_id`
between cycles stops a run for inspection.

## 13. Decisions and uncertainties for review

1. **Teacher and Anthropic's terms.** Anthropic's Usage Policy lists as prohibited: "Use outputs to
   train an AI model (e.g., 'model scraping' or 'model distillation') without prior authorization from
   Anthropic". The Commercial Terms (D.4) bar using the services "to build a competing product or
   service, including to train competing AI models" (both fetched 2026-11-10). Training the student on
   Fable 5.2 demonstrations is distillation in that sense. The design therefore defaults to an
   open-weight teacher under a licence that permits it, served on the same GPU, and uses Fable 5.2 for
   design, implementation and analysis. With written authorisation, a Fable arm can be added through
   the same interface for an estimated $110–220. Experiments 0 and 1 use no teacher. The generator's
   request phrasing will be written with Claude's help, as Study 01's templates were; include that in
   any authorisation request if you want certainty.
2. **Compute access.** This session cannot reach huggingface.co or arxiv.org. Launching and monitoring
   jobs from here needs huggingface.co allowed in the environment's network settings and a Hugging Face
   token stored as an environment secret. Whether Jobs requires a paid PRO account was not confirmed.
3. **Throughput is extrapolated.** The cost estimates could be wrong by a factor of two in either
   direction until Experiment 0 measures them.
4. **Literature verification is abstract-level.** See [LITERATURE.md](LITERATURE.md).
5. **The task generator is the largest piece of new code** and the main schedule risk. Its acceptance
   tests are in the implementation plan.
