# Experiment 2, amendment 2: preparation of Stage 3

*Opened 2026-11-11 on the owner's instruction of that date ("Proceed with Stage 3 preparation only. Do not
launch the main experiment or score any sealed split."). It amends the [plan](PLAN.md) and
[amendment 1](AMENDMENT_1.md). The sections are recorded in the order their work is done, each before the
work it governs; the commit history dates them. Section 3 was recorded first, before the one paid job of
this preparation. Nothing here starts Stage 3 training or scores a sealed split.*

## 1. Primary endpoint (owner's decision 1)

**From Stage 3 on, the primary endpoint is correctly attributed, shortcut-free success of the final adapter on
`test_comp`**: the field `success_clean` of amendment 1, which is success by the original validator, and each
requested fact given for its own part (`ccl.verify.parts_ok`), and no shortcut in the trace
(`ccl.verify.shortcut`). The plan's endpoint was success by the original validator.

- H2(a), H2(b), H3 and H4, their thresholds (10, 7.5, 5 and 5 points), the two intervals that must both meet
  a criterion, and Holm's correction are unchanged; they are evaluated on the new endpoint.
- **Reported separately for every final adapter and every contrast**, never as a criterion: (i) success by the
  original validator, the measure of Experiments 0 and 1 and of the plan; (ii) complete-world adversarial
  accuracy, `success_parts` on the twins of §2, beside the same adapter's result on the originals of those
  tasks. A contrast whose sign or verdict differs between the primary endpoint and (i) or (ii) is named as
  such in the report's summary.
- **Robustness label, descriptive.** H2(a) is called *robust to complete worlds* if the gain of
  `practice+curriculum` over `seed` on the `test_comp` twins has both intervals above zero. It changes no
  verdict.
- The interim look of block 3A on `dev` (`practice+curriculum` at least 5 points above `seed` in the mean of
  three replicates) and the replicate rule (pooled standard deviation above 0.03) use the primary measure.
  The stop rule inside a run (a fall of more than 5 points on `dev` between cycles) keeps the original
  validator, as amendment 1 fixed it and as both pilots ran it.

**This is a change to a field the plan froze** (§6: endpoints are "not to be changed after any `dev` result
of a trained Experiment 2 adapter is seen"). It is made by the owner after two pilots' `dev` results and
before any test split has been scored for any Experiment 2 adapter. Its direction is known from `dev`: the new
measure is stricter, and stricter on practice adapters than on the seed adapter (cycle-2 adapter 44.2% to
40.8%; seed adapter 25.8% to 25.0%), so it lowers the expected gain that H2(a) tests. It rests on a detector
that describes the forms of the shortcut seen so far; the twins are the check that does not.

## 2. Complete-world twins of the test tasks (owner's decision 2)

Manifest: [`configs/study02/e2/twins.json`](../../configs/study02/e2/twins.json), written by
`python scripts/s2_score.py --twins`.

| split | tasks | twins | SHA-256 (first 16) | unchanged from version 1 |
|---|---|---|---|---|
| `v2/test_comp_adv` | 1,200 | 866 | `eb0889f67c5b695a` | 852 |
| `v2/test_depth_adv` | 400 | 400 | `32ddd79a18343f1f` | 400 |
| `v2/test_tool_adv` | 400 | 278 | `7f954bc23dcc7c7a` | 270 |
| `v2/test_domain_adv` | 400 | 190 | `69d2d379b5d00075` | 190 |
| `v2/test_id_adv` | 400 | 218 | `8d84e0d5b0caf7f4` | 216 |
| `v2/dev_adv` | 600 | 350 | `8689eeba0c670adf` | 338 |

**Version 2 of the twins.** The review of PR #3 (§7) found the set of amendment 1 one step short: in 12 of
the 350 `dev` twins a walk from another candidate along the rest of the gold path's calls still met a missing
record or an empty list, and the test that claimed otherwise could not see it. Version 2 is version 1 with
those walks followed and what they miss created (`ccl.adversarial.leaks`, `complete_world(repair=True)`);
every other twin is byte-identical. Version 1 (`c19010c9cb77bfeb`) stays reproducible
(`adversarial_dev(version=1)`) and is what the repeated pilot was scored on; on its 12 affected twins all
three adapters scored 0. Everything from here on uses version 2. The manifest records, per split, that no walk
from another candidate misses a record.

- **Rule**, the one of amendment 1 applied to every split: the twin of every task in which a gold call
  depends on a list selection and whose expected status is done or blocked (`ccl.adversarial.twins`). The
  generator is deterministic and both versions are tested against their recorded hashes.
- **Isolation.** `scripts/s2_score.py` stays the only code that opens a test split. `ccl/adversarial.py`
  names no split and opens none but `dev`; the training loop, the engine and the job script of the practice
  arms cannot reach a test split or a twin of one (tested). The twins are never written to disk: the scorer
  rebuilds them and refuses to score a set whose count or hash is not the recorded one.
- **What was looked at.** The manifest was built without a model. The program checked, for every twin, that
  the oracle solves it with the transcript it has on the original and that request, gold graph, answer,
  catalogue and faults are the original's; it printed counts only. No test task and no outcome of any model
  on a test split was displayed or read. The one thing learned about the test splits is how many of their
  tasks contain a list selection.
- **Scoring.** Each final adapter is scored on a twin split in the job that scores the original, once; the
  first access is written to the run's access log like any other test split.
- **A guard changed.** `MAX_RECORDS`, the assertion that a completed world did not run away, was 3,000 and
  is 10,000: six-hop tasks of `test_depth` need up to 3,708 added records. It is an assertion, not a
  truncation, and no world changes with it.

## 3. Teacher re-qualification under protocol 2 (recorded before its job)

**Why.** Under protocol 2 no teacher qualifies on the Stage 1 data (amendment 1, §7): `Qwen/Qwen3-32B` has
67.5% overall and 30% at four hops, `Qwen/Qwen3-14B` 55.5% and 18%. That reading is of attempts collected
under protocol 1, where a task whose greedy attempt succeeded by the shortcut got no retry. The owner
approved a re-qualification with a hard cap of 2 USD and unchanged thresholds.

**Unchanged.** The thresholds (accepted on at least 60% of the 200 tasks and on at least 40% of the four-hop
ones); the 200 tasks (the first 50 of each number of hops in replicate 0's stream); the few-shot prompt; one
greedy attempt and one retry sampled at temperature 0.7, top-p 0.95, seed `e2|teacher`; the engine settings
of the Stage 1 qualification (R1 to R4, 16 rows, token budget 28,000; R6 is not used); the code path
(`scripts/s2_e2.py teacher`). The licence requirement of the plan.

**What protocol 2 changes, by construction.** Acceptance is `ccl.s2.rejection(..., protocol=2)`: an attempt
that reaches the right answer by the shortcut, or gives a fact for the wrong part, is rejected, and the task
then gets its retry like any other rejected task. The result file records `protocol: 2`.

**Candidate.** `Qwen/Qwen2.5-32B-Instruct` at revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`
(Apache-2.0, not gated, dense, 32.8 billion parameters, `Qwen2ForCausalLM`: the architecture family of the
student, which is the one the engine was calibrated on). Reasons, all known before the job:

- Re-running `Qwen3-32B` is not expected to qualify. Of its 50 greedy four-hop attempts, 13 are accepted
  under protocol 2, 13 succeed by the shortcut and 24 fail; its four-hop retries were accepted in 2 of 24
  cases. With retries for the 13 as well, about 16 of 50 is the expectation, and 20 are needed.
- The newest Apache-2.0 candidates of similar size (`Qwen/Qwen3.8-27B`, `Qwen3.6-27B`, `Qwen3.5-27B`) mix
  attention layers with recurrent layers (`layer_types`: `full_attention`, `linear_attention`). The engine
  keeps a batch's cache between turns and switches off, by attention mask, positions that belong to no row's
  prompt. A recurrent state cannot be masked after the fact, so the engine would compute wrong continuations
  for these models, and a correct rollout path for them is new work that cannot be validated inside 2 USD.
  They are not tested here. This is listed as an open option for the owner.
- Mixture-of-experts candidates were excluded in Stage 1 for their cost on this engine (plan, deviation 3).

**Rule for a second candidate.** After the first job settles, a second candidate
(`Qwen/Qwen2.5-Coder-32B-Instruct`, Apache-2.0, same architecture) is tested only if the first did not
qualify and what is left of the cap covers the worst case of a job whose time limit is at least 1.25 times
the first job's measured duration. Otherwise it is not run.

**Budget.** `configs/study02/budget_e2_teacher2.json`: cap 2.00 USD, own ledger
(`results/study02/e2/ledger_teacher2.jsonl`), provider baseline 56.192762 USD. One job on `a100-large`
(2.50 USD per hour) with a 40-minute limit: worst case 1.75 USD. The Stage 1 run of a model of the same size
took 20 minutes (0.88 USD).

**Reading the result.** Qualifies: both thresholds met on a complete run (both passes finished). Reported
beside it: acceptance by hops, the count of attempts rejected for the shortcut and for a wrong part, and the
acceptance the same attempts would have under protocol 1. Every accepted attempt is replayed on a fresh
simulator in a separate process (`scripts/s2_e2.py replay`). **If no candidate qualifies**, H2(b) is
reported as not tested, as the plan provides (§3), Stage 4 is not proposed, and the thresholds stay as
they are. The accepted demonstrations of a qualifying teacher are not used for anything in this preparation.

## 4. The corrected engine (owner's decision 4)

Evidence: [`results/study02/e2/stage3_prep.md`](../../results/study02/e2/stage3_prep.md) §1, computed by
`scripts/s2_e2_stage3.py` from committed files.

| comparison | result |
|---|---|
| greedy, two Experiment 1 adapters, 400 own-family tasks each, against Experiment 1's rows | 4 and 1 outcomes changed (the engine without R6: also 4 and 1); success 88.0% and 88.8% against 87.5% and 89.0% |
| the same episodes, text against full re-encoding of every turn | identical on 48 of 48 checked |
| corrected engine against the engine without R6, same tasks | 4 and 2 outcomes differ; 5 and 3 of 400 episodes differ in length or calls |
| seed adapter on `dev`, greedy, 600 tasks | 25.7% and 25.8%; 3 tasks up, 2 down; 32 episodes differ in length or calls |
| cycle 1 of the two pilots, sampled at 0.8, same task, seed and attempt number: 5,664 pairs | identical text in 82.9%; a different outcome in 72 (1.3%); success 24.9% and 24.8% |
| speed, greedy | 0.43 s per episode against 0.66 (without R6) and 1.62 (Experiment 1) |

**Known numerical differences.**

1. R6 computes the attention of a decoding step in float32 over grouped heads; the library path computes it in
   bfloat16 over expanded heads. The two round differently, so a token whose two best candidates are nearly
   tied can come out differently, and an episode that changes one token can change its outcome. Measured: about
   1% of greedy outcomes and 1.3% of sampled outcomes differ, in both directions, with no shift of the mean
   beyond 0.5 points on 400 tasks and 0.1 points elsewhere.
2. A sampled attempt is therefore **not** reproducible token for token across the two engines (83% are), and
   an arm's attempts can be regenerated exactly only on the engine that produced them. Every attempt's text
   is stored, and acceptance is recomputed from stored text by replay, so no result depends on regeneration.
3. Numbers from this engine are not interchangeable at the level of single tasks with Experiment 1's rows;
   at the level of a split the measured differences are 0.5 points or less.
4. The first version of R6 was wrong on the GPU (113 of 400 outcomes changed) and correct on the CPU, where
   the tests run. A CPU test cannot certify this engine; only a calibration on the GPU can.

**Not measured:** the corrected engine against itself. Experiment 0 measured 0 changed outcomes in 720 for the
Experiment 1 path scored twice.

**Rule for Stage 3.** Every arm, every `dev` scoring and every final scoring of Experiment 2 uses the
corrected engine, so no contrast between arms mixes engines. The first Stage 3 job scores the seed adapter of
replicate 0 on `dev` twice more (about 0.3 USD); if more than 6 of 600 outcomes differ between two runs of
the same engine (1%, the difference measured between engines), Stage 3 stops before any training. With the
evidence above recorded, the engine is used for the test splits too; the plan allowed that only for an engine
that passed its calibration, and this one met the outcome criterion and missed the speed criterion (0.43 s
against 0.36 s), which bears on cost and not on correctness.

## 5. The audit (owner's decision 5)

**The repeated pilot's audit, restated.** 960 episodes were accepted. 200 were sampled and each was checked
by program. 39 were read: the 19 the program flagged, which are all 19 accepted episodes that are not strictly
successful, and 20 unflagged ones drawn blind. 161 were checked by program only. No false accept was found.
"0 false accepts in 200" in amendment 1 §8 and in the report meant this; read as 200 episodes adjudicated by
reading, it overstated the reading by a factor of five. Reader: the assistant that ran the pilot; no person
and no second reader.

| one-sided 95% upper bound on the share of false accepts | strictly successful (941) | all accepted (960) |
|---|---|---|
| relying on the program for the 181 unflagged sampled episodes | 1.6% | 1.6% |
| from the 39 read episodes alone | 13.9% | 13.6% |

The program's check of an unflagged episode is strong (its calls are the oracle's, its answer states no value
the reference lacks, it used no unobserved identifier, it passes the rule's checks), and the 20 blind readings
found it right 20 times. But the 2% gate was met as a point estimate; **reading alone does not show the rate
to be under 2%**. The first pilot's audit is restated in the same terms: 200 sampled and checked by program,
80 read (all 80 that the program flagged; no blind reading), 54 false, a weighted rate of 5.6%.

**The audit of Stage 3, pre-registered.** After block 3A and again after 3B, before any test split is scored:

1. Population: every accepted episode of the verified practice arms of the block.
2. Sample: 600, seed `e2|audit|3A` (or `3B`): the not strictly successful up to 300, the rest strictly
   successful, weighted back to the strata. All 600 are checked by program with the flags of amendment 1 §5.
3. Read: 150 of the flagged, drawn at random if there are more (all if fewer), and 150 unflagged drawn blind.
   150 is the number at which no false accept found gives a one-sided 95% bound under 2% from reading alone.
4. A false accept is as defined in amendment 1 §5. Verdicts are written before the rate is computed.
5. Gate, unchanged: the weighted share of false accepts is at most 2%. Reported with it: the one-sided 95%
   upper bound by stratum and overall, from reading alone and relying on the program, and the three counts
   (sampled and checked by program, read, false).
6. Second reading: 60 of the read episodes (30 flagged, 30 blind) are read again by an independent instance
   that does not see the first verdicts; agreement is reported. A person's reading of any part is the owner's
   to add.
7. If the gate fails: stop, as in amendment 1 §5.7.

## 6. Stage 3 as it would run (design; not started)

Unchanged from the plan: student and training settings, replicates, stream and seed sets (frozen hashes),
arms, cycles (6 of 480 accepted, at most 2,400 tasks), sampling (4 attempts at 0.8), curriculum, matched
training budgets, hypotheses and thresholds, intervals, the replicate rule, the stop rules.

Changed by amendments 1 and 2: protocol 2 for every buffer; the corrected engine everywhere; the primary
measure (§1); the adversarial `dev` set at every `dev` scoring and the twins at every final scoring (§2); the
audit (§5); the engine's self-repeat check before training (§4).

Replicate 0's `practice` arm continues from the repeated pilot's cycles 1 and 2 (`r0_practice_p2`), which ran
under exactly this protocol, engine, stream and seeds. Replicate 0's seed adapter is the one of Stage 1.

| block | what | estimate, USD | proposed hard cap, USD |
|---|---|---|---|
| 3A | replicates 0-2 of `seed`, `oracle`, `practice`, `practice+curriculum`, `practice-unverified`, `oracle-matched`; `dev` and `dev_adv` every cycle; audit; then the interim look | 104 | 120 |
| 3B | replicates 3-4 of the four main arms; audit; test splits and twins for every final adapter of 3A and 3B, once | 50 | 60 |

Estimates are at the repeated pilot's measurements (yield 37%, 0.29 s per attempt, 3.19 s per step) and
include job overhead and a 25% contingency; the twins are 13 and 6 USD of them. On the Experiment 1 scoring
path for test splits the two blocks would be 171 and 83 USD. In the cost model the test splits of a block's
adapters are costed inside the block; they are scored only after the last block that runs.

Stage 4 (teacher arm) is conditional on H2(a), as the plan has it. With the teacher of §3 its demonstrations
would cost about 63 USD for five replicates and 38 USD for three (4.6 GPU-seconds per accepted demonstration
on `a100-large`, with overhead and contingency), and its adapters 8 to 13 USD. The plan's rule (five
replicates only if the demonstrations are projected at 60 USD or less) gives three replicates.

## 3a. Outcome of the teacher re-qualification (added after the job; §3 is unchanged)

| model | attempts | accepted, protocol 2 | by hops 1 to 4 | rejected for the shortcut | qualifies |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-32B-Instruct` @ `5ede1c97bbab` | 239 (200 greedy, 39 retries) | **164 of 200 (82.0%)** | 98%, 86%, 74%, **70%** | 16 attempts | **yes** (60% and 40%) |

All 164 accepted demonstrations replay on a fresh simulator. Under protocol 1 the same attempts give 84.5%
and 74%. The job ran with the detector as it was before the correction of §7; recomputed with the corrected
detector the result is the same on every task. Cost 0.67 USD of the 2.00 USD cap (14.7 minutes on
`a100-large`); the second candidate was not run, as the rule provides; the budget is closed. **H2(b) can be
tested.** The demonstrations are not used for anything before Stage 4.

## 7. Review of PR #3 (owner's decision 6) and what it changed

Two independent reviewers read the branch at `a68dfcd`; their findings, what was verified, and what was done
are in [docs/study02/REVIEW_PR3.md](../../docs/study02/REVIEW_PR3.md). Changes to rules and measures:

1. **The shortcut detector did not implement its own definition** (amendment 1 §3: a probe and a hit on
   *different* values of a column). The code dropped a hit whose value had itself failed once, so a probe of
   other candidates went unseen if the right candidate was first asked with a wrong tool. Corrected in
   `ccl.verify.shortcut`, with regression tests. Effect on what is recorded: no accepted episode of either
   pilot changes (11 attempts do, all failures); no `dev` or adversarial figure of the repeated pilot changes
   (recomputed from its traces); two demonstrations of `Qwen3-14B` are now rejected (55.5% to 54.5%); the
   re-qualification of §3a is unchanged. The definition is not changed, so the protocol stays version 2.
2. **Twins, version 2** (§2).
3. **The multi-part check has a weak third tier.** If an answer names only some of the parts, the check falls
   back to counting occurrences and can pass a swapped answer or reject a correct one in other words. A
   student trained on the reference format answers under the first tier (172 of 176 multi-part episodes in
   the buffers; none under the third). Not changed now. Every Stage 3 report gives the share of multi-part
   successes decided by each tier, and third-tier successes are read if they exceed 2% of an adapter's
   multi-part successes. It matters more for a teacher, which answers in its own words (40 of 44).
4. **Adversarial success does not show that the selection rule is applied.** A policy that makes every call
   correctly and takes a row at random at each selection would score about 22% on the adversarial `dev` set
   (16% by the reviewer's count of list sizes); the cycle-2 adapter scores 18.3%. Measured directly
   (`ccl.adversarial.selection_picks`): the gold row is the first pick in 44.5% of the seed adapter's
   selections on `dev` and 48.8% of the cycle-2 adapter's, against 34% at random and 100% for the oracle; in
   rollouts 39.6% and 44.8% against 35.5%. **Most of the gain of two cycles is getting further down the
   chain; the selection itself improved by a few points.** A lucky pick that the validator passes is a correct
   trajectory and not a false accept, and it is how outcome-verified practice finds what to learn from; but
   no measure so far separates the two.
   **Pre-registered for Stage 3, secondary:** first-pick accuracy at selections against chance, for every
   final adapter and split (`picks`, `picks_gold`, `picks_chance` in every metrics row), and adversarial
   success beside the random-pick level by number of selections. The robustness label of §1 additionally
   requires that the first-pick accuracy of `practice+curriculum` exceeds the seed adapter's with both
   intervals above zero.
5. **A resumed cycle now refuses attempt files written under another protocol; a stop rule leaves a
   `STOPPED` file that later cycles of the arm refuse to pass; the launcher also checks the provider's own
   usage figure against the cap before it reserves.**
6. **The audit's estimator** weights a false accept found in the blind reading by the whole unflagged
   stratum, not by one sampled episode (it made no difference at zero found; the Stage 3 audit of §5 uses
   the corrected weight).
7. **Recorded as deviations in the plan** (new items 4 to 6): the acceptance rule and the endpoint were both
   changed after `dev` results of trained Experiment 2 adapters, which the plan's §6 forbids; the rule, the
   detector and the endpoint were developed on replicate 0 of `practice`, which is also a confirmatory
   replicate. Stage 3 reports every contrast with and without replicate 0.

Known and not changed: in the twins the picked row's list is often longer than the other candidates' lists
(2 to 5 records against 1 to 2), which is not an existence signal but would be a tell if complete worlds were
ever used for training; an exception raised by the provider's client after a job was created would release
its reservation (the provider check of item 5 bounds the damage).
