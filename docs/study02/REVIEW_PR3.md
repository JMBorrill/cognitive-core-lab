# Review of PR #3 (Experiment 2: Stages 0 to 2, amendment 1, the repeated pilot)

*2026-11-11. Two independent reviewers (separate instances, each in its own checkout of `a68dfcd`, neither
shown the other's work or the author's conclusions) read the branch: one the code, leakage, tests and budget
enforcement; one the protocol and the reports against the generated results. Neither read a test split,
submitted a job or changed a file. This file lists every finding, whether the author re-derived it, and what
was done. "Verified" means recomputed by the author from committed files (or the bucket's traces), not taken
from the reviewer.*

## Findings that changed code, a rule or a measure

| # | finding | verified | disposition |
|---|---|---|---|
| C1 | `ccl.verify.shortcut` subtracts probed values from the hits (`if p and h - p`), so an elimination is missed when the right candidate also failed once. Both reviewers; two accepted `Qwen3-14B` demonstrations (stream indices 16, 98) | yes: 11 attempts of the pilots and 5 of the teachers change, 2 of them accepted demonstrations; no accepted practice episode and no `dev` figure of the repeated pilot changes | **fixed**, regression tests; amendment 2 §7.1 |
| C2 | The adversarial completion is one step short on 12 of 350 `dev` twins, and `wrong_walk` in the tests cannot see it (it silently returns to the gold arguments after the first off-path list) | yes: `ccl.adversarial.leaks` finds the same 12; all three adapters scored 0 on them | **fixed** as twins version 2 (repair along every walk; a further full round uses up identifiers on six-hop tasks); version 1 kept reproducible; exhaustive check in the tests and in the manifest; amendment 2 §2 |
| P1 | "Practice improves real selection" is not shown: a random-pick policy with perfect execution scores about as much on the adversarial set as the cycle-2 adapter; first picks are a few points above chance | yes: 22% by the author's count of list sizes (16% by the reviewer's) against 18.3%; first picks 44.5% (seed) and 48.8% (cycle 2) on `dev` against 34% at random | **new measure** `selection_picks`, in every metrics row; pre-registered secondary for Stage 3 and added to the robustness label; reports' errata; amendment 2 §7.4 |
| P2 | "0 false accepts in 200": 39 were read, 161 checked by program only; bounds 13.6% from reading alone, 1.6% relying on the program. The 20 blind readings are 19 one- and two-hop episodes and their notes are one sentence | yes (counts and bounds); the hop mix of the 20 not re-derived | **reworded everywhere**; bounds in the report and in `stage3_prep.md`; Stage 3 audit sized at 150 + 150 readings with a second reader; amendment 2 §5 |
| C4 | A resumed cycle trusts attempt files written under another protocol (a probe leaves chunks without state) | by reading the code | **fixed**: assertion on each chunk's protocol, tested |
| C6 | The launcher checks the cap against the local ledger only; a stale or missing ledger can hide spend. A reservation is released on any client exception | by reading | **partly fixed**: `submit` now also checks the provider's usage figure against the cap. The release on exception is unchanged and recorded as known |
| C7 | A stop rule lives only in the exit code of the job that detects it; the `dev` rule is skipped silently without prior metrics | by reading | **fixed**: `STOPPED` file that later cycles refuse to pass; missing prior metrics is an error; tested |
| C8 | The audit's estimator would under-weight a false accept among the blind reads | by reading | corrected in the Stage 3 audit's definition (amendment 2 §7.6); no effect at zero found |
| P5 | Acceptance rule and endpoint changed after `dev` results against the plan's freeze clauses, not cited; developed on a confirmatory replicate; detector agreement in-sample | yes (the clauses are §6 and §7 of the plan) | **recorded** as deviations 4 to 6 of the plan; Stage 3 reports contrasts with and without replicate 0 |
| P4 | "No teacher qualifies under protocol 2" was a lower bound: tasks solved by the shortcut had had no retry. The amendment's "20 of 155" against Stage 2's "11 of 155" was unexplained (the 9 extra are parallel fan-outs the earlier detectors miss) | yes (retry mechanism); the 9 not re-read | **superseded** by the re-qualification under protocol 2: `Qwen/Qwen2.5-32B-Instruct` qualifies (amendment 2 §3a). The explanation of 11 against 20 is recorded here |

## Findings recorded as limits (no change now)

| # | finding | status |
|---|---|---|
| C3 | `parts_ok`: with only one part named the check falls to its third tier, which checks neither order nor attribution (a swapped answer can pass; "respectively" answers and an e-mail containing the anchor name are rejected). Students answer under tier 1 in 172 of 176 buffered multi-part episodes, none under tier 3; teachers under tier 2 in 40 of 44 | not changed before Stage 3; tier shares reported per adapter, tier-3 successes read above 2% (amendment 2 §7.3). To be tightened before a teacher arm |
| C5 | In the twins the pick's list has a length no other candidate has in 186 of 232 selections followed by a list (`CHILDREN = 2`) | not an existence signal; recorded. A tell if complete worlds were used for training |
| P6 | "The endpoint favours practice" compared oracle-trained adapters on own-family tasks (1 of 705) with practice adapters on `dev`; in the repeat, shortcut episodes on `dev` rose 9, 10, 35 with none in the buffers, so "learns it from its own accepted episodes" is not the whole mechanism | accepted; errata in both reports. No oracle-trained adapter has a `dev` shortcut count yet; Stage 3's `oracle` arm gives one |
| P7 | The two pilots are compared with unlike definitions ("39.8% and 40.8%": like for like 39.8% and 41.5%); after cycle 1 the same measure differs by 4.0 points between the runs; "its rate in rollouts barely moves" compares the seed and cycle-1 adapters only | accepted; erratum in the repeat's report. One replicate cannot separate these |
| P8 | Yield intervals: 220 of 1,143 cycle-2 tasks are new instances of families already trained on (yield 83% against 35%); on untrained families the one- and two-hop changes vanish, three and four hops hold (+14.0 and +7.6 points); intervals cover task sampling only | accepted, not re-derived; erratum. Stage 3 reports yield separately for families seen and not seen in training |
| P10 | Cost figures are point values from two cycles applied to six, while seconds per attempt rose 0.27 to 0.32 in one cycle | accepted: the caps proposed for Stage 3 are hard, and the first cycles of 3A are compared with the model before the rest runs |
| P11 | Figures that include the empty-list form found after the first audit are not labelled post hoc in four places; the pilot-to-pilot comparisons are not pre-registered | accepted; erratum |

## Number and text mismatches (errata appended to the reports; originals not rewritten)

- Stage 2 report §4: "In each of the 54 cases no other attempt at the task passed." 21 of the 54 had another passing attempt; none had a passing attempt free of the shortcut.
- Stage 1 report §6: "A practice replicate costs 11.4 USD ... 1.9 scoring" is the figure with test splits on the new engine; by the plan's rule it is 12.83 (3.32 scoring), as `budget_stage1.md` has it.
- Stage 1 report §4.6: "a third of what this teacher would hand the student" is not strict: 35 of 155 (23%) for `Qwen3-32B`.
- Stage 1 report §4.1 omits that cached decoding matched full re-encoding on 22 of 24 checked tasks for one adapter (`stage1.md` has it).
- Plan, deviations 1 and 3: clock times later than their commits (plan, erratum 8).
- Yield 33.3% in the reports is 133 of 400 = 33.25%.

Not re-derived by the author: P7, P8, P11 in detail, the hop mix of the blind readings, the 9 fan-out
demonstrations. They are taken from the reviewer's report and marked as such above.

## Checked and found sound by the reviewers

Every other figure of the three reports against the generated outputs; the ledgers (2.04, 2.00, 1.77 USD)
and replay counts; that each pre-registration was committed before its job; that `rejection`,
`audit_sample`, `cmd_audit` match amendment 1 apart from C1; the oracle under protocol 2 (0 of 5,200
rejected); the twins' gold paths; the loop's determinism, resume and curriculum; the engine's bookkeeping by
reading; **no path from training code to a test split**; `dev[:80]` is used for a logged validation loss
only, with no checkpoint selection.

## Could not be checked by anyone here

The engine's bfloat16 behaviour on a GPU and its handling of fully masked padded queries (hence the
self-repeat check before Stage 3, amendment 2 §4); provider-side billing; who read the audit's episodes and
when (the verdicts and the gate are in one commit). Test gaps the code reviewer listed that remain open:
`submit()` end to end against a stubbed API, the exit paths of `cmd_cycle` on a GPU, an assertion that the
grouped-attention branch is taken.
