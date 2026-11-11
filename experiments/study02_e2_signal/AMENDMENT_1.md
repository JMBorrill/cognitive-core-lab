# Experiment 2, amendment 1: protocol version 2

*Recorded 2026-11-11, before any job of the repeated pilot, on the owner's instruction of that date. It
amends the [plan](PLAN.md) after the [first pilot](../../docs/study02/REPORT_E2_STAGE2.md) failed its verifier
audit. Machine-readable parameters: [`configs/study02/e2/protocol2.json`](../../configs/study02/e2/protocol2.json).
Hypotheses, arms, thresholds, the seed set, the task stream and the training settings are unchanged. No sealed
split is scored and no stage after the repeated pilot is approved.*

## 1. Why

The first pilot found two faults in what counts as a success.

- **A shortcut.** The generated worlds build only the records on a task's gold path in full. A record that is
  merely a candidate in a list refers to records that do not exist and has nothing listed under it. An agent
  can therefore find the right candidate by asking for each one's continuation, instead of applying the
  selection rule the request states. The validator accepts such an episode. Accepted episodes of this kind
  were 9 of 480 in cycle 1 and 45 of 480 in cycle 2, and they were 43 of the cycle-2 adapter's 282 successes
  on `dev`. Adapters trained on oracle episodes used it in 1 of 705 successes.
- **Multi-part answers.** For a question with several requested facts the validator checks that each fact
  occurs somewhere in the answer. A two-part question whose two answers are the same word passed with the
  word given once and a wrong value for the other part.

## 2. Versions

| | protocol 1 (proposal §6; Experiments 0 and 1; first pilot) | protocol 2 (this amendment) |
|---|---|---|
| an episode enters a buffer if | it passes the validator, and the runtime recorded no parse error, unknown tool or invalid argument | the same, **and** every fact of a multi-part answer is given for its own part, **and** the trace shows no shortcut |
| applies to | every data source | every data source: practice, the unverified control (the trace check; it has no validator), oracle episodes, and a teacher's demonstrations |
| code | `ccl.s2.rejection(..., protocol=1)` | `ccl.s2.rejection(..., protocol=2)`; `ccl.verify.parts_ok`, `ccl.verify.shortcut` |
| recorded | — | `protocol` in every attempt row, cycle record and arm state; an arm keeps the protocol it was started under |

Both added conditions read only the episode (its trace and final answer) and what the task states (its
request and reference answer). Neither reads the gold graph. The principle of the proposal therefore stands:
the acceptance rule uses the outcome and runtime observations, and criteria derived from the gold graph are
measured, never used for filtering.

Protocol 1 remains the default of `ccl.s2.rejection`, so the training sets of Experiments 0 and 1 rebuild to
the same hashes (tested). The first pilot's files are not reinterpreted: its arm stays a protocol-1 arm.

## 3. The two checks

**Shortcut** (`ccl.verify.shortcut`), a deterministic function of the trace. A *column* is one field of the
records of a successful list call that returned at least two records. A *probe* is a later call whose argument
is a value of that column and which comes back with "not found" or an empty list. A *hit* is a later call whose
argument is a different value of the same column and which succeeds with a record or a non-empty list. **A
trace with a probe and a hit on one column is a shortcut**, whatever the tools and whatever the order. Not a
shortcut: a retry of the same identifier; an identifier that is not a value of a listed column; "not found"
with no hit.

Checked before this amendment was recorded: on the first pilot's 2,894 admissible attempts it marks the same
128 as the two forms counted in that pilot's report; it agrees with all 80 verdicts of that pilot's audit (53
read as elimination, 26 read as clean, and the wrong-part episode, which also contains a probe); it marks no
oracle episode on `dev`.

**Multi-part answers** (`ccl.verify.parts_ok`). The parts of a question are read from its reference answer
("The ‹subject› is ‹fact›." per part). (1) If the answer repeats every subject, each fact must stand between
its subject and the next. (2) If it names the parts in other words (an identifier or name only that part has,
or, for two questions about one record, a word only that part's subject has), the facts must all follow, or all
precede, the names of their parts, each inside its own part's stretch of the answer. (3) If it does not name
the parts, every fact needs an occurrence of its own. One-fact answers are not affected.

Checked: every oracle answer on `dev` passes (102 of 600 tasks are multi-part); of the first pilot's admissible
attempts on multi-part tasks, 37 of 579 fail, and each of the 37 lacks the reference's sentence for some part;
no teacher demonstration fails (one correct answer in other words and in reverse order passes by rule 2).

## 4. Evaluation: what is preserved and what is added

`dev` (600 tasks) is unchanged, and so is the original validator. Every metrics row keeps `success` and
`accepted` with their meaning. Reported for every adapter, separately:

| measure | meaning |
|---|---|
| `success` | the original validator: comparable with Experiments 0 and 1 and with the first pilot |
| `success_parts` | and each fact given for its own part |
| `success_clean` | and no shortcut in the trace |
| adversarial success | `success_parts` on the adversarial set, beside the same adapter's success on the 350 original tasks |

**The adversarial set** (`ccl.adversarial`, split `v2/dev_adv`): the 350 `dev` tasks in which a gold call
depends on a list selection (50, 136 and 164 with two, three and four hops), each rebuilt on a world where,
for the tools of its gold path, every reference leads to a record and every record has records listed under
it, to the depth of the task. Request, catalogue, gold graph and answer are those of the original, and every
original record is unchanged, so an agent that applies the rule sees exactly the same episode on both: the
oracle's transcript is identical on all 350 pairs (tested). On the original worlds a walk from another
candidate runs into a missing record or an empty list in 300 of the 304 tasks where such a walk exists; on
the twins in none (tested). The set is generated deterministically; its SHA-256 is recorded and tested. It is
built from `dev`, is evaluation data only, and no stop rule reads it.

The in-run stop rule on `dev` (a fall of more than 5 points between cycles) keeps using `success`, as in the
first pilot. No primary endpoint is changed by this amendment: Stage 3 is not approved, and its endpoint is a
decision for that approval, to be made with these three measures in hand.

## 5. The audit, pre-registered

After cycle 2 of the repeated pilot, before its report:

1. **Population**: every accepted episode of both cycles.
2. **Sample**: 200, drawn with seed `e2|audit`: all that are not strictly successful, up to 100, and the rest
   from the strictly successful, each weighted back to its stratum (`ccl.loop.audit_sample`).
3. **Flags**, by program, per sampled episode: its calls differ from the oracle's; its answer states a value
   the reference answer lacks; a call uses an identifier that was neither in the request nor observed; it
   uses two candidates of one list in later calls, whatever came back (`ccl.verify.enumerates`, wider than the
   shortcut); it fails the rule's own shortcut or per-part check (which no buffered episode should).
4. **Reading**: every flagged episode, and 20 unflagged sampled episodes drawn with seed `e2|audit|blind`, as
   a check on the flags. Verdicts go to `audit_verdicts.json` before the gate is computed.
5. **A false accept** is an accepted episode that (i) states a requested fact for the wrong record or among
   other candidates' values; (ii) reached a required record by an identifier it had not observed; (iii)
   reached its record or its answer other than by what the request states, through any property of the
   simulated world or of the validator, whether or not `ccl.verify.shortcut` describes it; or that fails the
   rule's own checks.
6. **Gate**: the weighted share of false accepts is at most 2% (4 in 200), and every episode marked for
   reading has a verdict. An unread episode fails the gate.
7. **If the gate fails**: stop. Nothing further runs. The next step is to examine whether the world
   generation must be rebuilt so that candidate records are complete in training too.

The other gates of the pilot stand: at least 20% of attempted tasks accepted in both cycles; cost per cycle
within twice the cost model's; no fall of more than 5 points on `dev`.

## 6. The repeated pilot

Replicate 0, `practice`, cycles 1 and 2, as Stage 2 defined them, with these differences only:

- **protocol 2** for acceptance, replay and buffers;
- **the corrected engine** (R6, calibrated in Stage 2: 4 and 1 of 400 outcomes changed, the same text as full
  re-encoding on 48 of 48 checked tasks) for rollouts, `dev` and the adversarial set;
- a new arm directory, `r0_practice_p2`; the first pilot's files are untouched. Cycle 1 is sampled afresh on
  the new engine with the same sampling seeds, not continued from the first pilot's attempts;
- the seed adapter (unchanged, from Stage 1) is scored again on `dev` and on the adversarial set with the same
  engine, so that the three adapters are measured alike (`r0_seed_p2`);
- each cycle's adapter is also scored on the adversarial set.

**Budget**: a hard cap of 5.00 USD for everything (`configs/study02/budget_e2_stage2r.json`), 0.50 held back;
two jobs whose worst cases sum to 3.57 USD. Expected about 1.7 USD.

## 7. Known before launch, from data already collected

- Under protocol 2 the teacher qualification of Stage 1 reads differently. `Qwen3-32B`: 135 of 200 tasks
  accepted (67.5%), 30% at four hops, where the thresholds are 60% and 40%; 20 of its 155 demonstrations
  contain a shortcut. `Qwen3-14B`: 55.5% and 18%. **No teacher qualifies under protocol 2 on the Stage 1
  data.** Nothing is done about this here; it bears on whether H2(b) can be tested.
- The acceptance rule cannot stop a learner from trying the shortcut during rollouts. It stops such attempts
  from being rewarded. Whether the behaviour then fades, persists on `dev`, or changes form is what the
  repeated pilot and its audit measure.
- The detector describes the shortcut in the forms seen so far. The audit's wider flag and its blind reading
  exist because a third form is possible.

## 8. Outcome of the repeated pilot (added 2026-11-11, after it ran; nothing above was edited)

Report: [docs/study02/REPORT_E2_STAGE2R.md](../../docs/study02/REPORT_E2_STAGE2R.md). All four gates met:
yield 33.2% and 42.0%; audit 0 false accepts in 200 (19 flagged and 20 unflagged episodes read); cost 0.79
and 0.81 of the model; `dev` 25.8% → 30.7% → 44.2% by the original validator. Buffers: no shortcut, no wrong
part. Cycle-2 adapter: `dev` 44.2% (original validator), 43.5% (each fact for its own part), 40.8% (and no
shortcut); adversarial set 18.3%, against 23.7% on the same 350 tasks' original worlds. 1.77 USD of the 5 USD
cap. Stage 3 is not approved.

## 9. Not changed, not done

Hypotheses and thresholds; arms; seed set and stream (frozen hashes); training settings; sampling; the
curriculum; the teacher thresholds; `data/v2`. No sealed split is scored. Stage 3 is not run. The 1.5B scoring
Experiment 1 left out stays on hold.
