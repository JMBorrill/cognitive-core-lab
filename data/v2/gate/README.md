# Quality gate for `data/v2`

The gate of the proposal (§5.3) has three steps. Steps 1 and 2 run inside `scripts/build_v2.py` on every
build and their counts are in `data/v2/manifest.json` (`quality_gate`): the oracle solves every generated
task, and two degenerate policies (answer at once; call every listed tool) fail every task that needs a tool.

Step 3 asks a strong open-weight reference model to attempt the test families before any student is scored on
them. It ran as follows.

| run | what | result |
|---|---|---|
| `gate` (stopped) | `Qwen/Qwen2.5-7B-Instruct`, few-shot prompt, first task of every `test_comp` family, deepest first | stopped after 200 four-hop tasks: 3% solved |
| `gate_shallow` | the same model on families of at most two hops, second task where the first failed | see `gate_shallow_report.json` |

**Why the first run was stopped.** The transcripts show well-formed tasks and a reference model that fails
through its own errors: it invents identifiers (`SUP-12345`, `V-1234`), issues calls in parallel whose
arguments do not exist yet, and gives up after the resulting errors. A model that solves 3% of four-hop tasks
cannot separate a defective task from a hard one, so "families it fails twice" would have listed nearly all of
them. The run was replaced by one on shallow families, where the model is competent enough for a failure to
mean something, and where every phrase template of the schema still occurs.

**Reading.** Every episode in which the reference model made all the required calls and still failed was read
(the class in which an ambiguous request or a wrong gold answer would show up), together with a sample of the
others. Each one is an error of the model: the wrong record picked from a list, a tool that does not exist, a
stale answer, or an early stop. No request was ambiguous about which record or attribute it meant and no gold
graph penalised a defensible solution.

**Decision.** No family was removed (`decisions.json` is empty). This is weaker evidence than the proposal
intended: the reference model could not vouch for families of three hops or more, which are built from the
same phrases and the same world-building code but were not independently solved by any model other than the
oracle. The students' own results provide a later, partial check: a family that is never solved by any
replicate at any level is listed in the Experiment 1 report.
