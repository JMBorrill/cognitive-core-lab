"""An adversarial evaluation for Study 02, Experiment 2: `dev` tasks on worlds whose candidate records are complete.

The generated worlds build only the records on a task's gold path in full. A record that is merely a
candidate of a list refers to records that do not exist and has nothing listed under it, so an agent can find
the right candidate by asking for each one's continuation instead of applying the selection rule the request
states (`ccl.verify.shortcut`). `twin` rebuilds a task on a world where that no longer works: for the tools of
the task's gold path, every reference leads to a record and every record has records listed under it, to the
depth of the task. Whatever the gold path asks of the chosen record can be asked of every other candidate with
the same kind of result. For the other tools of the catalogue nothing changes, and there the chosen record
and the other candidates were already alike (neither has anything behind it).

A twin has the same request, catalogue, gold graph, faults and answer as its original, and every original
record unchanged, so an agent that applies the selection rule sees exactly the same episode on both. Any
difference in success between a task and its twin comes from what the agent does off the gold path.

`adversarial_dev` is the set used in Experiment 2: the twin of every `dev` task in which some call depends on
a list selection (350 of 600). It is an evaluation set only, built from `dev` and from nothing else, and
no training code reads it. Generation is deterministic; the set's SHA-256 is recorded in
`configs/study02/e2/protocol2.json` and checked by the tests.
"""

from __future__ import annotations

import copy
import random
from dataclasses import replace
from datetime import date

from . import compose as C
from . import s2
from .tasks import Task, task_hash
from .world2 import ENTITIES, TOOLS2

SALT = "ccl-study02-e2-adversarial-1"
CHILDREN = 2  # records created under a record that has none
MAX_RECORDS = 3000  # a world larger than this means the completion ran away; it is an error, not a truncation


def depends_on_selection(task: Task) -> bool:
    """Does some gold call take its argument from a record chosen out of a list? Only then can a candidate's
    continuation be probed."""
    lists = {g.id for g in task.gold if TOOLS2[g.tool].kind == "list"}
    return any(set(g.deps) & lists for g in task.gold)


def complete_world(task: Task, rounds: int | None = None) -> tuple[dict, dict]:
    """A copy of the task's world, completed for the tools of its gold path, and what was added.

    One round: (a) for every record and every reference field whose target type some gold-path tool takes as
    an identifier, create the referenced record if it does not exist; (b) for every list tool of the gold path
    and every record of its parent type with nothing listed under it, create two records under it. New
    records get fresh identifiers, unused names and random attribute values, and their own references point
    at identifiers that the next round resolves. `rounds` defaults to the task's depth: the first selection
    is made after at least one call, so a walk from any candidate along the rest of the gold path's tools
    stays inside the completed part.

    Records under a gold-path parent are never added to, so every list on the gold path returns what it
    returned before and every selection is unchanged."""
    world = copy.deepcopy(task.world)
    rng = random.Random(f"{SALT}|{task.task_id}")
    b = C.Builder(rng, "eval")
    b.w, b.today = world, date.fromisoformat(task.today)
    tables = [e for e in ENTITIES if world.get(e)]
    b.used = {rid for e in tables for rid in world[e]} | {r["name"] for e in tables for r in world[e].values() if "name" in r} \
        | {r[ref.field] for e in tables for r in world[e].values() for ref in ENTITIES[e].refs}
    keys = [TOOLS2[k] for k in sorted({k for g in task.gold for k in (g.tool, g.fallback) if k})]
    # entity types whose identifier some gold-path tool accepts
    addressable = {t.parent if t.kind == "list" else t.entity for t in keys} | \
        {kind[4:] for t in keys if t.action for _, kind in t.action.params if kind.startswith("ref:")}
    lists = sorted((t for t in keys if t.kind == "list"), key=lambda t: t.key)
    added = {"records": 0, "references_resolved": 0, "lists_filled": 0}
    serial = [0]

    def make(ent, rid=None, **fixed) -> dict:
        rec = {ent.id_field: rid or b.uid(ent)}
        if ent.named:
            try:
                rec["name"] = b._name(ent.named, False)
            except RuntimeError:  # the name pool is used up in a large world
                serial[0] += 1
                rec["name"] = f"{rng.choice(C.NAMES[ent.named][:18]) if ent.named != 'person' else 'Alex Morgan'} {1000 + serial[0]}"
                b.used.add(rec["name"])
        for a in ent.attrs:
            rec[a.name] = b._value(a, rec)
        for r in ent.refs:
            rec[r.field] = b.uid(ENTITIES[r.target])
        rec.update(fixed)
        world[ent.name][rec[ent.id_field]] = rec
        added["records"] += 1
        return rec

    for _ in range(task.depth if rounds is None else rounds):
        before = {name: list(world[name].values()) for name in sorted(ENTITIES)}  # a round works on what exists at its start
        for name, recs in before.items():  # (a) references
            for rec in recs:
                for r in ENTITIES[name].refs:
                    if r.target in addressable and rec[r.field] not in world[r.target]:
                        make(ENTITIES[r.target], rid=rec[r.field])
                        added["references_resolved"] += 1
        for t in lists:  # (b) lists
            have = {r[t.ref_field] for r in world[t.entity].values()}
            for parent in before[t.parent]:
                pid = parent[ENTITIES[t.parent].id_field]
                if pid not in have:
                    for _ in range(CHILDREN):
                        make(ENTITIES[t.entity], **{t.ref_field: pid})
                    added["lists_filled"] += 1
        assert added["records"] <= MAX_RECORDS, f"{task.task_id}: {added['records']} records added"
    return world, added


def twin(task: Task) -> Task:
    """The task on its completed world, in split `dev_adv`. `twin_of=<task id>` in its tags names the original."""
    world, _ = complete_world(task)
    t = replace(task, world=world, split="dev_adv", tags=list(task.tags) + [f"twin_of={task.task_id}"], task_id="")
    t.task_id = f"dev_adv-{task.template}-{task_hash(t)[:6]}-{task.task_id.split('-')[-1]}"
    return t


def adversarial_dev() -> list[Task]:
    return [twin(t) for t in s2.load_split("dev") if depends_on_selection(t) and t.expected_status in ("done", "blocked")]
