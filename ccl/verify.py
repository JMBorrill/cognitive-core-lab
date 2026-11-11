"""Protocol version 2 of Study 02, Experiment 2: the two checks the first pilot showed to be missing.

Both read only what an episode produced (its trace and final answer) and what the task states (its request
and reference answer). Neither reads the gold graph, so they can be part of the acceptance rule, whose
principle is that it uses the outcome and runtime observations only. No tensor library is imported here.

1. `parts_ok`: a multi-part answer must give each requested fact for its own part. The Study 01 validator
   checks that every fact string occurs somewhere in the answer, so "merit ... fail" passed a question whose
   two answers were both "merit".

2. `shortcut`: an episode must not have learned which candidate of a list is on the gold path from the world
   being incomplete. In the generated worlds a record that is only a candidate refers to records that do not
   exist and has nothing listed under it, so asking for one candidate's continuation after another reveals
   the right one without applying the selection rule the request states.

The original validator (`ccl.evaluate.validate`) is unchanged, and its `success` stays in every metrics row
for comparability with Experiments 0 and 1 and with the first pilot.
"""

from __future__ import annotations

import re

from .evaluate import _NUM

PROTOCOL = 2  # acceptance rule in force; 1 is the rule of the proposal (`ccl.s2.rejection`)

_NUMERIC = re.compile(r"-?\d+(?:\.\d+)?")


# ------------------------------------------------------------------ multi-part answers

def answer_parts(reference_answer: str, facts: list[str]) -> list[tuple[str, str]] | None:
    """(subject, fact) for each requested fact, read from a reference answer of the generated form
    "The <subject> is <fact>." (optionally after "For <...>: "). None if the answer is not of that form."""
    parts, at = [], 0
    for f in facts:
        end = reference_answer.find(f" is {f}.", at)
        if end < 0:
            return None
        start = reference_answer.rfind("The ", at, end)
        if start < 0:
            return None
        parts.append((reference_answer[start + 4:end], f))
        at = end + len(f" is {f}.")
    return parts


def _find_fact(fact: str, text: str, start: int, stop: int | None = None) -> int:
    """Position just after the first occurrence of `fact` in text[start:stop], or -1. Numbers match by
    value (as `ccl.evaluate.fact_present`: "87.8" is "87.80"), other facts by case-insensitive text."""
    stop = len(text) if stop is None else stop
    if _NUMERIC.fullmatch(fact):
        for m in _NUM.finditer(text, start, stop):
            try:
                if abs(float(m.group().replace(",", "")) - float(fact)) < 0.01:
                    return m.end()
            except ValueError:
                pass
        return -1
    at = text.lower().find(fact.lower(), start, stop)
    return -1 if at < 0 else at + len(fact)


_TOKEN = re.compile(r"[A-Z]{1,3}-\d{2,6}|[A-Z][\w'-]+")  # identifiers and capitalised names
_WORD = re.compile(r"[a-z]{4,}")


def _anchors(parts: list[tuple[str, str]], final: str) -> list[tuple[int, int]] | None:
    """Where the answer names each part, when it does not repeat the reference's wording: the first
    occurrence of something only that part's subject contains. Identifiers and names are used if they tell
    every part apart (two records asked the same question); otherwise words of four letters or more that
    only one subject has (two questions about one record). None if some part is not named."""
    for pattern, text in ((_TOKEN, final), (_WORD, final.lower())):
        own = []
        for i, (subject, _) in enumerate(parts):
            src = subject if pattern is _TOKEN else subject.lower()
            mine = set(pattern.findall(src))
            other = {t for j, (o, _) in enumerate(parts) if j != i for t in pattern.findall(o if pattern is _TOKEN else o.lower())}
            own.append(mine - other)
        if not all(own):
            continue
        spans = []
        for tokens in own:
            found = [m.span() for t in tokens for m in [re.search(r"(?<![\w-])" + re.escape(t) + r"(?![\w-])", text)] if m]
            spans.append(min(found) if found else None)
        if all(spans) and len(set(spans)) == len(spans):
            return spans
    return None


def _attributed(parts, spans, final: str, after: bool) -> bool:
    """Every fact stands in the stretch that belongs to its part: from the part's name to the next name
    (`after`), or from the previous name to the part's name."""
    order = sorted(spans)
    for (_, fact), (s, e) in zip(parts, spans):
        k = order.index((s, e))
        lo, hi = (e, order[k + 1][0] if k + 1 < len(order) else len(final)) if after else \
            (order[k - 1][1] if k else 0, s)
        if _find_fact(fact, final, lo, hi) < 0:
            return False
    return True


def parts_ok(task, final: str | None) -> bool:
    """Does the answer give each requested fact for its own part?

    One fact or none: nothing to attribute (the original fact check decides). Several facts, in this order:
      1. the answer repeats the subject of every sentence of the reference answer (what a model trained on
         this format writes): the fact of a part must stand between its subject and the next subject;
      2. the answer names every part in other words (`_anchors`): the facts must all follow the names of
         their parts, or all precede them, each inside its own part's stretch of the answer;
      3. the answer does not name the parts: every fact must have an occurrence of its own, so two equal
         facts need two occurrences. Nothing more can be attributed.
    A reference answer that is not of the generated form (Study 01's hand-written tasks) falls under 3."""
    facts = list(task.answer_facts)
    if len(facts) < 2:
        return True
    if final is None:
        return False
    parts = answer_parts(task.reference_answer, facts)
    if parts:
        low = final.lower()
        exact = [m.span() if m else None for m in
                 (re.search(r"(?<![\w-])" + re.escape(subject.lower()) + r"(?![\w-])", low) for subject, _ in parts)]
        if all(exact) and len(set(exact)) == len(exact):
            return _attributed(parts, exact, final, after=True)
        spans = _anchors(parts, final)
        if spans:
            return _attributed(parts, spans, final, True) or _attributed(parts, spans, final, False)
    used: list[tuple[int, int]] = []  # 3: a distinct occurrence for every fact, in any order
    for f in facts:
        at = 0
        while True:
            end = _find_fact(f, final, at)
            if end < 0:
                return False
            span = (end - 1, end)  # the occurrence is identified by where it ends
            if span not in used:
                used.append(span)
                break
            at = end
    return True


# ------------------------------------------------------------------ the shortcut

def _rows(call: dict) -> list | None:
    """The records a successful list or find call returned, or None."""
    if call["outcome"] != "ok":
        return None
    return next((v for v in (call["result"] or {}).values() if isinstance(v, list)), None)


def _values(call: dict) -> set[str]:
    return {str(v).strip().upper() for v in (call["args"] or {}).values() if isinstance(v, (str, int))}


def shortcut(trace: dict) -> bool:
    """True if the trace shows a list's candidates being told apart by whether their continuation exists.

    Definitions, all over the calls of one episode (calls refused for a limit are ignored):
      * a *column* is one field of the records of one successful list (or find) call that returned at least
        two records, for example the `instrument_id` values of the assays a doctor ordered;
      * a *probe* is a call made after that list whose argument is a value of the column and which comes back
        with "not found" or with an empty list: the record that candidate refers to does not exist, or
        nothing is listed under it;
      * a *hit* is a call made after that list whose argument is a different value of the same column and
        which succeeds with a record or a non-empty list.
    A trace with a probe and a hit on the same column is a shortcut, whatever the tools and whatever the
    order of the two: the outcome of the probe told the episode which candidate to follow, or confirmed it.

    Not a shortcut: retrying the same identifier; an identifier that is not a value of a listed column (a
    mistyped or invented one); "not found" with no hit afterwards or before (a task whose path ends there).
    The function is deterministic and reads nothing but the trace."""
    calls = [c for c in trace["calls"] if c["outcome"] != "limit"]
    for k, lst in enumerate(calls):
        rows = _rows(lst)
        if not rows or len(rows) < 2 or not all(isinstance(r, dict) for r in rows):
            continue
        later = calls[k + 1:]
        probes, hits = set(), set()
        for c in later:
            if c["outcome"] == "ok":
                r = _rows(c)
                (probes if r == [] else hits).update(_values(c))
            elif (c["result"] or {}).get("error") == "not_found":
                probes |= _values(c)
        if not probes or not hits:
            continue
        for f in {f for r in rows for f in r}:
            col = {str(r.get(f)).strip().upper() for r in rows}
            p, h = col & probes, col & hits
            if any(x != y for x in p for y in h):  # a probe and a hit on different values (the hit may itself
                return True                         # have failed once, e.g. with another tool, before it succeeded)
    return False


def enumerates(trace: dict) -> bool:
    """True if later calls use two or more different values of one column (see `shortcut`), whatever came
    back. Wider than `shortcut` and not part of any rule: the audit lists such episodes for reading, so that a
    way of telling candidates apart that `shortcut` does not describe would still be looked at."""
    calls = [c for c in trace["calls"] if c["outcome"] != "limit"]
    for k, lst in enumerate(calls):
        rows = _rows(lst)
        if not rows or len(rows) < 2 or not all(isinstance(r, dict) for r in rows):
            continue
        used = set().union(*[_values(c) for c in calls[k + 1:]]) if calls[k + 1:] else set()
        for f in {f for r in rows for f in r}:
            if len({str(r.get(f)).strip().upper() for r in rows} & used) >= 2:
                return True
    return False
