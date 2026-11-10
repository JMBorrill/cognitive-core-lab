"""Study 02 task families: typed paths through the schema, and tasks generated from them.

Vocabulary
  plan     a typed path through the schema (`ccl.world2`): a start entity given by ID or by
           name, steps that follow a reference or list the records linked to the current
           entity and pick one, and an ending (report attributes, fan out over two
           references, or perform an action), plus one optional behaviour modifier.
  family   all tasks that share one gold dependency graph, ignoring concrete values: the
           `graph_signature` of Study 01. It is the unit that generalisation is claimed over.
           Plans whose graphs coincide (for example every request about a record that does
           not exist, whatever would have followed) belong to the same family.
  edge     one tool's output feeding another tool's argument: a (source tool, target tool) pair.

What is, and is not, a different family. Two families always differ in the tools called,
in how those calls depend on each other, or in the required behaviour (retry, fallback,
stop on not-found, stop on denial, refuse). Phrasing, concrete values, the attribute asked
for, the rule used to pick from a list, distractor tools and the order of independent calls
never create a new family. Each base graph has at most one retry variant and one fallback
variant (the faulty call is fixed by a hash), so fault position does not multiply families,
and the same question asked about two records at once is a family only for paths of at most
two calls.

In `Task` objects the family identifier is stored in `template` (the unit of generalisation,
as the template was in Study 01) and the shape label in `family`.

Hold-outs are assigned by hash, never by hand: two domains and a set of tools and edges are
reserved for the sealed test splits, and a separate domain, tools and edges for `dev`.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from datetime import date, timedelta

from .tasks import GoldCall, Task, task_hash
from .tools import FaultPlan
from .world2 import BACKUP_OF, DOMAINS, ENTITIES, TOOLS2, Entity, empty_world

MAX_HOPS = 6
PRACTICE_HOPS = 4
TWIN_HOPS = 2  # "the same question about two records" only for short paths, so twins do not double the space
SALT = "ccl-study02-v2"
TEST_DOMAINS, DEV_DOMAINS = 2, 1
TEST_TOOL_QUOTA = {"get": 2, "find": 2, "list": 2, "action": 2}
DEV_TOOL_QUOTA = {"get": 1, "find": 1, "list": 1, "action": 1}
TEST_EDGE_FRAC, DEV_EDGE_FRAC = 0.20, 0.08
SPLITS = ("practice", "dev_comp", "dev_tool", "dev_domain", "test_comp", "test_depth", "test_tool", "test_domain")


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16)


# ------------------------------------------------------------------ value pools

FIRST = ["Agnes", "Bashir", "Celine", "Dawit", "Edda", "Fabio", "Greta", "Hamza", "Ines", "Jarek", "Kalinda",
         "Lorcan", "Mirela", "Nasser", "Odile", "Paavo", "Rahel", "Soren", "Tamsin", "Ulrik", "Valeria",
         "Wendell", "Xiomara", "Yannick", "Zdenka", "Anselm", "Briony", "Cosmin", "Delphine", "Emeka",
         "Fenna", "Goran", "Hedda", "Ignacio", "Jolanta", "Kwame", "Liesel", "Matteo", "Nadia", "Orla"]
LAST = ["Achterberg", "Bellamy", "Castellano", "Dragomir", "Ekström", "Fontaine", "Galanis", "Hollis",
        "Iversen", "Jankowski", "Kowalczyk", "Lindqvist", "Marchetti", "Nwosu", "Oyelaran", "Pellegrini",
        "Quintero", "Rasmussen", "Stavros", "Thorvald", "Uzun", "Vasquez", "Whitlock", "Xenakis", "Yildirim",
        "Zielinski", "Aldridge", "Brennan", "Cardoso", "Demir"]
CITIES = ["Aarhus", "Bergen", "Cork", "Dresden", "Eindhoven", "Faro", "Gdansk", "Graz", "Innsbruck", "Kaunas",
          "Leuven", "Malaga", "Nantes", "Odense", "Patras", "Rennes", "Salzburg", "Tartu", "Uppsala", "Valencia",
          "Wroclaw", "Zagreb", "Bilbao", "Tampere"]
NAMES = {
    "book": ["Silent Rivers", "The Glass Orchard", "Winter Atlas", "A Field Guide to Echoes", "Salt and Iron",
             "The Lantern Keeper", "Northbound", "Paper Harbours", "The Quiet Engine", "Maps of Rain",
             "Small Hours", "The Copper Road", "Hollow Stars", "An Almanac of Tides", "Borrowed Light",
             "The Seventh Meadow", "Ashes of Summer", "Stone Letters", "The Long Thaw", "Ink and Ember",
             "Cartographers of Dusk", "The Wren's Ledger", "Low Tide Songs", "A Measure of Frost"],
    "course": ["Applied Statistics", "Modern Logic", "Marine Ecology", "Structural Design", "Public Finance",
               "Clinical Ethics", "Signal Processing", "Urban History", "Organic Chemistry", "Contract Law",
               "Game Theory", "Machine Vision", "Soil Science", "Typography", "Epidemiology", "Thermodynamics",
               "Comparative Politics", "Acoustics", "Number Theory", "Hydrology", "Phonetics", "Plant Genetics",
               "Labour Economics", "Robotics"],
    "show": ["Midnight Brass", "The Paper Moon Revue", "Echoes in Blue", "Northern Lights Gala", "Strings Unbound",
             "The Velvet Hour", "Comedy at the Quay", "Harvest Chorus", "Neon Folk Night", "The Clockwork Ballet",
             "Open Deck Sessions", "Saltwater Jazz", "A Night of Tango", "The Tin Drum Parade", "Baroque Unplugged",
             "Riverside Proms", "The Lantern Parade", "Voices of Spring", "Static and Silk", "The Winter Cabaret",
             "Brass on the Green", "Quiet Riot Poetry", "The Amber Recital", "Drums of the Delta"],
    "company": ["Nordkap Metals", "Brightwater Supply", "Calder and Finch", "Meridian Tooling", "Pinecrest Paper",
                "Hollin Logistics", "Verano Foods", "Stannard Optics", "Kestrel Fabrics", "Orrin Chemicals",
                "Tallis Printing", "Wexford Timber", "Bluegate Plastics", "Farrow Electric", "Marlow Ceramics",
                "Sable Instruments", "Greyfield Stone", "Lumen Packaging", "Arden Hydraulics", "Coburn Glass",
                "Thistle Works", "Redfern Cables", "Osprey Trading", "Dunmore Mills"],
    "venue": ["Harbour Hall", "The Old Tannery", "Kingsmead Arena", "Linden Pavilion", "The Copper Room",
              "Riverside Theatre", "Foundry Stage", "Elm Court", "The Gasworks", "Beacon Hall", "Marble Arcade",
              "The Boathouse", "Orchard Dome", "Quarry Theatre", "The Signal Box", "Saltmarsh Hall",
              "Granary Stage", "The Glasshouse", "Tower Rooms", "Millpond Arena", "The Observatory",
              "Anchor Yard", "Chapel Works", "The Rotunda"],
    "instrument": ["Spectra 200", "Helix Nine", "Orbit Mark II", "Vantage S4", "Corelab 7", "Prism Delta",
                   "Nova Trace", "Quanta 12", "Apex Lumen", "Sigma Reader", "Titan Fuge", "Vector One",
                   "Polar Scan", "Argon Six", "Kelvin Pro", "Lyra Micro", "Zenith 40", "Strata X",
                   "Boreal 3", "Cobalt Ray", "Pulsar Mini", "Tessera 8", "Ion Path", "Halo Scope"],
    "asset": ["Boiler North", "Lift Car Two", "Main Switchboard", "Roof Chiller", "Pump Station East",
              "Fire Panel West", "Generator Bay", "Cooling Tower", "Water Heater South", "Goods Lift",
              "Air Handler Four", "Sprinkler Riser", "Heat Exchanger", "Lighting Board", "Sump Pump",
              "Ventilation Fan Six", "Backup Battery", "Steam Valve", "Loading Dock Door", "Solar Inverter",
              "Chilled Water Loop", "Stair Pressuriser", "Gas Meter Room", "Transformer Yard"],
}
QUOTED = {"book", "course", "show"}
SMALLTALK = (["Thanks, that's everything for now.", "Hello there, just checking in.", "Great, thank you for your help.",
              "No further questions today.", "Good morning! Nothing needed yet.", "That is all, have a nice day."],
             ["Cheers, I'm all set.", "Hi, only saying hello.", "Nothing else from me, thanks."])
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
ANSWER_Q = ("What is the {a} of {x}?", "Tell me the {a} of {x}.", "I need the {a} of {x}.")
PAIR_Q = ("Consider {z}. What are the {a} of {x} and the {b} of {y}?",
          "Look at {z}. Tell me the {a} of {x} and the {b} of {y}.",
          "Regarding {z}: I need the {a} of {x} and the {b} of {y}.")
TWIN_Q = ("What is the {a} of {x} and of {y}?", "Tell me the {a} of {x} and of {y}.",
          "I need the {a} of {x} and of {y}.")
# Every phrase ends in the noun phrase it builds on, so nested requests stay right-branching and a
# selection rule always sits next to the noun it applies to.
RANK_P = ("the {c} with the {w} {a} among those {p} {x}", "the {c} that has the {w} {a} out of all {cs} {p} {x}",
          "whichever {c} has the {w} {a} among the {cs} {p} {x}")
MATCH_P = ("the {c} whose {a} is {v} among those {p} {x}", "the {c} with {a} {v} out of all {cs} {p} {x}",
           "whichever {c} has {a} {v} among the {cs} {p} {x}")


# ------------------------------------------------------------------ plans -> abstract graphs

def _ref(ent: Entity, field: str):
    return next(r for r in ent.refs if r.field == field)


def _list_tools(parent: str) -> list[str]:
    return sorted(k for k, t in TOOLS2.items() if t.kind == "list" and t.parent == parent)


def _walk(plan: dict, nodes: list, b=None, attr: str | None = None) -> dict:
    """Append the gold calls of one pass over `plan` to `nodes`.

    Without a builder `b` only the graph is produced (tools and dependencies). With one, the
    records are created in the builder's world, arguments are concrete and the request
    phrase is composed."""
    def add(tool, deps, args=None):
        nodes.append({"id": f"g{len(nodes) + 1}", "tool": tool, "deps": [d for d in deps if d], "args": args or {}})
        return nodes[-1]["id"]

    ent = ENTITIES[plan["start"]]
    by_name = plan["mode"] == "name"
    rec = b.record(ent, target=True) if b else None
    out = {"start_entity": ent.name, "start_rec": rec, "root": len(nodes)}
    np_ = b.mention(ent, rec, by_name) if b else ""
    out["start_np"] = np_
    src, have = None, False
    if by_name:
        src, have = add(f"find_{ent.name}", [], {"name": rec["name"]} if b else None), True
    for kind, arg in plan["steps"]:
        if kind == "ref":
            if not have:
                src = add(f"get_{ent.name}", [src], {ent.id_field: rec[ent.id_field]} if b else None)
            r = _ref(ent, arg)
            tgt = ENTITIES[r.target]
            if b:
                nxt = b.record(tgt)
                rec[r.field] = nxt[tgt.id_field]
                rec, np_ = nxt, r.phrases[b.variant()].format(x=np_)
            ent, have = tgt, False
        else:
            t = TOOLS2[arg]
            child = ENTITIES[t.entity]
            src = add(arg, [src], {ent.id_field: rec[ent.id_field]} if b else None)
            if b:
                rec, np_ = b.children(ent, rec, child, t.ref_field, np_)
            ent, have = child, True
    end = plan["end"]
    if end[0] in ("answer", "fanout") and not have:
        src = add(f"get_{ent.name}", [src], {ent.id_field: rec[ent.id_field]} if b else None)
    if end[0] == "answer":
        if b:
            a = ent.attr(attr) if attr else b.rng.choice(ent.attrs)  # a twin asks the same attribute twice
            out["asks"], out["attr"] = [(a.phrase, np_, b.fact(rec[a.name]))], a.name
    elif end[0] == "fanout":
        out["asks"], out["about"] = [], np_
        for f in end[1:]:
            r = _ref(ent, f)
            tgt = ENTITIES[r.target]
            if b:
                nxt = b.record(tgt)
                rec[r.field] = nxt[tgt.id_field]
                a = b.rng.choice(tgt.attrs)
                out["asks"].append((a.phrase, r.phrases[b.variant()].format(x=f"that {ent.noun}"),
                                    b.fact(nxt[a.name])))
            add(f"get_{tgt.name}", [src], {tgt.id_field: nxt[tgt.id_field]} if b else None)
    else:  # act
        act = next(a for a in ent.actions if a.name == end[1])
        deps, args, words = [src], ({ent.id_field: rec[ent.id_field]} if b else {}), {}
        for pname, kind in act.params:
            if kind.startswith("ref:"):
                other = ENTITIES[kind[4:]]
                orec = b.record(other, target=True) if b else None
                if end[2] == "name":
                    deps.append(add(f"find_{other.name}", [], {"name": orec["name"]} if b else None))
                if b:
                    args[pname] = orec[other.id_field]
                    words[pname] = b.mention(other, orec, end[2] == "name", bare=True)
            elif b and kind == "date":
                words[pname], args[pname] = b.date_phrase()
            elif b:
                args[pname] = words[pname] = b.rng.choice([v for v in ent.attr(kind[5:]).kind[1]
                                                           if v != rec.get(kind[5:])])
        add(act.name, deps, args)
        if b:
            out["request"] = act.phrases[b.variant()].format(x=np_, **words)
            out["answer"] = f"Done: {ent.noun} {rec[ent.id_field]} is now {act.done}."
        out["action"] = act
    return out


def _action_of(plan: dict):
    if plan["end"][0] != "act":
        return None
    ent = ENTITIES[plan["start"]]
    for kind, arg in plan["steps"]:
        ent = ENTITIES[_ref(ent, arg).target] if kind == "ref" else ENTITIES[TOOLS2[arg].entity]
    return next(a for a in ent.actions if a.name == plan["end"][1])


def _fault_index(plan: dict, nodes: list, kind: str) -> int | None:
    """The call that fails in the retry / fallback variant of a base plan: one fixed position per base graph."""
    ok = [i for i, n in enumerate(nodes) if (TOOLS2[n["tool"]].kind != "action" if kind == "retry"
                                             else n["tool"] in BACKUP_OF)]
    return ok[_h(SALT, kind, _signature(nodes, "user", "done", FaultPlan())) % len(ok)] if ok else None


def _signature(nodes: list, role: str, status: str, faults: FaultPlan) -> str:
    """Value-free description of a gold graph: role, expected status, calls with their dependencies, faults.

    This is `ccl.tasks.graph_signature` with one refinement: a tool that fails once (retry) and a
    tool that is down (fallback) are told apart, because they require different behaviour. Study 01's
    signature merges them."""
    idx = {n["id"]: n for n in nodes}
    parts = sorted(f"{n['tool']}<-[{','.join(sorted(idx[d]['tool'] for d in n['deps']))}]"
                   f"{'?' if n.get('optional') else ''}{'' if n.get('execute', True) else '!x'}" for n in nodes)
    fault = [f"down:{t}" for t in sorted(faults.down)] + \
            [f"retry:{t}" for t in sorted(k for k, v in faults.transient.items() if any(v))]
    return f"{role}|{status}|{';'.join(parts)}|faults={','.join(fault)}"


def family_signature(task: Task) -> str:
    """The signature of a generated task; equal to its family's signature (checked by the tests)."""
    nodes = [{"id": g.id, "tool": g.tool, "deps": g.deps, "optional": g.optional, "execute": g.execute}
             for g in task.gold]
    return _signature(nodes, task.role, task.expected_status, task.faults)


def family_id(signature: str) -> str:
    return "f" + hashlib.sha256(signature.encode()).hexdigest()[:10]


def compile_plan(plan: dict, b=None) -> dict:
    """Plan -> gold graph with its role, expected status, faults and approvals (concrete if `b` is given)."""
    mod = plan["mod"][0]
    if mod == "notool":
        return {"nodes": [], "role": "user", "status": "done", "faults": FaultPlan(), "approvals": {}, "info": [{}],
                "all_tools": []}
    nodes: list = []
    info = [_walk(plan, nodes, b)]
    if plan.get("twin"):
        info.append(_walk(plan, nodes, b, info[0].get("attr")))
    all_tools = sorted({n["tool"] for n in nodes})  # before a modifier trims the graph
    act = info[0].get("action")
    role = act.role if act and act.role and mod != "refused" else "user"
    status, faults, approvals = "done", FaultPlan(), {}
    if act and act.approval:
        approvals[act.name] = mod != "denied"
    if mod == "retry":
        faults = FaultPlan(transient={nodes[plan["mod"][1]]["tool"]: ["timeout"]})
    elif mod == "fallback":
        tool = nodes[plan["mod"][1]]["tool"]
        for n in nodes:
            if n["tool"] == tool:
                n["fallback"] = BACKUP_OF[tool]
        faults = FaultPlan(down=(tool,))
    elif mod == "notfound":
        nodes = nodes[:1]
        if TOOLS2[nodes[0]["tool"]].kind == "action":
            nodes[0]["execute"] = False
            approvals = {nodes[0]["tool"]: True} if act.approval else {}
        else:
            approvals = {}
        status = "failed"
        if b:
            del b.w[info[0]["start_entity"]][info[0]["start_rec"][ENTITIES[info[0]["start_entity"]].id_field]]
    elif mod == "denied":
        nodes[-1]["execute"] = False
        status = "blocked"
    elif mod == "refused":
        nodes = [{**nodes[-1], "id": "g1", "deps": [], "optional": True, "execute": False}]
        status, approvals = "refused", {}
    return {"nodes": nodes, "role": role, "status": status, "faults": faults, "approvals": approvals, "info": info,
            "all_tools": all_tools}


def _hops(nodes: list) -> int:
    idx = {n["id"]: n for n in nodes}
    memo: dict = {}

    def depth(n):
        if n["id"] not in memo:
            memo[n["id"]] = 1 + max((depth(idx[d]) for d in n["deps"]), default=0)
        return memo[n["id"]]
    return max((depth(n) for n in nodes), default=0)


def _shape(plan: dict, hops: int) -> str:
    if plan["mod"][0] == "notool":
        return "notool"
    f = ["single" if hops == 1 else "chain"]
    if any(k == "list" for k, _ in plan["steps"]):
        f.append("list")
    if plan["end"][0] == "fanout":
        f.append("fanout")
    if plan["end"][0] == "act":
        if len(plan["end"]) > 2 and plan["end"][2] == "name":
            f.append("fanin")
        f.append("act")
    if plan.get("twin"):
        f.append("twin")
    return "+".join(f)


def enumerate_plans(max_hops: int = MAX_HOPS):
    """Every base plan (no modifier) of at most `max_hops` sequential calls, with its variants.

    Yields (plan, base_plan). A path never returns along the reference it has just listed
    (which would only restate the starting record) and visits an entity type at most twice."""
    def endings(ent: Entity, banned: str | None):
        yield ["answer"]
        refs = [r.field for r in ent.refs if r.field != banned]
        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                yield ["fanout", refs[i], refs[j]]
        for a in ent.actions:
            other = next((k[4:] for _, k in a.params if k.startswith("ref:")), None)
            if other is None:
                yield ["act", a.name, None]
            else:
                yield ["act", a.name, "id"]
                if ENTITIES[other].named:
                    yield ["act", a.name, "name"]

    def paths(ent: Entity, steps: list, seen: dict, banned: str | None):
        yield ent, list(steps), banned
        if len(steps) >= max_hops:
            return
        for r in ent.refs:
            if r.field == banned or seen.get(r.target, 0) >= 2:
                continue
            yield from paths(ENTITIES[r.target], steps + [["ref", r.field]], {**seen, r.target: seen.get(r.target, 0) + 1}, None)
        for key in _list_tools(ent.name):
            t = TOOLS2[key]
            if seen.get(t.entity, 0) >= 2:
                continue
            yield from paths(ENTITIES[t.entity], steps + [["list", key]], {**seen, t.entity: seen.get(t.entity, 0) + 1},
                             t.ref_field)

    for start in ENTITIES.values():
        for mode in ["id"] + (["name"] if start.named else []):
            for ent, steps, banned in paths(start, [], {start.name: 1}, None):
                for end in endings(ent, banned):
                    base = {"start": start.name, "mode": mode, "steps": steps, "end": end, "mod": ["none"], "twin": False}
                    nodes = compile_plan(base)["nodes"]
                    if _hops(nodes) > max_hops:
                        continue
                    yield base, base
                    if end[0] == "answer" and _hops(nodes) <= TWIN_HOPS:
                        yield {**base, "twin": True}, base
                    act = _action_of(base)
                    for kind in ("retry", "fallback"):
                        i = _fault_index(base, nodes, kind)
                        if i is not None:
                            yield {**base, "mod": [kind, i]}, base
                    if not (end[0] == "act" and end[2] == "name"):
                        yield {**base, "mod": ["notfound"]}, base
                    if act and act.approval:
                        yield {**base, "mod": ["denied"]}, base
                    if act and act.role:
                        yield {**base, "mod": ["refused"]}, base
    nt = {"start": None, "mode": None, "steps": [], "end": ["none"], "mod": ["notool"], "twin": False}
    yield nt, nt


def describe(plan: dict) -> dict:
    """Value-free description of a plan's gold graph."""
    c = compile_plan(plan)
    nodes = c["nodes"]
    idx = {n["id"]: n for n in nodes}
    sig = _signature(nodes, c["role"], c["status"], c["faults"])
    mod = plan["mod"][0]
    hops = 1 if mod in ("notfound", "refused", "notool") else _hops(nodes)
    return {"id": family_id(sig), "signature": sig, "hops": hops,
            "shape": _shape(plan, hops), "mod": mod, "tools": sorted({n["tool"] for n in nodes}),
            "edges": sorted({f"{idx[d]['tool']}>{n['tool']}" for n in nodes for d in n["deps"]}),
            "calls": len(nodes)}


# ------------------------------------------------------------------ hold-outs and splits

def partition(descs: list[dict]) -> dict:
    """Assign domains, tools and edges to the practice space, `dev` and the sealed test splits by hash.

    `descs` are the descriptions of all base plans. Returns the sets that define every split."""
    order = sorted(DOMAINS, key=lambda d: _h(SALT, "domain", d))
    test_dom = set(order[:TEST_DOMAINS])
    dev_dom = set(order[TEST_DOMAINS:TEST_DOMAINS + DEV_DOMAINS])
    held_dom = test_dom | dev_dom
    test_tools, dev_tools = set(), set()
    for kind in TEST_TOOL_QUOTA:
        cands = sorted((k for k, t in TOOLS2.items() if t.kind == kind and t.domain not in held_dom),
                       key=lambda k: _h(SALT, "tool", k))
        test_tools |= set(cands[:TEST_TOOL_QUOTA[kind]])
        dev_tools |= set(cands[TEST_TOOL_QUOTA[kind]:TEST_TOOL_QUOTA[kind] + DEV_TOOL_QUOTA[kind]])
    for s in (test_tools, dev_tools):  # a mirror follows the lookup it mirrors
        s |= {BACKUP_OF[k] for k in s if k in BACKUP_OF}
    practice_tools = {k for k, t in TOOLS2.items() if t.domain not in held_dom} - test_tools - dev_tools
    edges = sorted({e for d in descs if d["hops"] <= PRACTICE_HOPS for e in d["edges"]
                    if set(e.split(">")) <= practice_tools}, key=lambda e: _h(SALT, "edge", e))
    out_deg, in_deg = defaultdict(int), defaultdict(int)
    for e in edges:
        s, t = e.split(">")
        out_deg[s] += 1
        in_deg[t] += 1
    held: dict[str, set] = {"test": set(), "dev": set()}
    quota = {"test": round(TEST_EDGE_FRAC * len(edges)), "dev": round(DEV_EDGE_FRAC * len(edges))}
    for e in edges:  # an edge is held out only if both of its tools stay practised through other edges
        s, t = e.split(">")
        which = "test" if len(held["test"]) < quota["test"] else "dev" if len(held["dev"]) < quota["dev"] else None
        if which and out_deg[s] > 1 and in_deg[t] > 1:
            held[which].add(e)
            out_deg[s] -= 1
            in_deg[t] -= 1
    return {"test_domains": sorted(test_dom), "dev_domains": sorted(dev_dom), "test_tools": sorted(test_tools),
            "dev_tools": sorted(dev_tools), "practice_tools": sorted(practice_tools),
            "test_edges": sorted(held["test"]), "dev_edges": sorted(held["dev"]),
            "practice_edges": sorted(set(edges) - held["test"] - held["dev"])}


def classify(d: dict, part: dict) -> str | None:
    """The split a graph belongs to, from its tools, edges and number of hops; None if it mixes hold-outs."""
    tools, edges = set(d["tools"]), set(d["edges"])
    doms = {TOOLS2[t].domain for t in tools}
    shallow = d["hops"] <= PRACTICE_HOPS
    for name, held in (("test", set(part["test_domains"])), ("dev", set(part["dev_domains"]))):
        if doms & held:
            return f"{name}_domain" if doms <= held and shallow else None
    t_test, t_dev = tools & set(part["test_tools"]), tools & set(part["dev_tools"])
    e_test, e_dev = edges & set(part["test_edges"]), edges & set(part["dev_edges"])
    if t_test or t_dev:
        if (t_test and t_dev) or e_test or e_dev or not shallow:
            return None
        return "test_tool" if t_test else "dev_tool"
    if e_test or e_dev:
        if (e_test and e_dev) or not shallow:
            return None
        return "test_comp" if e_test else "dev_comp"
    return "practice" if shallow else "test_depth"


def build_space(max_hops: int = MAX_HOPS) -> dict:
    """Enumerate every family and assign it to exactly one split.

    A family takes the split of its own graph. Its request texts come only from plans whose
    base graph lies in the same split, so that a practice family (for instance "the record
    does not exist") is never phrased as a held-out composition."""
    rows, memo = [], {}
    for p, base in enumerate_plans(max_hops):
        if id(base) not in memo:
            memo[id(base)] = describe(base)
        rows.append((p, memo[id(base)] if p is base else describe(p), memo[id(base)]))
    part = partition([d for p, d, _ in rows if p["mod"][0] in ("none", "notool") and not p["twin"]])
    fams: dict[str, dict] = {}
    for plan, d, base in rows:
        split = classify(d, part)
        if split is None or classify(base, part) != split:
            continue
        f = fams.setdefault(d["id"], {**d, "split": split, "plans": [], "graphs": set()})
        assert f["signature"] == d["signature"] and f["split"] == split
        f["plans"].append(plan)
        f["graphs"].add(base["id"])
    for f in fams.values():
        f["graphs"] = sorted(f["graphs"])  # the base graphs (no modifier) this family is derived from
    by_split = {s: sorted((f for f in fams.values() if f["split"] == s), key=lambda f: f["id"]) for s in SPLITS}
    return {"partition": part, "families": by_split}


def stratum(f: dict) -> tuple:
    mod = f["mod"] if f["mod"] in ("none", "retry", "fallback", "denied") else "other"
    return (f["hops"], mod)


MOD_SHARE = {"none": 0.55, "retry": 0.15, "fallback": 0.10, "denied": 0.10, "other": 0.10}


def balanced_order(families: list[dict], seed, hops: tuple[int, ...], horizon: int) -> list[dict]:
    """A seeded order of `families` whose prefixes all have the same composition.

    The composition is the one a subset of `horizon` families can have when it is spread as
    evenly as possible over the numbers of hops with the modifier classes in fixed proportions
    (`MOD_SHARE`): a class with too few families contributes all it has and the rest is shared
    out. Every prefix then follows those proportions, so nested subsets (prefixes of one order)
    differ in how many families they hold and not in their mix of depth and behaviour."""
    rng = random.Random(f"{SALT}|order|{seed}")
    pools: dict[tuple, list] = defaultdict(list)
    for f in sorted(families, key=lambda f: f["id"]):
        if f["hops"] in hops:
            pools[stratum(f)].append(f)
    for p in pools.values():
        rng.shuffle(p)
    share = {}
    for h in hops:
        mods = [m for m in MOD_SHARE if (h, m) in pools]
        tot = sum(MOD_SHARE[m] for m in mods)
        for m in mods:
            share[(h, m)] = MOD_SHARE[m] / tot / len(hops)

    def sequence(share: dict) -> list[tuple]:
        left = {s: len(p) for s, p in pools.items()}
        seq, taken = [], defaultdict(int)
        while any(left.values()):
            live = [s for s in sorted(left) if left[s]]
            s = max(live, key=lambda s: (share[s] * (len(seq) + 1) - taken[s], -live.index(s)))
            seq.append(s)
            taken[s] += 1
            left[s] -= 1
        return seq

    head = sequence(share)[:horizon]
    seq = sequence({s: head.count(s) / len(head) for s in pools})
    return [pools[s].pop() for s in seq]


# ------------------------------------------------------------------ instances

class Builder:
    """Seeded world and wording for one task. `pool` separates training from evaluation instances:
    identifiers come from disjoint number ranges, and the names that a request mentions come from
    disjoint name lists, so no evaluation request can coincide with a training request."""

    def __init__(self, rng: random.Random, pool: str):
        assert pool in ("train", "eval")
        self.rng, self.pool = rng, pool
        self.w = empty_world()
        self.today = date(2026, 1, 5) + timedelta(days=rng.randrange(320))
        self.used: set[str] = set()
        self.reserved_phrasing = pool == "eval" and rng.random() < 0.5
        self.used_reserved = False
        self.rules: list[str] = []

    def variant(self) -> int:
        if self.reserved_phrasing and self.rng.random() < 0.6:
            self.used_reserved = True
            return 2
        return self.rng.randrange(2)

    def uid(self, ent: Entity) -> str:
        lo, hi = (1000, 6000) if self.pool == "train" else (6000, 10000)
        while True:
            s = f"{ent.prefix}-{self.rng.randrange(lo, hi)}"
            if s not in self.used:
                self.used.add(s)
                return s

    def _name(self, kind: str, target: bool) -> str:
        reserved = target and self.pool == "eval"
        for attempt in range(200):
            if kind == "person":
                first = FIRST[30:] if reserved else FIRST[:30]
                n = f"{self.rng.choice(first)} {self.rng.choice(LAST)}"
            else:
                pool = NAMES[kind]
                n = self.rng.choice(pool[18:] if reserved else pool[:18])
                if attempt >= 40:  # the list is used up in this world: number further records
                    n = f"{n} {['II', 'III', 'IV', 'V', 'VI'][self.rng.randrange(5)]}"
            if n not in self.used:
                self.used.add(n)
                return n
        raise RuntimeError(f"no free {kind} name")

    def record(self, ent: Entity, target: bool = False, **fixed) -> dict:
        rec = {ent.id_field: self.uid(ent)}
        if ent.named:
            rec["name"] = self._name(ent.named, target)
        for a in ent.attrs:
            rec[a.name] = self._value(a, rec)
        for r in ent.refs:
            rec[r.field] = self.uid(ENTITIES[r.target])  # points nowhere unless the plan fills it in
        rec.update(fixed)
        self.w[ent.name][rec[ent.id_field]] = rec
        if target or not self.w[ent.name].keys() - {rec[ent.id_field]}:
            for _ in range(self.rng.choice([1, 2])):  # other records of the same type
                if len(self.w[ent.name]) < 4:
                    self.record(ent)
        return rec

    def _value(self, a, rec):
        k = a.kind
        if k[0] == "enum":
            return self.rng.choice(k[1])
        if k[0] == "int":
            return self.rng.randrange(k[1], k[2] + 1)
        if k[0] == "money":
            return round(self.rng.uniform(k[1], k[2]), 2)
        if k[0] == "date":
            return (self.today + timedelta(days=self.rng.randrange(-200, 200))).isoformat()
        if k[0] == "city":
            return self.rng.choice(CITIES)
        slug = "".join(c for c in rec.get("name", "office").lower().replace(" ", ".") if c.isalnum() or c == ".")
        return f"{slug}@{self.rng.choice(['mailbox.org', 'postbox.net', 'inbox.eu'])}"

    @staticmethod
    def fact(v) -> str:
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    def mention(self, ent: Entity, rec: dict, by_name: bool, bare: bool = False) -> str:
        if not by_name:
            return f"{ent.noun} {rec[ent.id_field]}"
        name = f"'{rec['name']}'" if ent.named in QUOTED else rec["name"]
        return name if bare and ent.named == "person" else f"the {ent.noun} {name}"

    def children(self, parent: Entity, prec: dict, child: Entity, ref_field: str, np_: str):
        """Create the records listed under `prec`, choose one by a rule, and phrase that choice.

        The rule is evaluated over everything the list tool will return, which can include a
        record the path came from, and it singles out exactly one record."""
        pid, cid = prec[parent.id_field], child.id_field
        rules = [(how, a) for how in ("min", "max") for a in child.attrs if a.rank]
        if child.match:
            rules.append(("match", child.attr(child.match)))
        how, a = self.rng.choice(rules)
        kids = [self.record(child, **{ref_field: pid}) for _ in range(self.rng.choice([2, 3, 3, 4]))]
        new = {kid[cid] for kid in kids}
        listed = [r for r in self.w[child.name].values() if r[ref_field] == pid]
        if how == "match":
            pick = self.rng.choice(kids)
            taken = {r[a.name] for r in listed if r[cid] not in new}
            pick[a.name] = self.rng.choice([v for v in a.kind[1] if v not in taken])
            for kid in kids:
                if kid is not pick:
                    kid[a.name] = self.rng.choice([v for v in a.kind[1] if v != pick[a.name]])
        else:
            for attempt in range(400):
                best = (min if (how == "min") == (attempt < 200) else max)(listed, key=lambda r: r[a.name])
                if len({r[a.name] for r in listed}) == len(listed) and best[cid] in new:
                    break
                for kid in kids:
                    kid[a.name] = self._value(a, kid)
            else:
                raise RuntimeError("could not make a unique extreme")
            how, pick = ("min" if best is min(listed, key=lambda r: r[a.name]) else "max"), best
        other = self.record(parent)  # another parent with its own records, so that the filter matters
        for _ in range(self.rng.choice([1, 2])):
            self.record(child, **{ref_field: other[parent.id_field]})
        words = dict(c=child.noun, cs=child.plural, p=_ref(child, ref_field).listed, x=np_, a=a.phrase)
        if how == "match":
            np_ = MATCH_P[self.variant()].format(v=pick[a.name], **words)
        else:
            lo_hi = ("earliest", "latest") if a.kind[0] == "date" else ("lowest", "highest")
            np_ = RANK_P[self.variant()].format(w=lo_hi[how == "max"], **words)
        self.rules.append(how)
        return pick, np_

    def date_phrase(self) -> tuple[str, str]:
        r = self.rng.random()
        if r < 0.2:
            d, p = self.today + timedelta(days=1), "tomorrow"
        elif r < 0.45:
            d = self.today + timedelta(days=self.rng.randrange(2, 7))
            p = WEEKDAYS[d.weekday()]
        elif r < 0.85:
            d = self.today + timedelta(days=self.rng.randrange(2, 40))
            p = f"{d.day} {MONTHS[d.month - 1]}" if self.rng.random() < 0.5 else f"{MONTHS[d.month - 1]} {d.day}"
        else:
            d = self.today + timedelta(days=self.rng.randrange(2, 40))
            p = d.isoformat()
        return p, d.isoformat()


def make_task(family: dict, seed: int, pool: str, split: str, distractors: list[str]) -> Task:
    """One task of `family`. `distractors` is the set of tools that may be shown beside the gold ones."""
    rng = random.Random(f"{family['id']}|{seed}|{pool}")
    plan = rng.choice(family["plans"])
    b = Builder(rng, pool)
    c = compile_plan(plan, b)
    mod, info = plan["mod"][0], c["info"]
    facts: list[str] = []
    if mod == "notool":
        if rng.random() < 0.5:
            ent = rng.choice(sorted(ENTITIES.values(), key=lambda e: e.name))
            rid = b.uid(ent)
            request = (f"Note {ent.noun} {rid} for later. There is no need to look it up, "
                       f"just confirm it back to me.")
            answer, facts = f"Noted: {ent.noun} {rid}.", [rid]
        else:
            said = SMALLTALK[1] if pool == "eval" else SMALLTALK[0]
            request, answer = rng.choice(said), "You're welcome!"
    elif "asks" in info[0]:
        asks = [a for i in info for a in i["asks"]]
        v = b.variant()
        if plan.get("twin"):
            request = TWIN_Q[v].format(a=asks[0][0], x=asks[0][1], y=asks[1][1])
        elif len(asks) == 2:
            request = PAIR_Q[v].format(z=info[0]["about"], a=asks[0][0], x=asks[0][1], b=asks[1][0], y=asks[1][1])
        else:
            request = ANSWER_Q[v].format(a=asks[0][0], x=asks[0][1])
        answer = " ".join(f"The {a} of {x} is {f}." for a, x, f in asks)
        if len(asks) == 2 and not plan.get("twin"):
            answer = f"For {info[0]['about']}: " + answer
        facts = [f for _, _, f in asks]
    else:
        request, answer = info[0]["request"], info[0]["answer"]
    if mod == "notfound":
        answer, facts = f"I could not find {info[0]['start_np']}.", []
    elif mod == "denied":
        answer = "The change was not made because approval was denied."
    elif mod == "refused":
        answer = f"Your role does not allow {info[0]['action'].name}."

    gold = [GoldCall(n["id"], n["tool"], n["args"], deps=n["deps"], optional=n.get("optional", False),
                     fallback=n.get("fallback"), execute=n.get("execute", True)) for n in c["nodes"]]
    required = {g.tool for g in gold} | set(c["all_tools"])  # the whole requested path stays available
    required |= {BACKUP_OF[t] for t in required if t in BACKUP_OF}
    pool_ = sorted(set(distractors) - required)
    ents = {TOOLS2[t].entity for t in required} | {TOOLS2[t].parent for t in required if TOOLS2[t].parent}
    near = [t for t in pool_ if TOOLS2[t].entity in ents or TOOLS2[t].parent in ents]
    k = rng.randrange(2, 7)
    chosen = rng.sample(near, min(len(near), (k + 1) // 2))
    rest = [t for t in pool_ if t not in chosen]
    chosen += rng.sample(rest, min(len(rest), k - len(chosen)))
    names = sorted(required) + chosen
    rng.shuffle(names)

    tags = [f"hops={family['hops']}", f"mod={mod}", f"phr={'reserved' if b.used_reserved else 'seen'}",
            f"calls={len(gold)}"] + [f"rule={r}" for r in b.rules] + (["not_found"] if mod == "notfound" else [])
    task = Task(task_id="", family=family["shape"], template=family["id"], split=split, role=c["role"],
                today=b.today.isoformat(), request=request, catalog=[TOOLS2[n].spec for n in names], world=b.w,
                faults=c["faults"], approvals=c["approvals"], gold=gold, expected_status=c["status"],
                answer_facts=facts, reference_answer=answer, depth=family["hops"], tags=tags)
    task.task_id = f"{split}-{family['id']}-{seed}-{task_hash(task)[:6]}"
    return task
