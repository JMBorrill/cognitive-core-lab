"""Tool specifications, deterministic world state and tool simulators.

Every tool is a pure function of (world, args) -> result. The world is a plain
dict of entity tables generated from a seed, so episodes are fully replayable.
"""

from __future__ import annotations

import ast
import copy
import operator
import re
from dataclasses import dataclass, field, replace

DATE_RE = r"^\d{4}-\d{2}-\d{2}$"
TIME_RE = r"^\d{2}:\d{2}$"
EMAIL_RE = r"^[^@\s]+@[^@\s]+\.[a-z]+$"


@dataclass(frozen=True)
class Param:
    name: str
    type: str  # string | integer | number | boolean | array
    description: str = ""
    required: bool = True
    enum: tuple[str, ...] | None = None
    pattern: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: tuple[Param, ...]
    domain: str
    impl: str = ""  # simulator key; defaults to name
    consequential: bool = False  # requires human approval
    required_role: str | None = None
    # Catalogue renaming support: maps exposed param name -> canonical name.
    param_map: tuple[tuple[str, str], ...] = ()

    @property
    def key(self) -> str:
        return self.impl or self.name

    def canonical_args(self, args: dict) -> dict:
        m = dict(self.param_map)
        return {m.get(k, k): v for k, v in args.items()}

    def exposed_param(self, canonical: str) -> str:
        for exposed, canon in self.param_map:
            if canon == canonical:
                return exposed
        return canonical

    def renamed(self, name: str, params: dict[str, str] | None = None) -> "ToolSpec":
        params = params or {}
        new_params = tuple(replace(p, name=params.get(p.name, p.name)) for p in self.params)
        pmap = tuple((params.get(p.name, p.name), p.name) for p in self.params)
        return replace(self, name=name, impl=self.key, params=new_params, param_map=pmap)

    def signature(self) -> str:
        parts = []
        for p in self.params:
            t = p.type
            if p.enum:
                t = "|".join(p.enum)
            elif p.pattern == DATE_RE:
                t = "string YYYY-MM-DD"
            elif p.pattern == TIME_RE:
                t = "string HH:MM"
            parts.append(f"{p.name}{'' if p.required else '?'}: {t}")
        tags = []
        if self.consequential:
            tags.append("[approval]")
        if self.required_role:
            tags.append(f"[role: {self.required_role}]")
        tag = (" " + " ".join(tags)) if tags else ""
        return f"{self.name}({', '.join(parts)}){tag} - {self.description}"


class ToolError(Exception):
    def __init__(self, code: str, detail: str, retryable: bool = False):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail, self.retryable = code, detail, retryable

    def as_obs(self) -> dict:
        return {"error": self.code, "detail": self.detail, "retryable": self.retryable}


P = Param
TOOLS: dict[str, ToolSpec] = {
    t.name: t
    for t in [
        ToolSpec("find_contact", "Look up a person in the address book by name.",
                 (P("name", "string"),), "contacts"),
        ToolSpec("list_events", "List calendar events on a date.",
                 (P("date", "string", pattern=DATE_RE),), "calendar"),
        ToolSpec("create_event", "Create a calendar event with attendees (contact IDs).",
                 (P("title", "string"), P("date", "string", pattern=DATE_RE),
                  P("start_time", "string", pattern=TIME_RE), P("attendee_ids", "array")), "calendar"),
        ToolSpec("send_email", "Send an email.",
                 (P("to", "string", pattern=EMAIL_RE), P("subject", "string"), P("body", "string")),
                 "email", consequential=True),
        ToolSpec("get_weather", "Weather forecast for a city on a date.",
                 (P("city", "string"), P("date", "string", pattern=DATE_RE)), "weather"),
        ToolSpec("weather_backup", "Secondary weather provider. Use only if get_weather is unavailable.",
                 (P("city", "string"), P("date", "string", pattern=DATE_RE)), "weather"),
        ToolSpec("convert_currency", "Convert an amount between currencies (ISO codes).",
                 (P("amount", "number"), P("from_currency", "string", pattern=r"^[A-Z]{3}$"),
                  P("to_currency", "string", pattern=r"^[A-Z]{3}$")), "fx"),
        ToolSpec("lookup_order", "Get an order by order ID.",
                 (P("order_id", "string"),), "orders"),
        ToolSpec("get_customer", "Get a customer record by customer ID.",
                 (P("customer_id", "string"),), "orders"),
        ToolSpec("issue_refund", "Refund an order (amount in order currency).",
                 (P("order_id", "string"), P("amount", "number")), "orders",
                 consequential=True, required_role="support_agent"),
        ToolSpec("search_files", "Search documents by keyword; returns file IDs.",
                 (P("query", "string"),), "files"),
        ToolSpec("read_file", "Read a document by file ID.",
                 (P("file_id", "string"),), "files"),
        ToolSpec("delete_file", "Permanently delete a document.",
                 (P("file_id", "string"),), "files", consequential=True, required_role="admin"),
        ToolSpec("create_ticket", "Open a support ticket.",
                 (P("title", "string"), P("priority", "string", enum=("low", "medium", "high"))), "tickets"),
        ToolSpec("calculator", "Evaluate an arithmetic expression.",
                 (P("expression", "string"),), "math"),
        ToolSpec("search_flights", "Search flights between airports (IATA codes) on a date.",
                 (P("origin", "string", pattern=r"^[A-Z]{3}$"), P("destination", "string", pattern=r"^[A-Z]{3}$"),
                  P("date", "string", pattern=DATE_RE)), "flights"),
        ToolSpec("book_flight", "Book a flight for a passenger.",
                 (P("flight_id", "string"), P("passenger_name", "string")), "flights", consequential=True),
        # Held-out domains: never shown during training (unseen-tool evaluation).
        ToolSpec("check_stock", "Check inventory for a SKU at a warehouse.",
                 (P("sku", "string"), P("warehouse", "string")), "inventory"),
        ToolSpec("reserve_stock", "Reserve units of a SKU at a warehouse.",
                 (P("sku", "string"), P("warehouse", "string"), P("quantity", "integer")), "inventory"),
        ToolSpec("geocode", "Convert a place name to coordinates.",
                 (P("place", "string"),), "maps"),
        ToolSpec("get_timezone", "Time zone for coordinates.",
                 (P("lat", "number"), P("lon", "number")), "maps"),
    ]
}

HELDOUT_DOMAINS = frozenset({"inventory", "maps"})
# A plain set so that later tool tables (ccl.world2) can register their own state-changing tools.
MUTATING = {"create_event", "send_email", "issue_refund", "delete_file", "create_ticket",
            "book_flight", "reserve_stock"}


# ---------------------------------------------------------------- validation

def validate_args(spec: ToolSpec, args) -> dict:
    """Validate exposed args against the spec; return canonical args or raise ToolError."""
    if not isinstance(args, dict):
        raise ToolError("invalid_argument", "args must be an object")
    names = {p.name for p in spec.params}
    unknown = sorted(set(args) - names)
    if unknown:
        raise ToolError("invalid_argument", f"unknown parameter(s): {', '.join(unknown)}")
    for p in spec.params:
        if p.name not in args:
            if p.required:
                raise ToolError("invalid_argument", f"missing required parameter: {p.name}")
            continue
        v = args[p.name]
        ok = {
            "string": isinstance(v, str),
            "integer": isinstance(v, int) and not isinstance(v, bool),
            "number": isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": isinstance(v, bool),
            "array": isinstance(v, list),
        }[p.type]
        if not ok:
            raise ToolError("invalid_argument", f"{p.name} must be of type {p.type}")
        if p.type == "array" and not all(isinstance(x, str) for x in v):
            raise ToolError("invalid_argument", f"{p.name} must be a list of strings")
        if p.enum and v not in p.enum:
            raise ToolError("invalid_argument", f"{p.name} must be one of {list(p.enum)}")
        if p.pattern and not re.match(p.pattern, v):
            raise ToolError("invalid_argument", f"{p.name} has invalid format")
    return spec.canonical_args(args)


# ---------------------------------------------------------------- simulators

def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def _find_contact(w, a):
    hits = [c for c in w["contacts"].values() if _norm(c["name"]) == _norm(a["name"])]
    if not hits:
        hits = [c for c in w["contacts"].values() if _norm(a["name"]) in _norm(c["name"])]
    if not hits:
        raise ToolError("not_found", f"no contact named {a['name']}")
    return {"matches": [dict(c) for c in hits[:3]]}


def _list_events(w, a):
    return {"events": [e for e in w["events"] if e["date"] == a["date"]]}


def _create_event(w, a):
    for cid in a["attendee_ids"]:
        if cid not in w["contacts"]:
            raise ToolError("invalid_argument", f"unknown attendee id {cid}")
    eid = f"EV-{1000 + len(w['events'])}"
    w["events"].append({"event_id": eid, "title": a["title"], "date": a["date"],
                        "start_time": a["start_time"], "attendee_ids": list(a["attendee_ids"])})
    return {"event_id": eid, "status": "created"}


def _send_email(w, a):
    mid = f"MSG-{500 + len(w['outbox'])}"
    w["outbox"].append({"message_id": mid, **a})
    return {"message_id": mid, "status": "sent"}


def _weather(w, a):
    key = f"{_norm(a['city'])}|{a['date']}"
    if key not in w["weather"]:
        raise ToolError("not_found", f"no forecast for {a['city']} on {a['date']}")
    return {"city": a["city"], "date": a["date"], **w["weather"][key]}


def _convert(w, a):
    rates = w["fx"]
    f, t = a["from_currency"], a["to_currency"]
    if f not in rates or t not in rates:
        raise ToolError("invalid_argument", "unsupported currency")
    val = round(a["amount"] * rates[t] / rates[f], 2)
    return {"amount": val, "currency": t}


def _lookup_order(w, a):
    o = w["orders"].get(a["order_id"].strip().upper())
    if not o:
        raise ToolError("not_found", f"order {a['order_id']} not found")
    return dict(o)


def _get_customer(w, a):
    c = w["customers"].get(a["customer_id"])
    if not c:
        raise ToolError("not_found", f"customer {a['customer_id']} not found")
    return dict(c)


def _refund(w, a):
    o = w["orders"].get(a["order_id"])
    if not o:
        raise ToolError("not_found", f"order {a['order_id']} not found")
    if a["amount"] > o["total"] + 1e-9:
        raise ToolError("invalid_argument", "refund exceeds order total")
    rid = f"RF-{700 + len(w['refunds'])}"
    w["refunds"].append({"refund_id": rid, "order_id": a["order_id"], "amount": a["amount"]})
    return {"refund_id": rid, "status": "refunded"}


def _search_files(w, a):
    q = _norm(a["query"])
    hits = [{"file_id": f["file_id"], "name": f["name"]} for f in w["files"].values()
            if all(tok in _norm(f["name"] + " " + f["tags"]) for tok in q.split())]
    return {"results": hits[:5]}


def _read_file(w, a):
    f = w["files"].get(a["file_id"])
    if not f:
        raise ToolError("not_found", f"file {a['file_id']} not found")
    return {"file_id": f["file_id"], "name": f["name"], "content": f["content"]}


def _delete_file(w, a):
    if a["file_id"] not in w["files"]:
        raise ToolError("not_found", f"file {a['file_id']} not found")
    w["deleted"].append(a["file_id"])
    del w["files"][a["file_id"]]
    return {"status": "deleted"}


def _create_ticket(w, a):
    tid = f"TK-{300 + len(w['tickets'])}"
    w["tickets"].append({"ticket_id": tid, **a})
    return {"ticket_id": tid, "status": "open"}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg, ast.Pow: operator.pow}


def safe_eval(expr: str) -> float:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 10:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr.replace("^", "**").replace("x", "*").replace("×", "*"), mode="eval"))


def _calculator(w, a):
    try:
        v = safe_eval(a["expression"])
    except Exception:
        raise ToolError("invalid_argument", "could not parse expression")
    return {"result": round(v, 4)}


def _search_flights(w, a):
    fl = [f for f in w["flights"] if (f["origin"], f["destination"], f["date"]) ==
          (a["origin"], a["destination"], a["date"])]
    return {"flights": [{k: f[k] for k in ("flight_id", "depart", "price")} for f in fl]}


def _book_flight(w, a):
    if not any(f["flight_id"] == a["flight_id"] for f in w["flights"]):
        raise ToolError("not_found", f"flight {a['flight_id']} not found")
    bid = f"BK-{900 + len(w['bookings'])}"
    w["bookings"].append({"booking_id": bid, **a})
    return {"booking_id": bid, "status": "confirmed"}


def _check_stock(w, a):
    k = f"{a['sku'].upper()}|{_norm(a['warehouse'])}"
    if k not in w["stock"]:
        raise ToolError("not_found", "unknown sku/warehouse")
    return {"sku": a["sku"].upper(), "warehouse": a["warehouse"], "quantity": w["stock"][k]}


def _reserve_stock(w, a):
    k = f"{a['sku'].upper()}|{_norm(a['warehouse'])}"
    if k not in w["stock"]:
        raise ToolError("not_found", "unknown sku/warehouse")
    if a["quantity"] <= 0 or a["quantity"] > w["stock"][k]:
        raise ToolError("invalid_argument", "insufficient stock")
    w["stock"][k] -= a["quantity"]
    rid = f"RS-{400 + len(w['reservations'])}"
    w["reservations"].append({"reservation_id": rid, **a})
    return {"reservation_id": rid, "status": "reserved"}


def _geocode(w, a):
    g = w["places"].get(_norm(a["place"]))
    if not g:
        raise ToolError("not_found", f"unknown place {a['place']}")
    return {"place": a["place"], **g}


def _timezone(w, a):
    for g in w["places"].values():
        if abs(g["lat"] - a["lat"]) < 0.01 and abs(g["lon"] - a["lon"]) < 0.01:
            return {"timezone": g["tz"], "utc_offset": g["utc_offset"]}
    raise ToolError("not_found", "no timezone for coordinates")


SIMULATORS = {
    "find_contact": _find_contact, "list_events": _list_events, "create_event": _create_event,
    "send_email": _send_email, "get_weather": _weather, "weather_backup": _weather,
    "convert_currency": _convert, "lookup_order": _lookup_order, "get_customer": _get_customer,
    "issue_refund": _refund, "search_files": _search_files, "read_file": _read_file,
    "delete_file": _delete_file, "create_ticket": _create_ticket, "calculator": _calculator,
    "search_flights": _search_flights, "book_flight": _book_flight, "check_stock": _check_stock,
    "reserve_stock": _reserve_stock, "geocode": _geocode, "get_timezone": _timezone,
}
assert set(SIMULATORS) == set(TOOLS)


@dataclass
class FaultPlan:
    """Deterministic fault injection: tool key -> list of per-call error codes (None = ok).

    `down` tools fail permanently with service_unavailable."""
    transient: dict[str, list[str | None]] = field(default_factory=dict)
    down: tuple[str, ...] = ()


class Simulator:
    def __init__(self, world: dict, faults: FaultPlan | None = None):
        self.world = copy.deepcopy(world)
        self.faults = faults or FaultPlan()
        self.calls: dict[str, int] = {}

    def execute(self, spec: ToolSpec, canonical_args: dict) -> dict:
        key = spec.key
        n = self.calls.get(key, 0)
        self.calls[key] = n + 1
        if key in self.faults.down:
            raise ToolError("service_unavailable", f"{spec.name} is currently unavailable")
        plan = self.faults.transient.get(key, [])
        if n < len(plan) and plan[n]:
            raise ToolError(plan[n], "request timed out, try again", retryable=True)
        try:
            return SIMULATORS[key](self.world, canonical_args)
        except (TypeError, ValueError, KeyError, AttributeError) as e:
            # malformed model input must surface as a tool error, never crash the episode loop
            raise ToolError("invalid_argument", f"could not process arguments: {type(e).__name__}") from e
