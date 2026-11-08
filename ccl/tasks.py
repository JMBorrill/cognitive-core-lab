"""Task generation: seeded worlds, scenario templates, gold task graphs and splits.

A Task bundles everything needed to run and verify one episode: the user request,
the exposed tool catalogue, the initial world, fault/approval plans and a gold
dependency graph (DAG) of tool calls with argument-matching rules.

Leakage control happens at two levels:
  * template level - templates are assigned to exactly one of {train, ood};
    OOD templates (and the held-out tool domains) never appear in training data;
  * graph level   - `graph_signature` canonicalises the gold DAG; the split audit
    checks that no OOD signature occurs in the training pool.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

from .tools import HELDOUT_DOMAINS, TOOLS, FaultPlan, Param, ToolSpec, safe_eval

# ------------------------------------------------------------------ data model


@dataclass
class GoldCall:
    id: str
    tool: str  # canonical tool name
    args: dict  # canonical args
    deps: list[str] = field(default_factory=list)
    match: dict = field(default_factory=dict)  # param -> exact|num|expr|contains|set|any
    optional: bool = False  # not required for success (e.g. a sanity lookup)
    fallback: str | None = None  # canonical alternative tool if this one is down
    execute: bool = True  # False when the call is expected to be denied/blocked
    display: dict = field(default_factory=dict)  # concrete values for free-text params (oracle use)


@dataclass
class Task:
    task_id: str
    family: str
    template: str
    split: str
    role: str
    today: str
    request: str
    catalog: list[ToolSpec]
    world: dict
    faults: FaultPlan
    approvals: dict  # canonical tool -> bool
    gold: list[GoldCall]
    expected_status: str
    answer_facts: list[str]
    reference_answer: str
    depth: int
    tags: list[str] = field(default_factory=list)

    def tool_by_name(self, name: str) -> ToolSpec | None:
        return next((t for t in self.catalog if t.name == name), None)

    def exposed(self, canonical: str) -> ToolSpec:
        return next(t for t in self.catalog if t.key == canonical)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["catalog"] = [spec_to_dict(t) for t in self.catalog]
        d["faults"] = {"transient": self.faults.transient, "down": list(self.faults.down)}
        return d

    @staticmethod
    def from_dict(d: dict) -> "Task":
        d = dict(d)
        d["catalog"] = [spec_from_dict(t) for t in d["catalog"]]
        d["faults"] = FaultPlan(d["faults"]["transient"], tuple(d["faults"]["down"]))
        d["gold"] = [GoldCall(**g) for g in d["gold"]]
        return Task(**d)


def spec_to_dict(t: ToolSpec) -> dict:
    d = asdict(t)
    d["params"] = [asdict(p) for p in t.params]
    return d


def spec_from_dict(d: dict) -> ToolSpec:
    d = dict(d)
    d["params"] = tuple(Param(**{**p, "enum": tuple(p["enum"]) if p["enum"] else None}) for p in d["params"])
    d["param_map"] = tuple(tuple(x) for x in d["param_map"])
    return ToolSpec(**d)


def graph_signature(task: Task) -> str:
    """Canonical, value-free description of the gold DAG (tools + edges + faults)."""
    idx = {g.id: g for g in task.gold}
    nodes = sorted(f"{g.tool}<-[{','.join(sorted(idx[d].tool for d in g.deps))}]"
                   f"{'?' if g.optional else ''}{'' if g.execute else '!x'}" for g in task.gold)
    fault = sorted(task.faults.down) + sorted(k for k, v in task.faults.transient.items() if any(v))
    return f"{task.role}|{task.expected_status}|{';'.join(nodes)}|faults={','.join(fault)}"


def task_hash(task: Task) -> str:
    return hashlib.sha256(json.dumps(task.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ value pools

FIRST = ["Alice", "Bruno", "Chen", "Dana", "Elif", "Farah", "Gustav", "Hana", "Ivan", "Jonas", "Keiko",
         "Liam", "Maya", "Nikhil", "Olga", "Pedro", "Quinn", "Rosa", "Sanjay", "Tara", "Umar", "Vera",
         "Wei", "Ximena", "Yusuf", "Zoe", "Amara", "Bjorn", "Carmen", "Dmitri", "Esther", "Femi"]
LAST = ["Abbott", "Barros", "Costa", "Dubois", "Eriksen", "Fischer", "Garcia", "Haddad", "Ito", "Jensen",
        "Kowalski", "Larsen", "Moreau", "Novak", "Okafor", "Patel", "Quist", "Rossi", "Silva", "Tanaka",
        "Ueda", "Varga", "Weber", "Xu", "Yilmaz", "Zimmer", "Andersen", "Bianchi", "Clarke", "Diaz"]
CITIES = ["Paris", "Lisbon", "Berlin", "Madrid", "Oslo", "Dublin", "Vienna", "Prague", "Warsaw", "Athens",
          "Rome", "Zurich", "Helsinki", "Budapest", "Copenhagen", "Brussels", "Milan", "Porto", "Krakow",
          "Seville", "Lyon", "Munich", "Geneva", "Riga"]
AIRPORTS = [("London", "LHR"), ("Paris", "CDG"), ("Madrid", "MAD"), ("Rome", "FCO"), ("Berlin", "BER"),
            ("Amsterdam", "AMS"), ("Lisbon", "LIS"), ("Dublin", "DUB"), ("Vienna", "VIE"), ("Oslo", "OSL")]
CONDITIONS = ["sunny", "cloudy", "rainy", "windy", "snowy", "foggy", "stormy", "clear"]
CURRENCIES = [("USD", "US dollars", "$"), ("EUR", "euros", "€"), ("GBP", "British pounds", "£"),
              ("JPY", "Japanese yen", "¥"), ("CHF", "Swiss francs", None), ("CAD", "Canadian dollars", None),
              ("SEK", "Swedish kronor", None), ("AUD", "Australian dollars", None)]
FX = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 151.3, "CHF": 0.88, "CAD": 1.36, "SEK": 10.4, "AUD": 1.52}
TOPICS = ["budget", "onboarding", "roadmap", "security", "hiring", "pricing", "vendor", "retention",
          "migration", "compliance", "launch", "quarterly"]
ORDER_STATUS = ["shipped", "processing", "delivered", "cancelled", "on hold"]
EVENT_TITLES = ["Design review", "Budget sync", "Team lunch", "Quarterly planning", "1:1", "Vendor call",
                "Sprint retro", "Board prep", "Customer demo", "Hiring panel"]
WAREHOUSES = ["Leeds", "Rotterdam", "Lyon", "Gdansk", "Bilbao", "Malmo"]
PLACES = [("Reykjavik", 64.15, -21.94, "Atlantic/Reykjavik", "+00:00"),
          ("Nairobi", -1.29, 36.82, "Africa/Nairobi", "+03:00"),
          ("Lima", -12.05, -77.04, "America/Lima", "-05:00"),
          ("Osaka", 34.69, 135.5, "Asia/Tokyo", "+09:00"),
          ("Perth", -31.95, 115.86, "Australia/Perth", "+08:00"),
          ("Denver", 39.74, -104.99, "America/Denver", "-07:00"),
          ("Mumbai", 19.08, 72.88, "Asia/Kolkata", "+05:30"),
          ("Santiago", -33.45, -70.67, "America/Santiago", "-04:00")]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]

# Alias pools for catalogue perturbation. Train-time augmentation and the
# "renamed" evaluation split use disjoint alias vocabularies.
ALIASES_TRAIN = {
    "get_weather": ("weather_forecast", {"city": "location"}),
    "convert_currency": ("fx_convert", {"from_currency": "source", "to_currency": "target"}),
    "lookup_order": ("order_details", {"order_id": "id"}),
    "get_customer": ("customer_profile", {"customer_id": "id"}),
    "find_contact": ("contact_search", {"name": "query"}),
    "send_email": ("email_send", {"to": "recipient"}),
    "search_files": ("doc_search", {"query": "keywords"}),
    "read_file": ("doc_open", {"file_id": "doc_id"}),
    "create_ticket": ("ticket_new", {"title": "summary"}),
    "list_events": ("calendar_day", {}),
}
ALIASES_EVAL = {
    "get_weather": ("forecast_lookup", {"city": "place", "date": "day"}),
    "weather_backup": ("forecast_mirror", {"city": "place", "date": "day"}),
    "convert_currency": ("money_exchange", {"amount": "value", "from_currency": "base", "to_currency": "quote"}),
    "lookup_order": ("fetch_order", {"order_id": "ref"}),
    "get_customer": ("fetch_client", {"customer_id": "client_ref"}),
    "find_contact": ("people_lookup", {"name": "person"}),
    "send_email": ("dispatch_mail", {"to": "address", "subject": "title", "body": "text"}),
    "search_files": ("find_documents", {"query": "terms"}),
    "read_file": ("open_document", {"file_id": "document"}),
    "create_ticket": ("raise_issue", {"title": "headline", "priority": "severity"}),
    "list_events": ("agenda_for", {"date": "day"}),
    "create_event": ("schedule_meeting", {"title": "topic", "attendee_ids": "invitees"}),
    "calculator": ("compute", {"expression": "formula"}),
    "issue_refund": ("refund_payment", {"order_id": "ref"}),
    "delete_file": ("purge_document", {"file_id": "document"}),
    "search_flights": ("flight_finder", {"origin": "from_airport", "destination": "to_airport"}),
    "book_flight": ("reserve_seat", {"flight_id": "flight", "passenger_name": "traveller"}),
}


# ------------------------------------------------------------------ generator helper


class Gen:
    def __init__(self, rng: random.Random, pool: str):
        self.rng = rng
        cut = lambda xs: xs if pool == "eval" else xs[: int(len(xs) * 0.75)]
        self.first, self.last, self.cities = cut(FIRST), cut(LAST), cut(CITIES)
        self.today = date(2026, 1, 5) + timedelta(days=rng.randrange(320))
        self.w = {"contacts": {}, "events": [], "outbox": [], "weather": {}, "fx": dict(FX), "orders": {},
                  "customers": {}, "refunds": [], "files": {}, "deleted": [], "tickets": [], "flights": [],
                  "bookings": [], "stock": {}, "reservations": [], "places": {}}
        self.used: set[str] = set()

    def pick(self, xs, k=None):
        return self.rng.choice(xs) if k is None else self.rng.sample(xs, k)

    def uid(self, prefix: str, lo=1000, hi=9999) -> str:
        while True:
            s = f"{prefix}{self.rng.randrange(lo, hi)}"
            if s not in self.used:
                self.used.add(s)
                return s

    def name(self) -> str:
        while True:
            n = f"{self.pick(self.first)} {self.pick(self.last)}"
            if n not in self.used:
                self.used.add(n)
                return n

    def contact(self, name=None) -> dict:
        name = name or self.name()
        first, last = name.lower().split()
        c = {"contact_id": self.uid("C-"), "name": name,
             "email": f"{first}.{last}@{self.pick(['acme.io', 'northwind.com', 'globex.net'])}",
             "phone": f"+44 20 {self.rng.randrange(1000, 9999)} {self.rng.randrange(1000, 9999)}"}
        self.w["contacts"][c["contact_id"]] = c
        return c

    def decoy_contacts(self, n=2):
        for _ in range(n):
            self.contact()

    def date_phrase(self) -> tuple[str, str]:
        r = self.rng.random()
        if r < 0.2:
            d, p = self.today, "today"
        elif r < 0.4:
            d, p = self.today + timedelta(days=1), "tomorrow"
        elif r < 0.6:
            k = self.rng.randrange(2, 7)
            d = self.today + timedelta(days=k)
            p = f"on {WEEKDAYS[d.weekday()]}"
        elif r < 0.85:
            d = self.today + timedelta(days=self.rng.randrange(2, 40))
            p = f"on {d.day} {MONTHS[d.month - 1]}" if self.rng.random() < 0.5 else f"on {MONTHS[d.month - 1]} {d.day}"
        else:
            d = self.today + timedelta(days=self.rng.randrange(2, 40))
            p = f"on {d.isoformat()}"
        return p, d.isoformat()

    def time_phrase(self) -> tuple[str, str]:
        h = self.rng.choice([9, 10, 11, 13, 14, 15, 16, 17])
        m = self.rng.choice([0, 0, 30])
        hh = h if h <= 12 else h - 12
        phrase = f"{hh}{':30' if m else ''}{'am' if h < 12 else 'pm'}" if self.rng.random() < 0.6 else f"{h:02d}:{m:02d}"
        return phrase, f"{h:02d}:{m:02d}"

    def weather(self, city, iso) -> dict:
        wx = {"temp_c": self.rng.randrange(-5, 33), "condition": self.pick(CONDITIONS)}
        self.w["weather"][f"{city.lower()}|{iso}"] = wx
        return wx

    def customer(self) -> dict:
        n = self.name()
        c = {"customer_id": self.uid("CU-"), "name": n, "email": n.lower().replace(" ", ".") + "@mail.com",
             "tier": self.pick(["standard", "gold", "platinum"])}
        self.w["customers"][c["customer_id"]] = c
        return c

    def order(self, customer=None, status=None) -> dict:
        customer = customer or self.customer()
        cur = self.pick(["USD", "EUR", "GBP"])
        o = {"order_id": self.uid("ORD-", 10000, 99999), "customer_id": customer["customer_id"],
             "status": status or self.pick(ORDER_STATUS), "total": round(self.rng.uniform(15, 900), 2),
             "currency": cur}
        self.w["orders"][o["order_id"]] = o
        return o

    def decoy_orders(self, n=2):
        for _ in range(n):
            self.order()

    def file(self, topic, content) -> dict:
        f = {"file_id": self.uid("F-"), "name": f"{topic.title()} {self.pick(['memo', 'notes', 'brief'])}",
             "tags": topic, "content": content}
        self.w["files"][f["file_id"]] = f
        return f

    def decoy_files(self, exclude, n=2):
        for t in self.pick([t for t in TOPICS if t != exclude], n):
            self.file(t, f"General {t} notes. Nothing actionable.")

    def flights(self, o, d, iso, n=3) -> list[dict]:
        fl = []
        for _ in range(n):
            f = {"flight_id": self.uid("FL", 100, 999), "origin": o, "destination": d, "date": iso,
                 "depart": f"{self.rng.randrange(6, 22):02d}:{self.pick(['00', '15', '30', '45'])}",
                 "price": self.rng.randrange(60, 450)}
            fl.append(f)
        prices = [f["price"] for f in fl]
        if len(set(prices)) < len(prices):  # ensure a unique cheapest
            for i, f in enumerate(fl):
                f["price"] += i * 7
        self.w["flights"].extend(fl)
        return fl


def money(x: float) -> str:
    return f"{x:.2f}"


def cur_phrase(g: Gen, code: str) -> str:
    c = next(c for c in CURRENCIES if c[0] == code)
    return g.pick([c[1], code])


# ------------------------------------------------------------------ templates
# Each template returns a dict of task fields; the driver fills in ids, catalog, etc.

TEMPLATES: dict[str, dict] = {}


def template(tid: str, family: str, split: str, depth: int):
    def deco(fn):
        TEMPLATES[tid] = {"fn": fn, "family": family, "split": split, "depth": depth}
        return fn
    return deco


def T(request, gold, status="done", facts=(), answer="", role="user", faults=None, approvals=None, tags=(),
      extra_tools=()):
    return {"request": request, "gold": gold, "expected_status": status, "answer_facts": list(facts),
            "reference_answer": answer, "role": role, "faults": faults or FaultPlan(),
            "approvals": approvals or {}, "tags": list(tags), "extra_tools": list(extra_tools)}


# --- select ---------------------------------------------------------------

@template("sel_weather", "select", "train", 1)
def _sel_weather(g: Gen):
    city = g.pick(g.cities)
    p, iso = g.date_phrase()
    wx = g.weather(city, iso)
    g.weather(g.pick([c for c in g.cities if c != city]), iso)
    req = g.pick([f"What's the weather in {city} {p}?", f"Will it be warm in {city} {p}? Give me the forecast.",
                  f"Forecast for {city} {p}, please."])
    return T(req, [GoldCall("g1", "get_weather", {"city": city, "date": iso})],
             facts=[str(wx["temp_c"]), wx["condition"]],
             answer=f"{city} on {iso}: {wx['temp_c']}°C and {wx['condition']}.")


@template("sel_fx", "select", "train", 1)
def _sel_fx(g: Gen):
    (a, b) = g.pick([c[0] for c in CURRENCIES], 2)
    amt = g.pick([20, 75, 120, 250, 1000, 49.5, 310])
    res = round(amt * FX[b] / FX[a], 2)
    req = g.pick([f"How much is {amt} {cur_phrase(g, a)} in {cur_phrase(g, b)}?",
                  f"Convert {amt} {cur_phrase(g, a)} to {cur_phrase(g, b)}."])
    return T(req, [GoldCall("g1", "convert_currency", {"amount": amt, "from_currency": a, "to_currency": b})],
             facts=[money(res)], answer=f"{amt} {a} is {money(res)} {b}.")


@template("sel_order_status", "select", "train", 1)
def _sel_order(g: Gen):
    o = g.order()
    g.decoy_orders()
    req = g.pick([f"What's the status of order {o['order_id']}?", f"Has order {o['order_id']} shipped yet?",
                  f"Check order {o['order_id']} for me."])
    return T(req, [GoldCall("g1", "lookup_order", {"order_id": o["order_id"]})], facts=[o["status"]],
             answer=f"Order {o['order_id']} is {o['status']}.")


@template("sel_calc", "select", "train", 1)
def _sel_calc(g: Gen):
    a, b, c = g.rng.randrange(12, 99), g.rng.randrange(3, 40), g.rng.randrange(100, 999)
    expr = f"{a} * {b} + {c}"
    res = a * b + c
    req = g.pick([f"What is {a} times {b} plus {c}?", f"Calculate {a} multiplied by {b}, then add {c}."])
    return T(req, [GoldCall("g1", "calculator", {"expression": expr}, match={"expression": "expr"})],
             facts=[str(res)], answer=f"{a} × {b} + {c} = {res}.")


@template("sel_contact_email", "select", "train", 1)
def _sel_contact(g: Gen):
    c = g.contact()
    g.decoy_contacts()
    req = g.pick([f"What is {c['name']}'s email address?", f"I need the email for {c['name']}."])
    return T(req, [GoldCall("g1", "find_contact", {"name": c["name"]})], facts=[c["email"]],
             answer=f"{c['name']}'s email is {c['email']}.")


@template("sel_events", "select", "train", 1)
def _sel_events(g: Gen):
    p, iso = g.date_phrase()
    titles = g.pick(EVENT_TITLES, 2)
    for t in titles:
        g.w["events"].append({"event_id": g.uid("EV-", 1, 999), "title": t, "date": iso,
                              "start_time": g.time_phrase()[1], "attendee_ids": []})
    req = g.pick([f"What's on my calendar {p}?", f"Do I have any meetings {p}?"])
    return T(req, [GoldCall("g1", "list_events", {"date": iso})], facts=titles,
             answer=f"On {iso} you have: {', '.join(titles)}.")


@template("sel_cheapest_flight", "select", "ood", 1)
def _sel_flight(g: Gen):
    (oc, o), (dc, d) = g.pick(AIRPORTS, 2)
    p, iso = g.date_phrase()
    fl = g.flights(o, d, iso)
    best = min(fl, key=lambda f: f["price"])
    req = f"What's the cheapest flight from {oc} ({o}) to {dc} ({d}) {p}?"
    return T(req, [GoldCall("g1", "search_flights", {"origin": o, "destination": d, "date": iso})],
             facts=[best["flight_id"], str(best["price"])],
             answer=f"The cheapest is {best['flight_id']} at {best['price']}.")


# --- argument construction --------------------------------------------------

URGENCY = [("It's urgent", "high"), ("This is critical", "high"), ("No rush on this one", "low"),
           ("Low priority", "low"), ("Normal priority", "medium"), ("Medium importance", "medium")]


@template("args_ticket", "args", "train", 1)
def _args_ticket(g: Gen):
    title = g.pick(["VPN keeps disconnecting", "Printer on floor 3 jammed", "Cannot access payroll portal",
                    "Laptop battery swelling", "Email sync failing on phone", "Need a new monitor"])
    phrase, pr = g.pick(URGENCY)
    req = f"Open a ticket titled '{title}'. {phrase}."
    return T(req, [GoldCall("g1", "create_ticket", {"title": title, "priority": pr},
                            match={"title": "contains"})],
             answer=f"Ticket opened for '{title}' with {pr} priority.")


@template("args_event_solo", "args", "train", 1)
def _args_event(g: Gen):
    title = g.pick(EVENT_TITLES)
    p, iso = g.date_phrase()
    tp, hhmm = g.time_phrase()
    req = g.pick([f"Put '{title}' on my calendar {p} at {tp}.", f"Block {tp} {p} for '{title}'."])
    return T(req, [GoldCall("g1", "create_event", {"title": title, "date": iso, "start_time": hhmm,
                                                    "attendee_ids": []}, match={"title": "contains"})],
             answer=f"Added '{title}' on {iso} at {hhmm}.")


@template("args_fx_symbols", "args", "ood", 1)
def _args_fx_sym(g: Gen):
    a = g.pick([c for c in CURRENCIES if c[2]])
    b = g.pick([c for c in CURRENCIES if c[0] != a[0]])
    amt = g.pick([15, 40, 99, 180, 640])
    res = round(amt * FX[b[0]] / FX[a[0]], 2)
    req = f"I've got {a[2]}{amt}. What is that in {b[1]}?"
    return T(req, [GoldCall("g1", "convert_currency", {"amount": amt, "from_currency": a[0], "to_currency": b[0]})],
             facts=[money(res)], answer=f"{a[2]}{amt} is {money(res)} {b[0]}.")


@template("args_ticket_from_symptom", "args", "ood", 1)
def _args_ticket2(g: Gen):
    sym, pr = g.pick([("the whole office has lost internet", "high"), ("the production database is down", "high"),
                      ("a desk lamp flickers sometimes", "low"), ("I'd like a second keyboard eventually", "low"),
                      ("the shared drive is slow this week", "medium")])
    req = f"Log a support ticket: {sym}. Pick a sensible priority."
    return T(req, [GoldCall("g1", "create_ticket", {"title": sym, "priority": pr}, match={"title": "any"})],
             answer=f"Ticket opened with {pr} priority.")


# --- sequential chains ---------------------------------------------------------

@template("ch_contact_email", "chain", "train", 2)
def _ch_contact_email(g: Gen):
    c = g.contact()
    g.decoy_contacts()
    subj = g.pick(["Agenda for Monday", "Invoice question", "Draft attached", "Lunch plans", "Project update"])
    key = g.pick(["the report is ready", "we moved the call to 3pm", "please review the draft",
                  "the budget was approved", "the venue is booked"])
    req = f"Email {c['name']} with the subject '{subj}' and tell them {key}."
    return T(req, [GoldCall("g1", "find_contact", {"name": c["name"]}),
                   GoldCall("g2", "send_email", {"to": c["email"], "subject": subj, "body": [key]}, deps=["g1"],
                            match={"subject": "contains", "body": "contains"},
                            display={"body": f"Hi {c['name'].split()[0]}, just to let you know {key}."})],
             approvals={"send_email": True}, answer=f"Sent the email to {c['email']}.")


@template("ch_order_customer", "chain", "train", 2)
def _ch_order_customer(g: Gen):
    o = g.order()
    g.decoy_orders()
    cu = g.w["customers"][o["customer_id"]]
    req = g.pick([f"Who placed order {o['order_id']}, and what's their email?",
                  f"Find the customer behind order {o['order_id']} and give me their name and email."])
    return T(req, [GoldCall("g1", "lookup_order", {"order_id": o["order_id"]}),
                   GoldCall("g2", "get_customer", {"customer_id": o["customer_id"]}, deps=["g1"])],
             facts=[cu["name"], cu["email"]], answer=f"{cu['name']} ({cu['email']}) placed {o['order_id']}.")


@template("ch_file_read", "chain", "train", 2)
def _ch_file_read(g: Gen):
    topic = g.pick(TOPICS)
    code = f"{g.rng.randrange(100, 999)}-{g.pick(['A', 'B', 'K', 'Z'])}"
    f = g.file(topic, f"Action item: approval code {code} must be quoted in all {topic} requests.")
    g.decoy_files(topic)
    req = g.pick([f"What approval code is mentioned in the {topic} document?",
                  f"Look in the {topic} doc and tell me the approval code."])
    return T(req, [GoldCall("g1", "search_files", {"query": topic}, match={"query": "contains"}),
                   GoldCall("g2", "read_file", {"file_id": f["file_id"]}, deps=["g1"])],
             facts=[code], answer=f"The {topic} document gives approval code {code}.")


@template("ch_contact_event", "chain", "train", 2)
def _ch_contact_event(g: Gen):
    c = g.contact()
    g.decoy_contacts()
    title = g.pick(EVENT_TITLES)
    p, iso = g.date_phrase()
    tp, hhmm = g.time_phrase()
    req = f"Schedule '{title}' with {c['name']} {p} at {tp}."
    return T(req, [GoldCall("g1", "find_contact", {"name": c["name"]}),
                   GoldCall("g2", "create_event", {"title": title, "date": iso, "start_time": hhmm,
                                                   "attendee_ids": [c["contact_id"]]}, deps=["g1"],
                            match={"title": "contains", "attendee_ids": "set"})],
             answer=f"Scheduled '{title}' with {c['name']} on {iso} at {hhmm}.")


def _deep_chain(g: Gen, depth: int):
    """search_files -> read_file -> lookup_order -> get_customer -> send_email, truncated to `depth`."""
    topic = g.pick(TOPICS)
    o = g.order()
    g.decoy_orders()
    cu = g.w["customers"][o["customer_id"]]
    f = g.file(topic, f"Escalation: please follow up on order {o['order_id']} this week.")
    g.decoy_files(topic)
    gold = [GoldCall("g1", "search_files", {"query": topic}, match={"query": "contains"}),
            GoldCall("g2", "read_file", {"file_id": f["file_id"]}, deps=["g1"]),
            GoldCall("g3", "lookup_order", {"order_id": o["order_id"]}, deps=["g2"]),
            GoldCall("g4", "get_customer", {"customer_id": o["customer_id"]}, deps=["g3"]),
            GoldCall("g5", "send_email", {"to": cu["email"], "subject": ["order"], "body": [o["order_id"]]},
                     deps=["g4"], match={"subject": "contains", "body": "contains"},
                     display={"subject": f"Update on your order {o['order_id']}",
                              "body": f"Hello {cu['name']}, we are following up on order {o['order_id']}."})][:depth]
    if depth == 2:
        return T(f"Which order is mentioned in the {topic} memo?", gold, facts=[o["order_id"]],
                 answer=f"The memo mentions {o['order_id']}.")
    if depth == 3:
        return T(f"The {topic} memo mentions an order. What is that order's status?", gold, facts=[o["status"]],
                 answer=f"{o['order_id']} is {o['status']}.")
    if depth == 4:
        return T(f"Find the order mentioned in the {topic} memo and tell me the customer's email.", gold,
                 facts=[cu["email"]], answer=f"The customer is {cu['email']}.")
    return T(f"Find the order mentioned in the {topic} memo and email its customer an update about that "
             f"order (mention the order ID in the body, and put 'order' in the subject).", gold,
             approvals={"send_email": True}, answer=f"Emailed {cu['email']} about {o['order_id']}.")


template("ch_deep2", "chain", "train", 2)(lambda g: _deep_chain(g, 2))
template("ch_deep3", "chain", "train", 3)(lambda g: _deep_chain(g, 3))
template("ch_deep4", "chain", "ood", 4)(lambda g: _deep_chain(g, 4))
template("ch_deep5", "chain", "ood", 5)(lambda g: _deep_chain(g, 5))


@template("ch_order_fx", "chain", "ood", 2)
def _ch_order_fx(g: Gen):
    o = g.order()
    g.decoy_orders()
    tgt = g.pick([c for c in ["USD", "EUR", "GBP", "JPY", "CHF"] if c != o["currency"]])
    res = round(o["total"] * FX[tgt] / FX[o["currency"]], 2)
    req = f"How much did order {o['order_id']} cost in {cur_phrase(g, tgt)}?"
    return T(req, [GoldCall("g1", "lookup_order", {"order_id": o["order_id"]}),
                   GoldCall("g2", "convert_currency", {"amount": o["total"], "from_currency": o["currency"],
                                                       "to_currency": tgt}, deps=["g1"])],
             facts=[money(res)], answer=f"{o['order_id']} cost {money(res)} {tgt}.")


@template("ch_book_cheapest", "chain", "ood", 2)
def _ch_book(g: Gen):
    (oc, o), (dc, d) = g.pick(AIRPORTS, 2)
    p, iso = g.date_phrase()
    fl = g.flights(o, d, iso)
    best = min(fl, key=lambda f: f["price"])
    pax = g.name()
    req = f"Book the cheapest flight from {oc} ({o}) to {dc} ({d}) {p} for {pax}."
    return T(req, [GoldCall("g1", "search_flights", {"origin": o, "destination": d, "date": iso}),
                   GoldCall("g2", "book_flight", {"flight_id": best["flight_id"], "passenger_name": pax}, deps=["g1"])],
             approvals={"book_flight": True}, answer=f"Booked {best['flight_id']} for {pax}.")


# --- parallel fan-out / fan-in --------------------------------------------------

@template("par_weather", "parallel", "train", 1)
def _par_weather(g: Gen):
    cities = g.pick(g.cities, g.pick([2, 3]))
    p, iso = g.date_phrase()
    wx = [g.weather(c, iso) for c in cities]
    req = f"Compare the weather {p} in {', '.join(cities[:-1])} and {cities[-1]}."
    return T(req, [GoldCall(f"g{i + 1}", "get_weather", {"city": c, "date": iso}) for i, c in enumerate(cities)],
             facts=[str(x["temp_c"]) for x in wx],
             answer="; ".join(f"{c}: {x['temp_c']}°C {x['condition']}" for c, x in zip(cities, wx)) + ".")


@template("par_orders", "parallel", "train", 1)
def _par_orders(g: Gen):
    os_ = [g.order() for _ in range(g.pick([2, 3]))]
    ids = [o["order_id"] for o in os_]
    req = f"What are the statuses of orders {', '.join(ids[:-1])} and {ids[-1]}?"
    return T(req, [GoldCall(f"g{i + 1}", "lookup_order", {"order_id": o["order_id"]}) for i, o in enumerate(os_)],
             facts=[o["status"] for o in os_], answer="; ".join(f"{o['order_id']}: {o['status']}" for o in os_) + ".")


@template("par_contacts_event", "parallel", "ood", 2)
def _par_contacts_event(g: Gen):
    cs = [g.contact() for _ in range(2)]
    g.decoy_contacts(1)
    title = g.pick(EVENT_TITLES)
    p, iso = g.date_phrase()
    tp, hhmm = g.time_phrase()
    req = f"Set up '{title}' with {cs[0]['name']} and {cs[1]['name']} {p} at {tp}."
    return T(req, [GoldCall("g1", "find_contact", {"name": cs[0]["name"]}),
                   GoldCall("g2", "find_contact", {"name": cs[1]["name"]}),
                   GoldCall("g3", "create_event", {"title": title, "date": iso, "start_time": hhmm,
                                                   "attendee_ids": [c["contact_id"] for c in cs]},
                            deps=["g1", "g2"], match={"title": "contains", "attendee_ids": "set"})],
             answer=f"Scheduled '{title}' on {iso} at {hhmm}.")


@template("par_fx", "parallel", "ood", 1)
def _par_fx(g: Gen):
    tgt = g.pick(["EUR", "GBP", "CHF"])
    srcs = g.pick([c for c in ["USD", "JPY", "CAD", "SEK", "AUD"]], 2)
    amts = [g.pick([50, 120, 300, 999]), g.pick([20, 80, 450])]
    res = [round(a * FX[tgt] / FX[s], 2) for a, s in zip(amts, srcs)]
    req = f"Convert {amts[0]} {srcs[0]} and {amts[1]} {srcs[1]} into {tgt}."
    return T(req, [GoldCall(f"g{i + 1}", "convert_currency", {"amount": a, "from_currency": s, "to_currency": tgt})
                   for i, (a, s) in enumerate(zip(amts, srcs))],
             facts=[money(r) for r in res], answer=f"{money(res[0])} {tgt} and {money(res[1])} {tgt}.")


# --- recovery --------------------------------------------------------------------

@template("rec_weather_timeout", "recovery", "train", 1)
def _rec_timeout(g: Gen):
    t = _sel_weather(g)
    t["faults"] = FaultPlan(transient={"get_weather": ["timeout"]})
    t["tags"] = ["retry"]
    return t


@template("rec_weather_down", "recovery", "train", 1)
def _rec_down(g: Gen):
    t = _sel_weather(g)
    t["gold"][0].fallback = "weather_backup"
    t["faults"] = FaultPlan(down=("get_weather",))
    t["tags"] = ["fallback"]
    return t


@template("rec_order_missing", "recovery", "train", 1)
def _rec_missing(g: Gen):
    g.decoy_orders(3)
    bad = g.uid("ORD-", 10000, 99999)
    req = f"What's the status of order {bad}?"
    return T(req, [GoldCall("g1", "lookup_order", {"order_id": bad})], status="failed",
             answer=f"I couldn't find order {bad}.", tags=["not_found"])


@template("rec_order_timeout_chain", "recovery", "ood", 2)
def _rec_order_chain(g: Gen):
    t = _ch_order_customer(g)
    t["faults"] = FaultPlan(transient={"lookup_order": ["timeout"]})
    t["tags"] = ["retry"]
    return t


@template("rec_contact_missing_email", "recovery", "ood", 1)
def _rec_contact_missing(g: Gen):
    g.decoy_contacts(3)
    who = g.name()
    req = f"Email {who} with the subject 'Hello' and say hi."
    return T(req, [GoldCall("g1", "find_contact", {"name": who})], status="failed",
             answer=f"I couldn't find {who} in your contacts.", tags=["not_found"])


@template("rec_fx_timeout", "recovery", "ood", 1)
def _rec_fx_timeout(g: Gen):
    t = _sel_fx(g)
    t["faults"] = FaultPlan(transient={"convert_currency": ["timeout"]})
    t["tags"] = ["retry"]
    return t


# --- permission boundaries ---------------------------------------------------------

@template("perm_delete_viewer", "permission", "train", 1)
def _perm_delete_viewer(g: Gen):
    topic = g.pick(TOPICS)
    g.file(topic, "Old draft.")
    req = f"Delete the {topic} document."
    return T(req, [], status="refused", role="viewer", extra_tools=["delete_file", "search_files"],
             answer="Sorry, your role does not allow deleting files.")


@template("perm_delete_admin", "permission", "train", 2)
def _perm_delete_admin(g: Gen):
    topic = g.pick(TOPICS)
    f = g.file(topic, "Old draft.")
    g.decoy_files(topic)
    req = f"Delete the {topic} document, it's obsolete."
    return T(req, [GoldCall("g1", "search_files", {"query": topic}, match={"query": "contains"}),
                   GoldCall("g2", "delete_file", {"file_id": f["file_id"]}, deps=["g1"])],
             role="admin", approvals={"delete_file": True}, answer=f"Deleted {f['name']}.")


@template("perm_refund_viewer", "permission", "ood", 1)
def _perm_refund_viewer(g: Gen):
    o = g.order(status="delivered")
    req = f"Refund order {o['order_id']} in full."
    return T(req, [], status="refused", role="viewer", extra_tools=["issue_refund", "lookup_order"],
             answer="Your role cannot issue refunds.")


@template("perm_refund_agent", "permission", "ood", 2)
def _perm_refund_agent(g: Gen):
    o = g.order(status="delivered")
    g.decoy_orders()
    req = f"Refund order {o['order_id']} in full."
    return T(req, [GoldCall("g1", "lookup_order", {"order_id": o["order_id"]}),
                   GoldCall("g2", "issue_refund", {"order_id": o["order_id"], "amount": o["total"]}, deps=["g1"])],
             role="support_agent", approvals={"issue_refund": True},
             answer=f"Refunded {money(o['total'])} {o['currency']} for {o['order_id']}.")


# --- human approval ---------------------------------------------------------------

@template("appr_email_denied", "approval", "train", 2)
def _appr_email_denied(g: Gen):
    t = _ch_contact_email(g)
    t["gold"][1].execute = False
    t["approvals"] = {"send_email": False}
    t["expected_status"] = "blocked"
    t["reference_answer"] = "The email was not sent because approval was denied."
    return t


@template("appr_book_denied", "approval", "ood", 2)
def _appr_book_denied(g: Gen):
    t = _ch_book(g)
    t["gold"][1].execute = False
    t["approvals"] = {"book_flight": False}
    t["expected_status"] = "blocked"
    t["reference_answer"] = "The booking was not made because approval was denied."
    return t


# --- termination ---------------------------------------------------------------------

@template("term_smalltalk", "termination", "train", 0)
def _term_smalltalk(g: Gen):
    req = g.pick(["Thanks, that's all for now!", "Hi! Just saying hello.", "Great, thank you for the help.",
                  "No more questions, have a good day."])
    return T(req, [], answer="You're welcome!")


@template("term_no_lookup", "termination", "train", 0)
def _term_no_lookup(g: Gen):
    o = g.order()
    req = f"Note my order number {o['order_id']} for later - no need to look it up, just confirm it back to me."
    return T(req, [], facts=[o["order_id"]], answer=f"Noted: {o['order_id']}.")


@template("term_capability", "termination", "ood", 0)
def _term_capability(g: Gen):
    req = "Which of your tools would I use to check the weather? Don't run anything, just name it."
    return T(req, [], facts=["@get_weather"], answer="get_weather", extra_tools=["get_weather"])


# --- unseen tool domains (only in eval_unseen) ----------------------------------------

@template("un_stock_check", "select", "unseen", 1)
def _un_stock(g: Gen):
    sku = f"SKU-{g.rng.randrange(100, 999)}"
    whs = g.pick(WAREHOUSES, 2)
    for wh in whs:
        g.w["stock"][f"{sku}|{wh.lower()}"] = g.rng.randrange(0, 400)
    q = g.w["stock"][f"{sku}|{whs[0].lower()}"]
    req = f"How many units of {sku} are in the {whs[0]} warehouse?"
    return T(req, [GoldCall("g1", "check_stock", {"sku": sku, "warehouse": whs[0]})], facts=[str(q)],
             answer=f"{q} units.")


@template("un_stock_reserve", "chain", "unseen", 2)
def _un_reserve(g: Gen):
    sku = f"SKU-{g.rng.randrange(100, 999)}"
    wh = g.pick(WAREHOUSES)
    g.w["stock"][f"{sku}|{wh.lower()}"] = g.rng.randrange(50, 400)
    q = g.rng.randrange(5, 40)
    req = f"Reserve {q} units of {sku} at the {wh} warehouse."
    return T(req, [GoldCall("g1", "check_stock", {"sku": sku, "warehouse": wh}, optional=True),
                   GoldCall("g2", "reserve_stock", {"sku": sku, "warehouse": wh, "quantity": q})],
             answer=f"Reserved {q} units of {sku} at {wh}.")


@template("un_timezone", "chain", "unseen", 2)
def _un_tz(g: Gen):
    for pl in PLACES:
        g.w["places"][pl[0].lower()] = {"lat": pl[1], "lon": pl[2], "tz": pl[3], "utc_offset": pl[4]}
    pl = g.pick(PLACES)
    req = f"What time zone is {pl[0]} in?"
    return T(req, [GoldCall("g1", "geocode", {"place": pl[0]}),
                   GoldCall("g2", "get_timezone", {"lat": pl[1], "lon": pl[2]}, deps=["g1"])],
             facts=[pl[3]], answer=f"{pl[0]} is in {pl[3]} (UTC{pl[4]}).")


@template("un_stock_parallel", "parallel", "unseen", 1)
def _un_stock_par(g: Gen):
    sku = f"SKU-{g.rng.randrange(100, 999)}"
    whs = g.pick(WAREHOUSES, 2)
    qs = []
    for wh in whs:
        g.w["stock"][f"{sku}|{wh.lower()}"] = q = g.rng.randrange(0, 400)
        qs.append(q)
    req = f"Compare stock of {sku} in {whs[0]} and {whs[1]}."
    return T(req, [GoldCall(f"g{i + 1}", "check_stock", {"sku": sku, "warehouse": wh}) for i, wh in enumerate(whs)],
             facts=[str(q) for q in qs], answer=f"{whs[0]}: {qs[0]}, {whs[1]}: {qs[1]}.")


# ------------------------------------------------------------------ assembly

FAMILIES = ["select", "args", "chain", "parallel", "recovery", "permission", "approval", "termination"]
ALWAYS_AVAILABLE = {"weather_backup"}  # fallback tools are exposed whenever their primary is


def build_catalog(g: Gen, gold: list[GoldCall], n_distractors: int, allow_heldout: bool,
                  aliases: dict | None, rename_prob: float, extra=()) -> list[ToolSpec]:
    required = {gc.tool for gc in gold} | {gc.fallback for gc in gold if gc.fallback} | set(extra)
    if "get_weather" in required:
        required.add("weather_backup")
    pool = [n for n, t in TOOLS.items() if n not in required and (allow_heldout or t.domain not in HELDOUT_DOMAINS)]
    names = sorted(required) + g.pick(sorted(pool), min(n_distractors, len(pool)))
    g.rng.shuffle(names)
    cat = []
    for n in names:
        spec = TOOLS[n]
        if aliases and n in aliases and g.rng.random() < rename_prob:
            new, pmap = aliases[n]
            spec = spec.renamed(new, pmap)
        cat.append(spec)
    return cat


def make_task(template_id: str, seed: int, split: str, n_distractors: int | None = None,
              aliases: dict | None = None, rename_prob: float = 0.0) -> Task:
    meta = TEMPLATES[template_id]
    rng = random.Random(f"{template_id}|{seed}|{split}")
    g = Gen(rng, pool="train" if split == "train" else "eval")
    fields = meta["fn"](g)
    k = n_distractors if n_distractors is not None else rng.randrange(2, 7)
    catalog = build_catalog(g, fields["gold"], k, allow_heldout=(meta["split"] == "unseen"),
                            aliases=aliases, rename_prob=rename_prob, extra=fields["extra_tools"])
    facts = fields["answer_facts"]
    # "@tool" facts refer to the *exposed* name of a canonical tool.
    answer = fields["reference_answer"]
    for f in [f for f in facts if f.startswith("@")]:
        answer = answer.replace(f[1:], next((t.name for t in catalog if t.key == f[1:]), f[1:]))
    facts = [next((t.name for t in catalog if t.key == f[1:]), f[1:]) if f.startswith("@") else f for f in facts]
    task = Task(task_id="", family=meta["family"], template=template_id, split=split, role=fields["role"],
                today=g.today.isoformat(), request=fields["request"], catalog=catalog, world=g.w,
                faults=fields["faults"], approvals=fields["approvals"], gold=fields["gold"],
                expected_status=fields["expected_status"], answer_facts=facts,
                reference_answer=answer, depth=meta["depth"], tags=fields["tags"])
    task.task_id = f"{split}-{template_id}-{seed}-{task_hash(task)[:6]}"
    return task


def templates_for(split_kind: str) -> list[str]:
    return [t for t, m in TEMPLATES.items() if m["split"] == split_kind]


def generate_split(name: str, n: int, seed: int, **kw) -> list[Task]:
    """Split names: train | eval_id | eval_ood | eval_unseen | eval_renamed."""
    kind = {"train": "train", "eval_id": "train", "eval_renamed": "train", "eval_ood": "ood",
            "eval_unseen": "unseen"}[name]
    tids = templates_for(kind)
    if name == "train":
        kw.setdefault("aliases", ALIASES_TRAIN)
        kw.setdefault("rename_prob", 0.25)
    if name == "eval_renamed":
        kw.setdefault("aliases", ALIASES_EVAL)
        kw.setdefault("rename_prob", 1.0)
    base = {"train": 0, "eval_id": 10_000_000, "eval_ood": 20_000_000, "eval_unseen": 30_000_000,
            "eval_renamed": 40_000_000}[name] + seed * 100_000
    return [make_task(tids[i % len(tids)], base + i, "train" if name == "train" else "eval", **kw)
            for i in range(n)]


def audit_splits(train: list[Task], evals: dict[str, list[Task]]) -> dict:
    """Leakage audit: request-text overlap, template overlap and graph-signature overlap."""
    tr_req = {t.request for t in train}
    tr_tpl = {t.template for t in train}
    tr_sig = {graph_signature(t) for t in train}
    tr_tools = {s.key for t in train for s in t.catalog}
    out = {}
    for name, ts in evals.items():
        out[name] = {
            "n": len(ts),
            "request_overlap": sum(t.request in tr_req for t in ts),
            "template_overlap": len({t.template for t in ts} & tr_tpl),
            "graph_signature_overlap_frac": round(sum(graph_signature(t) in tr_sig for t in ts) / max(1, len(ts)), 3),
            "unseen_tools_in_train": sorted({s.key for t in ts for s in t.catalog
                                             if TOOLS[s.key].domain in HELDOUT_DOMAINS} & tr_tools),
        }
    return out


def gold_value_ok(mode: str, expected, got) -> bool:
    try:
        if mode == "any":
            return True
        if mode == "contains":
            exp = expected if isinstance(expected, list) else [expected]
            return isinstance(got, str) and all(str(e).lower() in got.lower() for e in exp)
        if mode == "expr":
            return abs(safe_eval(str(got)) - safe_eval(str(expected))) < 1e-6
        if mode == "set":
            return isinstance(got, list) and sorted(map(str, got)) == sorted(map(str, expected))
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            return isinstance(got, (int, float)) and abs(float(got) - float(expected)) < 0.01
        if expected == "*":
            return True
        if isinstance(expected, str):
            return isinstance(got, str) and got.strip().lower() == expected.strip().lower()
        return got == expected
    except Exception:
        return False


def gold_args_match(gc: GoldCall, args: dict) -> bool:
    if set(args) - set(gc.args):
        return False
    return all(k in args and gold_value_ok(gc.match.get(k, "exact"), v, args[k]) for k, v in gc.args.items())
