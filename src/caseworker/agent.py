"""The case agent, a LangGraph state machine with tool calling, memory and a human approval step.

    intake -> triage -> research <-> tools -> collect -> draft -> check -> (problems?) -> draft
                                                                       -> review (pauses for a person)
                                                                       -> file ticket in Case Router -> done
"""

from __future__ import annotations

import json
import operator
import re
import time
import uuid
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt

from .services import History, Rules, Tickets, Triage, company_key

RESEARCH_PROMPT = """You are a caseworker's assistant for consumer financial complaints. Before anything is written, gather facts with your tools:
1. Call find_rules with a short search query in regulatory terms (for example "debt collector communication at inconvenient times" or "error resolution unauthorized electronic fund transfer"). You may search twice if the first results do not fit.
2. Call company_history once with the company name.
Then reply with one plain paragraph describing the consumer's problem. Do not give legal advice."""

DRAFT_PROMPT = """Write the case file for a consumer complaint. Use only the rule passages, company history and notes provided.
- summary: two sentences on what happened and what the consumer wants.
- priority: HIGH if the consumer risks losing a home, a car, access to their money, or faces identity theft; MEDIUM otherwise; LOW for questions.
- cited_sections: the section numbers (like 1006.6) from the passages that apply. Cite only sections shown in the passages.
- letter: an acknowledgment letter to the consumer, under 200 words, that includes the case number {case_id}, says the complaint was received and is being reviewed, and names the rules that apply in plain words. Do not promise any outcome, refund or timeline beyond what a cited rule requires. Do not give legal advice.
- next_steps: up to three concrete actions for the caseworker."""

DRAFT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "priority": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
        "cited_sections": {"type": "array", "items": {"type": "string"}},
        "letter": {"type": "string"},
        "next_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "priority", "cited_sections", "letter", "next_steps"],
}

_MASKS = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("phone", re.compile(r"(?<!\d)\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)")),
    ("account number", re.compile(r"\b(?:\d[ -]?){8,19}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
]
_SECTION = re.compile(r"(\d{4}\.\d+[a-z]?)")
_PROMISES = re.compile(r"\b(guarantee[sd]?|we will refund|you will (?:receive|get) (?:a|your) refund|you are entitled)\b",
                       re.I)


def redact(text: str) -> tuple[str, list[str]]:
    found = []
    for kind, pattern in _MASKS:
        text, count = pattern.subn(f"[{kind}]", text)
        if count:
            found.append(f"{count} {kind}{'s' if count > 1 else ''}")
    return text, found


def normalize_section(value: str) -> str | None:
    match = _SECTION.search(str(value))
    return match.group(1) if match else None


def check_draft(draft: dict, passages: list[dict], case_id: str) -> dict:
    """Rule-based checks on the draft. The model's own judgment is not trusted for these."""
    available = {p.get("section") for p in passages}
    cited = [s for s in (normalize_section(v) for v in draft.get("cited_sections", [])) if s]
    invented = sorted({s for s in cited if s not in available})
    letter = str(draft.get("letter", ""))
    problems = []
    if not cited:
        problems.append("The draft cites no rule.")
    if invented:
        problems.append(f"The draft cites sections that were not in the retrieved passages: {', '.join(invented)}.")
    if case_id not in letter:
        problems.append(f"The letter must include the case number {case_id}.")
    if re.search(r"\d{8,}", re.sub(r"[ -]", "", letter.replace(case_id, ""))):
        problems.append("The letter contains a long number; account numbers must stay masked.")
    words = len(letter.split())
    if words > 250:
        problems.append(f"The letter is {words} words; keep it under 200.")
    if _PROMISES.search(letter):
        problems.append("The letter promises an outcome.")
    return {"problems": problems, "cited": sorted(set(cited)), "invented": invented, "words": words,
            "passed": not problems}


class CaseState(TypedDict, total=False):
    case_id: str
    company: str
    complaint: str
    redactions: list[str]
    notes: list[dict]
    triage: dict
    messages: Annotated[list[AnyMessage], add_messages]
    passages: list[dict]
    history: dict
    tool_use: dict
    draft: dict
    checks: dict
    first_checks: dict
    repairs: int
    retry: bool
    review: dict
    ticket: dict
    status: str
    trace: Annotated[list[dict], operator.add]
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]


def _step(name: str, started: float, detail: str = "") -> list[dict]:
    return [{"step": name, "ms": round((time.perf_counter() - started) * 1000, 1), "detail": detail}]


def _usage(message: AIMessage) -> dict:
    usage = getattr(message, "usage_metadata", None) or {}
    return {"input_tokens": int(usage.get("input_tokens", 0)), "output_tokens": int(usage.get("output_tokens", 0))}


class CaseAgent:
    def __init__(self, research_llm: BaseChatModel, draft_llm: BaseChatModel, triage: Triage, rules: Rules,
                 tickets: Tickets, history: History, max_tool_rounds: int = 3, max_repairs: int = 1,
                 checkpointer=None, store=None) -> None:
        self.triage_service = triage
        self.rules = rules
        self.tickets = tickets
        self.history = history
        self.max_tool_rounds = max_tool_rounds
        self.max_repairs = max_repairs
        self.store = store if store is not None else InMemoryStore()
        self.tools = self._make_tools()
        self.research_llm = research_llm.bind_tools(self.tools)
        self.draft_llm = draft_llm
        self.graph = self._build().compile(checkpointer=checkpointer or InMemorySaver(), store=self.store)

    # -- tools the model can call --------------------------------------------------------------------------

    def _make_tools(self):
        rules, history = self.rules, self.history

        @tool
        def find_rules(query: str) -> str:
            """Search the federal consumer finance rules (Regulations B, E, F, V, X and Z) and return the most
            relevant passages with their citations. Use regulatory terms in the query."""
            passages = rules.search(query, k=4)
            return json.dumps([{"citation": p["citation"], "section": p["section"], "heading": p.get("heading", ""),
                                "text": p["text"][:900]} for p in passages])

        @tool
        def company_history(company: str) -> str:
            """Look up how many complaints a company has received, its most common issues, and how it usually
            responds."""
            return json.dumps(history.summary(company))

        return [find_rules, company_history]

    # -- graph ---------------------------------------------------------------------------------------------

    def _build(self) -> StateGraph:
        graph = StateGraph(CaseState)
        for name in ("intake", "triage", "research", "collect", "draft", "check", "review", "file"):
            graph.add_node(name, getattr(self, f"_{name}"))
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_edge(START, "intake")
        graph.add_edge("intake", "triage")
        graph.add_edge("triage", "research")
        graph.add_conditional_edges("research", self._after_research, {"tools": "tools", "collect": "collect"})
        graph.add_edge("tools", "research")
        graph.add_edge("collect", "draft")
        graph.add_edge("draft", "check")
        graph.add_conditional_edges("check", self._after_check, {"draft": "draft", "review": "review"})
        graph.add_conditional_edges("review", lambda s: "file" if s["review"].get("approved") else END,
                                    {"file": "file", END: END})
        graph.add_edge("file", END)
        return graph

    def _intake(self, state: CaseState) -> dict:
        started = time.perf_counter()
        text, found = redact(state["complaint"])
        notes = [item.value for item in self.store.search(("companies", company_key(state.get("company", ""))),
                                                          limit=3)]
        return {"complaint": text, "redactions": found, "notes": notes, "repairs": 0, "status": "working",
                "trace": _step("intake", started, f"masked {', '.join(found) or 'nothing'}; {len(notes)} earlier notes")}

    def _triage(self, state: CaseState) -> dict:
        started = time.perf_counter()
        try:
            result = self.triage_service.triage(state["complaint"])
        except Exception as error:  # a down triage service sends the case to a person instead of failing it
            result = {"product": None, "route": "person", "routed_to": None, "confidence": 0.0,
                      "reason": f"Triage unavailable: {error}"}
        detail = f"{result.get('product')} at {result.get('confidence', 0):.0%}, route {result.get('route')}"
        return {"triage": result, "trace": _step("triage", started, detail)}

    def _research(self, state: CaseState) -> dict:
        started = time.perf_counter()
        messages = list(state.get("messages") or [])
        new: list[AnyMessage] = []
        if not messages:
            triage = state["triage"]
            notes = "\n".join(f"- {n.get('summary')} (rules {', '.join(n.get('rules', []))})" for n in state["notes"])
            new = [SystemMessage(RESEARCH_PROMPT), HumanMessage(
                f"Company: {state.get('company') or 'unknown'}\n"
                f"Triage: {triage.get('product')} ({triage.get('description', '')}), "
                f"confidence {triage.get('confidence', 0):.0%}\n"
                + (f"Earlier cases with this company:\n{notes}\n" if notes else "")
                + f"\nComplaint:\n{state['complaint']}")]
            messages += new
        reply = self.research_llm.invoke(messages)
        calls = ", ".join(c["name"] for c in reply.tool_calls) or "no tool call"
        return {"messages": new + [reply], **_usage(reply), "trace": _step("research", started, calls)}

    def _after_research(self, state: CaseState) -> str:
        last = state["messages"][-1]
        rounds = sum(1 for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls)
        return "tools" if isinstance(last, AIMessage) and last.tool_calls and rounds <= self.max_tool_rounds \
            else "collect"

    def _collect(self, state: CaseState) -> dict:
        started = time.perf_counter()
        passages: dict[str, dict] = {}
        history = None
        calls = [c for m in state["messages"] if isinstance(m, AIMessage) for c in m.tool_calls]
        for message in state["messages"]:
            if not isinstance(message, ToolMessage):
                continue
            try:
                payload = json.loads(message.content)
            except (TypeError, ValueError):
                continue
            if message.name == "find_rules" and isinstance(payload, list):
                for passage in payload:
                    passages.setdefault(passage["citation"], passage)
            elif message.name == "company_history" and isinstance(payload, dict):
                history = payload
        model_searched = bool(passages)
        if not passages:
            query = f"{state['triage'].get('description', '')} {state['complaint'][:300]}"
            for passage in self.rules.search(query, k=4):
                passages.setdefault(passage["citation"], {k: passage[k] for k in ("citation", "section", "text")}
                                    | {"heading": passage.get("heading", "")})
        if history is None:
            history = self.history.summary(state.get("company", ""))
        tool_use = {
            "calls": len(calls),
            "find_rules_calls": sum(c["name"] == "find_rules" for c in calls),
            "company_history_calls": sum(c["name"] == "company_history" for c in calls),
            "repeated_calls": len(calls) - len({json.dumps([c["name"], c["args"]], sort_keys=True) for c in calls}),
            "model_found_rules": model_searched,
            "fallback_search": not model_searched,
        }
        return {"passages": list(passages.values()), "history": history, "tool_use": tool_use,
                "trace": _step("collect", started, f"{len(passages)} passages, {len(calls)} tool calls")}

    def _draft(self, state: CaseState) -> dict:
        started = time.perf_counter()
        passages = "\n\n".join(f"[{i}] {p['citation']} ({p.get('heading', '')})\n{p['text']}"
                               for i, p in enumerate(state["passages"], start=1))
        notes = "\n".join(f"- {n.get('summary')}" for n in state.get("notes", []))
        messages = [
            SystemMessage(DRAFT_PROMPT.format(case_id=state["case_id"])),
            HumanMessage(f"Case number: {state['case_id']}\nCompany: {state.get('company') or 'unknown'}\n"
                         f"Product: {state['triage'].get('product')}\n"
                         f"Company history: {json.dumps(state['history'])}\n"
                         + (f"Earlier cases:\n{notes}\n" if notes else "")
                         + f"\nComplaint:\n{state['complaint']}\n\nRule passages:\n{passages}"),
        ]
        if state.get("repairs", 0) > 0 and state.get("draft"):
            messages += [AIMessage(json.dumps(state["draft"])),
                         HumanMessage("Fix these problems and return the whole case file again:\n- "
                                      + "\n- ".join(state["checks"]["problems"]))]
        reply = self.draft_llm.invoke(messages)
        try:
            draft = json.loads(reply.content)
            if not isinstance(draft, dict):
                raise ValueError("not an object")
        except (TypeError, ValueError):
            draft = {"summary": str(reply.content)[:300], "priority": "MEDIUM", "cited_sections": [], "letter": "",
                     "next_steps": [], "unreadable": True}
        draft = {**{"summary": "", "priority": "MEDIUM", "cited_sections": [], "letter": "", "next_steps": []}, **draft}
        if draft["priority"] not in ("LOW", "MEDIUM", "HIGH"):
            draft["priority"] = "MEDIUM"
        return {"draft": draft, **_usage(reply), "trace": _step("draft", started, f"{draft['priority']} priority")}

    def _check(self, state: CaseState) -> dict:
        started = time.perf_counter()
        checks = check_draft(state["draft"], state["passages"], state["case_id"])
        retry = bool(checks["problems"]) and state.get("repairs", 0) < self.max_repairs
        update: dict = {"checks": checks, "retry": retry,
                        "trace": _step("check", started, "; ".join(checks["problems"]) or "passed")}
        if "first_checks" not in state:
            update["first_checks"] = checks
        if retry:
            update["repairs"] = state.get("repairs", 0) + 1
        return update

    def _after_check(self, state: CaseState) -> str:
        return "draft" if state.get("retry") else "review"

    def _review(self, state: CaseState) -> dict:
        decision = interrupt({"case_id": state["case_id"], "draft": state["draft"], "checks": state["checks"]})
        approved = bool(decision.get("approved"))
        return {"review": decision, "status": "approved" if approved else "returned",
                "trace": [{"step": "review", "ms": 0.0, "detail": f"{'approved' if approved else 'returned'} by "
                                                                   f"{decision.get('reviewer') or 'a reviewer'}"}]}

    def _file(self, state: CaseState) -> dict:
        started = time.perf_counter()
        draft = state["draft"]
        letter = state["review"].get("letter") or draft["letter"]
        product = state["triage"].get("routed_to") or state["triage"].get("product") or "unsorted"
        cited = state["checks"]["cited"]
        description = (f"Case {state['case_id']} | Company: {state.get('company') or 'unknown'}\n"
                       f"Rules: {', '.join(cited) or 'none'}\n\nSummary: {draft['summary']}\n\n"
                       f"Next steps:\n" + "\n".join(f"- {s}" for s in draft["next_steps"]) + f"\n\nLetter:\n{letter}")
        ticket = self.tickets.open_case(f"[product:{product}] {draft['summary'][:150]}", description,
                                        draft["priority"])
        self.store.put(("companies", company_key(state.get("company", ""))), state["case_id"],
                       {"case_id": state["case_id"], "product": product, "summary": draft["summary"][:300],
                        "rules": cited, "queue": ticket.get("queue")})
        return {"ticket": ticket, "status": "filed",
                "trace": _step("file", started, f"ticket {ticket.get('id')} in {ticket.get('queue')}")}

    # -- public API ----------------------------------------------------------------------------------------

    @staticmethod
    def _config(case_id: str) -> dict:
        return {"configurable": {"thread_id": case_id}, "recursion_limit": 40}

    def start(self, complaint: str, company: str = "", case_id: str | None = None) -> dict:
        case_id = case_id or f"CW-{uuid.uuid4().hex[:8].upper()}"
        self.graph.invoke({"case_id": case_id, "company": company, "complaint": complaint, "trace": [],
                           "input_tokens": 0, "output_tokens": 0}, self._config(case_id))
        return self.get(case_id)

    def review(self, case_id: str, approved: bool, letter: str | None = None, reviewer: str = "") -> dict:
        decision: dict[str, Any] = {"approved": approved, "reviewer": reviewer}
        if letter:
            decision["letter"] = letter
        self.graph.invoke(Command(resume=decision), self._config(case_id))
        return self.get(case_id)

    def get(self, case_id: str) -> dict:
        snapshot = self.graph.get_state(self._config(case_id))
        values = snapshot.values
        if not values:
            raise KeyError(case_id)
        status = "awaiting review" if snapshot.next and "review" in snapshot.next else values.get("status")
        return {
            "case_id": case_id,
            "status": status,
            "company": values.get("company"),
            "complaint": values.get("complaint"),
            "redactions": values.get("redactions", []),
            "triage": values.get("triage"),
            "passages": values.get("passages", []),
            "history": values.get("history"),
            "notes": values.get("notes", []),
            "tool_use": values.get("tool_use"),
            "draft": values.get("draft"),
            "checks": values.get("checks"),
            "first_checks": values.get("first_checks"),
            "repairs": values.get("repairs", 0),
            "review": values.get("review"),
            "ticket": values.get("ticket"),
            "usage": {"input_tokens": values.get("input_tokens", 0), "output_tokens": values.get("output_tokens", 0)},
            "trace": values.get("trace", []),
        }
