from __future__ import annotations

import json
from collections.abc import Callable

import pandas as pd
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from caseworker.agent import CaseAgent
from caseworker.services import History

PASSAGES = {
    "1006.6": {"citation": "12 CFR 1006.6(b)(1)(i)", "section": "1006.6", "heading": "Communications",
               "text": "A debt collector must not communicate with a consumer at any unusual time or place."},
    "1006.34": {"citation": "12 CFR 1006.34(a)", "section": "1006.34", "heading": "Validation of debts",
                "text": "A debt collector must provide a consumer with the validation information."},
    "1005.11": {"citation": "12 CFR 1005.11(c)(1)", "section": "1005.11", "heading": "Procedures for resolving errors",
                "text": "A financial institution shall investigate promptly and determine whether an error occurred."},
}


class ScriptedChat(BaseChatModel):
    """A chat model that replays scripted replies (or functions of the messages) and accepts bound tools."""

    replies: list = Field(default_factory=list)
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(messages)
        reply = self.replies.pop(0) if self.replies else AIMessage(content="The consumer wants help.")
        if callable(reply):
            reply = reply(messages)
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name: str, args: dict, n: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{n}", "type": "tool_call"}],
                     usage_metadata={"input_tokens": 300, "output_tokens": 20, "total_tokens": 320})


def draft_reply(sections: list[str], letter_extra: str = "") -> Callable:
    def reply(messages) -> AIMessage:
        case_id = next(line.split(": ", 1)[1] for line in messages[1].content.splitlines()
                       if line.startswith("Case number"))
        body = {"summary": "A collector calls before 8 a.m. and never sent a validation notice.", "priority": "MEDIUM",
                "cited_sections": sections, "next_steps": ["Ask the company for its call logs"],
                "letter": f"Dear consumer, we received your complaint, case {case_id}. We are reviewing it under the "
                          f"debt collection rules on call times and validation notices.{letter_extra}"}
        return AIMessage(content=json.dumps(body), usage_metadata={"input_tokens": 900, "output_tokens": 150,
                                                                   "total_tokens": 1050})
    return reply


class FakeTriage:
    def __init__(self, product: str = "debt_collection", confidence: float = 0.93) -> None:
        self.product, self.confidence = product, confidence

    def triage(self, text: str) -> dict:
        return {"product": self.product, "description": "A debt collector trying to collect a debt",
                "confidence": self.confidence, "route": "automatic", "routed_to": self.product}


class FakeRules:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, k: int = 4) -> list[dict]:
        self.queries.append(query)
        wanted = ["1005.11"] if "error" in query.lower() else ["1006.6", "1006.34"]
        return [dict(PASSAGES[s]) for s in wanted][:k]


class FakeTickets:
    def __init__(self) -> None:
        self.opened: list[dict] = []

    def open_case(self, subject: str, description: str, priority: str) -> dict:
        self.opened.append({"subject": subject, "description": description, "priority": priority})
        return {"id": len(self.opened), "queue": "Debt Collection", "respond_by": "2026-11-24T18:00:00Z"}


@pytest.fixture
def history(tmp_path) -> History:
    store = History(tmp_path / "history.db")
    store.load(pd.DataFrame({
        "Company": ["Acme Collections"] * 3 + ["Other Bank"],
        "Product": ["Debt collection"] * 3 + ["Checking or savings account"],
        "Issue": ["Attempts to collect debt not owed", "Communication tactics", "Communication tactics", "Fees"],
        "Company response to consumer": ["Closed with explanation"] * 3 + ["Closed with monetary relief"],
        "Timely response?": ["Yes", "Yes", "No", "Yes"],
    }))
    return store


def make_agent(history, research_replies=None, draft_replies=None, tickets=None, rules=None, triage=None):
    research = ScriptedChat(replies=list(research_replies if research_replies is not None else [
        tool_call("find_rules", {"query": "debt collector communication at inconvenient times"}, 1),
        tool_call("company_history", {"company": "Acme Collections"}, 2),
        AIMessage(content="The consumer is being called early and never got a validation notice.",
                  usage_metadata={"input_tokens": 500, "output_tokens": 30, "total_tokens": 530}),
    ]))
    drafter = ScriptedChat(replies=list(draft_replies if draft_replies is not None else [draft_reply(["1006.6"])]))
    agent = CaseAgent(research, drafter, triage or FakeTriage(), rules or FakeRules(), tickets or FakeTickets(),
                      history)
    return agent, research, drafter


COMPLAINT = ("A debt collector named Acme Collections calls me at 6 a.m. every day about a debt I do not owe. "
             "They never sent me a validation notice. My card number is 4111 1111 1111 1111.")
