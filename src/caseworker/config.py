"""Settings and the routing tables shared by the agent, the API and the evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Case Router queue for each product firstlook can return.
QUEUES: dict[str, str] = {
    "credit_reporting": "Credit Reporting",
    "debt_collection": "Debt Collection",
    "bank_account": "Deposit Accounts",
    "credit_card": "Credit Cards",
    "mortgage": "Mortgage Servicing",
    "money_transfer": "Payments and Transfers",
    "prepaid_card": "Payments and Transfers",
    "vehicle_loan": "Auto Lending",
    "student_loan": "Student Lending",
    "personal_loan": "Consumer Lending",
    "debt_management": "Debt Relief Services",
}

# Which CFPB regulations (12 CFR parts) govern each product, used to check that a cited rule fits the case.
RULE_FAMILIES: dict[str, set[str]] = {
    "credit_reporting": {"1022"},
    "debt_collection": {"1006"},
    "bank_account": {"1005"},
    "money_transfer": {"1005"},
    "prepaid_card": {"1005"},
    "credit_card": {"1026"},
    "mortgage": {"1024", "1026", "1002"},
    "vehicle_loan": {"1026", "1002"},
    "student_loan": {"1026"},
    "personal_loan": {"1026", "1002"},
}


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    fineprint_url: str
    firstlook_url: str
    case_router_url: str
    case_router_tenant: str
    ollama_url: str
    llm_model: str
    history_db: str
    max_tool_rounds: int

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            fineprint_url=_env("FINEPRINT_URL", "http://localhost:8001"),
            firstlook_url=_env("FIRSTLOOK_URL", "http://localhost:8002"),
            case_router_url=_env("CASE_ROUTER_URL", "http://localhost:8003"),
            case_router_tenant=_env("CASE_ROUTER_TENANT", "acme"),
            ollama_url=_env("OLLAMA_URL", "http://localhost:11434"),
            llm_model=_env("LLM_MODEL", "llama3.2:3b"),
            history_db=_env("HISTORY_DB", "data/history.db"),
            max_tool_rounds=int(_env("MAX_TOOL_ROUNDS", "3")),
        )
