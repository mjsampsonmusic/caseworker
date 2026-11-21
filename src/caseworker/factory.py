"""Builds the real agent: Llama through Ollama, and HTTP clients for the other services."""

from __future__ import annotations

from .agent import DRAFT_SCHEMA, CaseAgent
from .config import Settings
from .services import CaseRouterClient, FineprintClient, FirstlookClient, History


def build_agent(settings: Settings) -> CaseAgent:
    from langchain_ollama import ChatOllama

    research = ChatOllama(model=settings.llm_model, base_url=settings.ollama_url, temperature=0, num_ctx=8192,
                          num_predict=400)
    draft = ChatOllama(model=settings.llm_model, base_url=settings.ollama_url, temperature=0, num_ctx=8192,
                       num_predict=700, format=DRAFT_SCHEMA)
    tickets = CaseRouterClient(settings.case_router_url, settings.case_router_tenant)
    tickets.install_rules()
    return CaseAgent(research, draft, FirstlookClient(settings.firstlook_url), FineprintClient(settings.fineprint_url),
                     tickets, History(settings.history_db), max_tool_rounds=settings.max_tool_rounds)
