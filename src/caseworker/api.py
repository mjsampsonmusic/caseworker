"""HTTP API: open a case, read it, and approve or return the draft."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import CaseAgent
from .config import Settings


class NewCase(BaseModel):
    complaint: str = Field(min_length=20, max_length=20_000)
    company: str = Field(default="", max_length=200)


class Review(BaseModel):
    approved: bool
    letter: str | None = Field(default=None, max_length=5000)
    reviewer: str = Field(default="", max_length=100)


def create_app(agent: CaseAgent | None = None) -> FastAPI:
    app = FastAPI(title="caseworker", version="1.0.0",
                  description="An agent that works a consumer complaint and waits for a person to approve it.")
    holder: dict = {"agent": agent}

    def current() -> CaseAgent:
        if holder["agent"] is None:
            from .factory import build_agent

            holder["agent"] = build_agent(Settings.from_env())
        return holder["agent"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/cases", status_code=201)
    def open_case(body: NewCase) -> dict:
        return current().start(body.complaint, body.company)

    @app.get("/cases/{case_id}")
    def get_case(case_id: str) -> dict:
        try:
            return current().get(case_id)
        except KeyError as error:
            raise HTTPException(404, f"No case {case_id}") from error

    @app.post("/cases/{case_id}/review")
    def review(case_id: str, body: Review) -> dict:
        try:
            case = current().get(case_id)
        except KeyError as error:
            raise HTTPException(404, f"No case {case_id}") from error
        if case["status"] != "awaiting review":
            raise HTTPException(409, f"Case {case_id} is {case['status']}, not awaiting review")
        return current().review(case_id, body.approved, body.letter, body.reviewer)

    return app


app = create_app()
