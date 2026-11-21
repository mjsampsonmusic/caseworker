"""Evaluation: real held-out complaints worked end to end, with a reviewer that approves every draft.

What is scored automatically:
- triage: firstlook's product against the bureau's product label
- rules: whether a cited section belongs to a regulation that governs that product
- letters: the rule-based checks (no invented citations, case number present, no promises, no long numbers)
- tools: whether the model searched the rules itself, repeated calls, and how often the fallback search ran
- tickets: whether Case Router put the ticket in the queue for the triaged product
- memory: whether a later case for the same company saw notes from an earlier one
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .agent import CaseAgent
from .config import QUEUES, RULE_FAMILIES


def pick_cases(path: Path, per_product: int = 3, max_chars: int = 2000) -> list[dict]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    cases = []
    for product in RULE_FAMILIES:
        rows = frame[frame["product"] == product].head(per_product)
        for row in rows.to_dict("records"):
            cases.append({"id": row.get("complaint_id") or str(len(cases)), "product": product,
                          "issue": row.get("issue", ""), "company": row.get("company", ""),
                          "text": row["text"][:max_chars]})
    return cases


def _pct(part: float, whole: float) -> float | None:
    return round(100.0 * part / whole, 1) if whole else None


def _record(case: dict, state: dict, seconds: float, error: str | None) -> dict:
    triage = state.get("triage") or {}
    checks = state.get("checks") or {}
    first = state.get("first_checks") or {}
    cited = checks.get("cited", [])
    families = RULE_FAMILIES.get(case["product"], set())
    ticket = state.get("ticket") or {}
    product = triage.get("routed_to") or triage.get("product")
    tool_use = state.get("tool_use") or {}
    return {
        "id": case["id"], "company": case["company"], "issue": case["issue"], "truth": case["product"],
        "text": case["text"][:600], "error": error, "status": state.get("status"),
        "triage_product": triage.get("product"), "triage_confidence": triage.get("confidence"),
        "triage_route": triage.get("route"), "triage_correct": triage.get("product") == case["product"],
        "cited": cited, "rule_family_match": any(s.split(".")[0] in families for s in cited),
        "first_draft_passed": bool(first.get("passed")), "final_passed": bool(checks.get("passed")),
        "invented_final": checks.get("invented", []), "problems_final": checks.get("problems", []),
        "repairs": state.get("repairs", 0), "tool_use": tool_use,
        "notes_seen": len(state.get("notes") or []),
        "ticket_queue": ticket.get("queue"), "expected_queue": QUEUES.get(product or "", "Intake Review"),
        "letter": (state.get("draft") or {}).get("letter", ""), "summary": (state.get("draft") or {}).get("summary", ""),
        "priority": (state.get("draft") or {}).get("priority"),
        "input_tokens": state.get("usage", {}).get("input_tokens", 0),
        "output_tokens": state.get("usage", {}).get("output_tokens", 0),
        "seconds": round(seconds, 1), "trace": state.get("trace", []),
    }


def summarize(records: list[dict]) -> dict:
    done = [r for r in records if not r["error"]]
    filed = [r for r in done if r["status"] == "filed"]
    tools = [r["tool_use"] for r in done if r["tool_use"]]
    repeat_company = [r for r in done if r["notes_seen"] > 0]
    seconds = [r["seconds"] for r in done]
    return {
        "cases": len(records),
        "completed": len(done),
        "triage_accuracy": _pct(sum(r["triage_correct"] for r in done), len(done)),
        "rule_family_match": _pct(sum(r["rule_family_match"] for r in done), len(done)),
        "letters_passing_first_draft": _pct(sum(r["first_draft_passed"] for r in done), len(done)),
        "letters_passing_after_repair": _pct(sum(r["final_passed"] for r in done), len(done)),
        "invented_citations_in_final_drafts": sum(len(r["invented_final"]) for r in done),
        "model_searched_rules_itself": _pct(sum(bool(t.get("model_found_rules")) for t in tools), len(tools)),
        "fallback_searches": _pct(sum(bool(t.get("fallback_search")) for t in tools), len(tools)),
        "avg_tool_calls": round(statistics.fmean([t.get("calls", 0) for t in tools]), 2) if tools else None,
        "repeated_tool_calls": sum(t.get("repeated_calls", 0) for t in tools),
        "tickets_in_expected_queue": _pct(sum(r["ticket_queue"] == r["expected_queue"] for r in filed), len(filed)),
        "cases_that_saw_earlier_notes": len(repeat_company),
        "avg_input_tokens": round(statistics.fmean([r["input_tokens"] for r in done]), 1) if done else None,
        "avg_output_tokens": round(statistics.fmean([r["output_tokens"] for r in done]), 1) if done else None,
        "median_seconds": round(float(np.percentile(seconds, 50)), 1) if seconds else None,
        "p95_seconds": round(float(np.percentile(seconds, 95)), 1) if seconds else None,
    }


def evaluate(agent: CaseAgent, cases: list[dict], out_dir: Path, model: str = "", log=print) -> dict:
    started = time.perf_counter()
    records = []
    for i, case in enumerate(cases, start=1):
        case_started = time.perf_counter()
        error = None
        state: dict = {}
        case_id = f"EV-{i:03d}"
        try:
            state = agent.start(case["text"], case["company"], case_id=case_id)
            if state["status"] == "awaiting review":
                state = agent.review(case_id, approved=True, reviewer="evaluation run")
        except Exception as exc:  # one broken case is recorded and the run goes on
            error = f"{type(exc).__name__}: {exc}"[:300]
        records.append(_record(case, state, time.perf_counter() - case_started, error))
        log(f"  {i}/{len(cases)} {case['product']}: {state.get('status') or 'error'} in "
            f"{time.perf_counter() - case_started:.0f}s")
    results = {
        "run": {"finished": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "minutes": round((time.perf_counter() - started) / 60, 1),
                "machine": f"{platform.system()} {platform.machine()}, {__import__('os').cpu_count()} CPUs, no GPU",
                "model": model, "cases": len(cases)},
        "summary": summarize(records),
        "records": records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (out_dir / "summary.md").write_text(summary_markdown(results))
    return results


def _fmt(value, suffix: str = "%") -> str:
    return "n/a" if value is None else f"{value}{suffix}"


def summary_markdown(results: dict, site_url: str | None = None) -> str:
    run, s = results["run"], results["summary"]
    rows = [
        ("Triage matched the bureau's product label", _fmt(s["triage_accuracy"])),
        ("A cited rule comes from a regulation that governs that product", _fmt(s["rule_family_match"])),
        ("Drafts passing every check on the first try", _fmt(s["letters_passing_first_draft"])),
        ("Drafts passing every check after one repair", _fmt(s["letters_passing_after_repair"])),
        ("Invented citations left in final drafts", _fmt(s["invented_citations_in_final_drafts"], "")),
        ("The model searched the rules itself", _fmt(s["model_searched_rules_itself"])),
        ("Average tool calls per case", _fmt(s["avg_tool_calls"], "")),
        ("Tickets Case Router put in the expected queue", _fmt(s["tickets_in_expected_queue"])),
        ("Cases that saw notes from an earlier case for the same company", _fmt(s["cases_that_saw_earlier_notes"], "")),
        ("Median / 95th percentile seconds per case", f"{_fmt(s['median_seconds'], ' s')} / {_fmt(s['p95_seconds'], ' s')}"),
    ]
    lines = [f"Measured {run['finished']} on a GitHub runner ({run['machine']}) with `{run['model']}`: "
             f"{s['completed']} of {s['cases']} held-out CFPB complaints worked end to end, every draft approved by "
             "the evaluation's reviewer.", "", "| | |", "|---|---|"]
    lines += [f"| {label} | {value} |" for label, value in rows]
    if site_url:
        lines += ["", f"Every case, tool call, draft letter and ticket: {site_url}"]
    return "\n".join(lines) + "\n"
