"""Clients for the services the agent works with: firstlook (triage), fineprint (rules) and Case Router (tickets),
plus the complaint-history database. Each has a small interface, so tests can swap in stand-ins.
"""

from __future__ import annotations

import io
import re
import sqlite3
import zipfile
from pathlib import Path
from typing import Protocol

import httpx
import pandas as pd

from .config import QUEUES


class Triage(Protocol):
    def triage(self, text: str) -> dict: ...


class Rules(Protocol):
    def search(self, query: str, k: int = 4) -> list[dict]: ...


class Tickets(Protocol):
    def open_case(self, subject: str, description: str, priority: str) -> dict: ...


class FirstlookClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def triage(self, text: str) -> dict:
        response = self.client.post("/triage", json={"text": text})
        response.raise_for_status()
        return response.json()


class FineprintClient:
    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def search(self, query: str, k: int = 4) -> list[dict]:
        response = self.client.get("/search", params={"q": query[:500], "k": k})
        response.raise_for_status()
        return response.json()["passages"]


class CaseRouterClient:
    def __init__(self, base_url: str, tenant: str, timeout: float = 60.0, client: httpx.Client | None = None) -> None:
        self.client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self.client.headers["X-Tenant-Id"] = tenant

    def install_rules(self) -> None:
        """Route by the product tag the agent puts in each ticket subject, such as product:mortgage."""
        rules = [{"kind": "priority", "value": "CRITICAL", "queue": "Escalations"}]
        rules += [{"kind": "keyword", "value": f"product:{label}", "queue": queue} for label, queue in QUEUES.items()]
        rules.append({"kind": "any", "value": "", "queue": "Intake Review"})
        self.client.put("/v1/rules", json=rules).raise_for_status()

    def open_case(self, subject: str, description: str, priority: str) -> dict:
        response = self.client.post("/v1/cases", json={"subject": subject[:200], "description": description[:5000],
                                                       "priority": priority, "channel": "WEB"})
        response.raise_for_status()
        return response.json()


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def company_key(company: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (company or "unknown").lower()).strip("-") or "unknown"


class History:
    """Past complaints by company, in SQLite, so the agent can see how a company usually responds."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)

    def load(self, frame: pd.DataFrame) -> int:
        frame = frame.rename(columns={c: _normalize(c) for c in frame.columns})
        aliases = {"company_response": ("company_response_to_consumer", "company_response"),
                   "timely": ("timely_response", "timely")}
        out = pd.DataFrame({
            "company": frame.get("company", ""),
            "product": frame.get("product", ""),
            "issue": frame.get("issue", ""),
            "company_response": next((frame[a] for a in aliases["company_response"] if a in frame), ""),
            "timely": next((frame[a] for a in aliases["timely"] if a in frame), ""),
        }).fillna("")
        out["company_key"] = out["company"].map(company_key)
        out.to_sql("complaints", self.conn, if_exists="replace", index=False)
        self.conn.execute("CREATE INDEX IF NOT EXISTS complaints_company ON complaints (company_key)")
        self.conn.commit()
        return len(out)

    def load_archives(self, paths: list[Path]) -> int:
        frames = []
        for path in paths:
            with zipfile.ZipFile(path) as archive:
                member = max((m for m in archive.infolist() if m.filename.lower().endswith(".csv")),
                             key=lambda m: m.file_size)
                with archive.open(member) as handle:
                    for chunk in pd.read_csv(io.TextIOWrapper(handle, encoding="utf-8", errors="replace"), dtype=str,
                                             keep_default_na=False, chunksize=200_000, on_bad_lines="skip"):
                        keep = [c for c in chunk.columns if _normalize(c) in {
                            "company", "product", "issue", "company_response_to_consumer", "timely_response"}]
                        frames.append(chunk[keep])
        return self.load(pd.concat(frames, ignore_index=True))

    def summary(self, company: str) -> dict:
        key = company_key(company)
        rows = self.conn.execute(
            "SELECT count(*), avg(CASE WHEN lower(timely) = 'yes' THEN 1.0 ELSE 0 END) FROM complaints WHERE company_key = ?",
            (key,)).fetchone()
        total = int(rows[0] or 0)
        if total == 0:
            return {"company": company, "complaints": 0}

        def top(column: str) -> list[dict]:
            found = self.conn.execute(
                f"SELECT {column}, count(*) n FROM complaints WHERE company_key = ? AND {column} != '' "
                f"GROUP BY {column} ORDER BY n DESC LIMIT 3", (key,)).fetchall()
            return [{"value": value, "share": round(100.0 * n / total, 1)} for value, n in found]

        return {"company": company, "complaints": total, "timely_response_pct": round(100.0 * (rows[1] or 0), 1),
                "top_products": top("product"), "top_issues": top("issue"), "usual_responses": top("company_response")}
