import io
import json
import zipfile

import httpx
import pandas as pd
import pytest
import respx
from fastapi.testclient import TestClient

from caseworker.api import create_app
from caseworker.config import QUEUES
from caseworker.evaluate import evaluate, pick_cases, summary_markdown
from caseworker.services import CaseRouterClient, FineprintClient, FirstlookClient, History
from caseworker.site import build_site, update_readme

from .conftest import COMPLAINT, FakeTickets, draft_reply, make_agent


def test_history_summary(history):
    summary = history.summary("ACME Collections")
    assert summary["complaints"] == 3 and summary["timely_response_pct"] == 66.7
    assert summary["top_issues"][0] == {"value": "Communication tactics", "share": 66.7}
    assert history.summary("Nobody") == {"company": "Nobody", "complaints": 0}


def test_history_loads_bureau_archive_zips(tmp_path):
    frame = pd.DataFrame({"Company": ["A", "A"], "Product": ["Mortgage", "Mortgage"], "Issue": ["x", "y"],
                          "Company response to consumer": ["Closed", "Closed"], "Timely response?": ["Yes", "No"],
                          "Consumer complaint narrative": ["n", "n"]})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("complaints.csv", frame.to_csv(index=False))
    (tmp_path / "a.zip").write_bytes(buffer.getvalue())
    store = History(tmp_path / "h.db")
    assert store.load_archives([tmp_path / "a.zip"]) == 2
    assert store.summary("A")["timely_response_pct"] == 50.0


@respx.mock
def test_http_clients():
    respx.post("http://firstlook/triage").mock(return_value=httpx.Response(200, json={"product": "mortgage"}))
    respx.get("http://fineprint/search").mock(return_value=httpx.Response(200, json={"passages": [{"section": "1024.35"}]}))
    assert FirstlookClient("http://firstlook").triage("escrow problem")["product"] == "mortgage"
    assert FineprintClient("http://fineprint").search("error notice")[0]["section"] == "1024.35"


def test_case_router_routes_tickets_by_product_tag(tmp_path):
    case_router = pytest.importorskip("case_router.api")
    app = case_router.create_app(case_router.Settings(database_path=str(tmp_path / "cr.db"), seed_demo_data=True))
    client = CaseRouterClient("http://testserver", "acme", client=TestClient(app))
    client.install_rules()
    ticket = client.open_case("[product:mortgage] Escrow shortage after a refund charge", "Escrow problem", "MEDIUM")
    assert ticket["queue"] == QUEUES["mortgage"]
    other = client.open_case("A question with no product tag", "", "LOW")
    assert other["queue"] == "Intake Review"


def test_api_flow(history):
    tickets = FakeTickets()
    agent, _, _ = make_agent(history, tickets=tickets)
    api = TestClient(create_app(agent))
    created = api.post("/cases", json={"complaint": COMPLAINT, "company": "Acme Collections"})
    assert created.status_code == 201
    case_id = created.json()["case_id"]
    assert api.get(f"/cases/{case_id}").json()["status"] == "awaiting review"
    filed = api.post(f"/cases/{case_id}/review", json={"approved": True, "reviewer": "Mitchell"}).json()
    assert filed["status"] == "filed" and len(tickets.opened) == 1
    assert api.post(f"/cases/{case_id}/review", json={"approved": True}).status_code == 409
    assert api.get("/cases/CW-NOPE").status_code == 404
    assert api.post("/cases", json={"complaint": "too short"}).status_code == 422


def test_evaluation_writes_results_site_and_readme(history, tmp_path):
    test_csv = tmp_path / "test.csv"
    pd.DataFrame([
        {"complaint_id": "1", "product": "debt_collection", "issue": "Communication tactics",
         "company": "Acme Collections", "text": COMPLAINT},
        {"complaint_id": "2", "product": "debt_collection", "issue": "Written notification about debt",
         "company": "Acme Collections", "text": COMPLAINT},
        {"complaint_id": "3", "product": "mortgage", "issue": "Trouble during payment process",
         "company": "Home Lender", "text": "My mortgage servicer misapplied my payment to escrow."},
    ]).to_csv(test_csv, index=False)
    cases = pick_cases(test_csv, per_product=2)
    assert [c["product"] for c in cases] == ["debt_collection", "debt_collection", "mortgage"]
    agent, research, drafter = make_agent(history, draft_replies=[draft_reply(["1006.6"])] * 3)
    research.replies = [r for _ in range(3) for r in make_agent(history)[1].replies]
    results = evaluate(agent, cases, tmp_path / "results", "scripted", log=lambda *_: None)
    s = results["summary"]
    assert s["completed"] == 3 and s["triage_accuracy"] == 66.7
    assert s["rule_family_match"] == 66.7
    assert s["tickets_in_expected_queue"] == 100.0
    assert s["cases_that_saw_earlier_notes"] == 1
    page = build_site(json.loads((tmp_path / "results" / "results.json").read_text()), tmp_path / "docs").read_text()
    assert "Every case" in page and "triage wrong" in page
    readme = tmp_path / "README.md"
    readme.write_text("x\n<!-- results:start -->\n<!-- results:end -->\n")
    assert update_readme(readme, summary_markdown(results))
    assert "Triage matched the bureau's product label | 66.7%" in readme.read_text()
