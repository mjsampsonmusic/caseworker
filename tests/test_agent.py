from langchain_core.messages import AIMessage

from caseworker.agent import check_draft, redact

from .conftest import COMPLAINT, PASSAGES, FakeRules, FakeTickets, draft_reply, make_agent, tool_call


def test_case_pauses_for_review_before_any_ticket(history):
    tickets = FakeTickets()
    agent, research, _ = make_agent(history, tickets=tickets)
    case = agent.start(COMPLAINT, "Acme Collections", case_id="CW-TEST0001")
    assert case["status"] == "awaiting review"
    assert tickets.opened == []
    assert "4111" not in case["complaint"] and case["redactions"] == ["1 account number"]
    assert {p["section"] for p in case["passages"]} == {"1006.6", "1006.34"}
    assert case["history"]["complaints"] == 3
    assert case["tool_use"] == {"calls": 2, "find_rules_calls": 1, "company_history_calls": 1, "repeated_calls": 0,
                                "model_found_rules": True, "fallback_search": False}
    assert case["checks"]["passed"] and case["checks"]["cited"] == ["1006.6"]
    assert case["usage"]["input_tokens"] > 0
    assert [s["step"] for s in case["trace"]] == ["intake", "triage", "research", "research", "research", "collect",
                                                  "draft", "check"]


def test_approval_files_a_ticket_and_remembers_the_company(history):
    tickets = FakeTickets()
    agent, research, _ = make_agent(history, tickets=tickets)
    agent.start(COMPLAINT, "Acme Collections", case_id="CW-TEST0001")
    filed = agent.review("CW-TEST0001", approved=True, reviewer="Mitchell")
    assert filed["status"] == "filed" and filed["ticket"]["queue"] == "Debt Collection"
    opened = tickets.opened[0]
    assert opened["subject"].startswith("[product:debt_collection]") and opened["priority"] == "MEDIUM"
    assert "CW-TEST0001" in opened["description"] and "1006.6" in opened["description"]

    agent2_research = make_agent(history)[1]
    agent.research_llm = agent2_research
    agent.draft_llm.replies.append(draft_reply(["1006.34"]))
    second = agent.start("The same collector keeps calling my job after I asked them to stop calling there.",
                         "Acme Collections", case_id="CW-TEST0002")
    assert len(second["notes"]) == 1 and second["notes"][0]["case_id"] == "CW-TEST0001"
    prompt = agent2_research.calls[0][1].content
    assert "Earlier cases with this company" in prompt


def test_reviewer_can_return_a_case_or_edit_the_letter(history):
    tickets = FakeTickets()
    agent, _, _ = make_agent(history, tickets=tickets)
    agent.start(COMPLAINT, "Acme Collections", case_id="CW-TEST0003")
    returned = agent.review("CW-TEST0003", approved=False, reviewer="Mitchell")
    assert returned["status"] == "returned" and tickets.opened == []

    agent, _, _ = make_agent(history, tickets=tickets)
    agent.start(COMPLAINT, "Acme Collections", case_id="CW-TEST0004")
    agent.review("CW-TEST0004", approved=True, letter="Edited letter for CW-TEST0004.")
    assert "Edited letter for CW-TEST0004." in tickets.opened[0]["description"]


def test_invented_citation_is_repaired(history):
    agent, _, drafter = make_agent(history, draft_replies=[draft_reply(["1006.6", "1024.41"]), draft_reply(["1006.6"])])
    case = agent.start(COMPLAINT, "Acme Collections", case_id="CW-TEST0005")
    assert case["repairs"] == 1
    assert not case["first_checks"]["passed"] and case["first_checks"]["invented"] == ["1024.41"]
    assert case["checks"]["passed"]
    assert "Fix these problems" in drafter.calls[-1][-1].content


def test_model_that_skips_tools_gets_a_fallback_search(history):
    rules = FakeRules()
    agent, _, _ = make_agent(history, research_replies=[AIMessage(content="No tools needed.")], rules=rules)
    case = agent.start(COMPLAINT, "Acme Collections", case_id="CW-TEST0006")
    assert case["tool_use"]["fallback_search"] and case["tool_use"]["calls"] == 0
    assert rules.queries and case["passages"]
    assert case["history"]["complaints"] == 3


def test_tool_rounds_are_capped(history):
    loops = [tool_call("find_rules", {"query": "debt collector calls"}, n) for n in range(10)]
    agent, _, _ = make_agent(history, research_replies=loops)
    case = agent.start(COMPLAINT, "Acme Collections", case_id="CW-TEST0007")
    assert case["tool_use"]["find_rules_calls"] == agent.max_tool_rounds + 1
    assert case["tool_use"]["repeated_calls"] == agent.max_tool_rounds
    assert case["status"] == "awaiting review"


def test_checks_catch_promises_long_numbers_and_missing_case_numbers():
    passages = [PASSAGES["1006.6"]]
    good = {"cited_sections": ["1006.6"], "letter": "Case CW-1A2B3C4D is under review."}
    assert check_draft(good, passages, "CW-1A2B3C4D")["passed"]
    bad = {"cited_sections": ["§ 1006.6", "1026.13"],
           "letter": "We guarantee a refund to account 123456789."}
    problems = check_draft(bad, passages, "CW-1A2B3C4D")["problems"]
    assert len(problems) == 4


def test_redaction_masks_identifiers():
    text, found = redact("SSN 123-45-6789, phone (812) 555-0100, card 4111-1111-1111-1111")
    assert "6789" not in text and "555" not in text and "4111" not in text
    assert found == ["1 ssn", "1 phone", "1 account number"]
