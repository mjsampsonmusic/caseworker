"""Streamlit front end: open a case, read the agent's work, approve or return it."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API = os.getenv("CASEWORKER_API", "http://localhost:8000")
EXAMPLE = ("A debt collector calls me at 6 a.m. every day about a debt I do not owe. They never sent me a "
           "validation notice, and they called my job after I asked them to stop.")

st.set_page_config(page_title="caseworker", layout="wide")
st.title("caseworker")
st.caption("An agent works the complaint; a person approves every letter before a ticket is opened.")

with st.form("new"):
    company = st.text_input("Company", value="Example Collections")
    complaint = st.text_area("Complaint", value=EXAMPLE, height=140)
    if st.form_submit_button("Work this case", type="primary"):
        with st.spinner("Triaging, looking up rules and company history, drafting..."):
            response = httpx.post(f"{API}/cases", json={"complaint": complaint, "company": company}, timeout=900)
        if response.status_code != 201:
            st.error(response.text[:400])
        else:
            st.session_state.case = response.json()

case = st.session_state.get("case")
if case:
    st.subheader(f"Case {case['case_id']}: {case['status']}")
    left, right = st.columns(2)
    with left:
        triage = case["triage"] or {}
        st.metric("Triage", str(triage.get("product")), f"{triage.get('confidence', 0):.0%} sure")
        st.write("**Rules found**")
        for passage in case["passages"]:
            st.caption(f"{passage['citation']}: {passage['text'][:220]}")
        st.write("**Company history**", case["history"])
        st.write("**Tool use**", case["tool_use"])
    with right:
        draft = case["draft"] or {}
        st.write(f"**Summary** ({draft.get('priority')} priority): {draft.get('summary')}")
        letter = st.text_area("Letter (edit before approving if needed)", value=draft.get("letter", ""), height=220)
        checks = case["checks"] or {}
        if checks.get("passed"):
            st.success("The draft passed every check.")
        else:
            st.warning("Problems: " + "; ".join(checks.get("problems", [])))
        if case["status"] == "awaiting review":
            approve, send_back = st.columns(2)
            decision = None
            if approve.button("Approve and open ticket", type="primary"):
                decision = {"approved": True, "letter": letter, "reviewer": "web reviewer"}
            if send_back.button("Return for rework"):
                decision = {"approved": False, "reviewer": "web reviewer"}
            if decision:
                result = httpx.post(f"{API}/cases/{case['case_id']}/review", json=decision, timeout=300).json()
                st.session_state.case = result
                st.rerun()
        if case.get("ticket"):
            st.success(f"Ticket {case['ticket']['id']} opened in {case['ticket']['queue']}, "
                       f"respond by {case['ticket'].get('respond_by')}")
    with st.expander("Steps, tokens and time"):
        st.table(case["trace"])
        st.write(case["usage"])
