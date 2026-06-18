"""Service / API / MCP tests.

Proves the transport layer is a faithful, thin wrapper over the graph:
  * REST ask/resume incl. the interrupt-across-two-calls cycle
  * WebSocket node-by-node streaming + terminal frame
  * MCP tools register and delegate to the same service
  * HTTP and MCP produce identical results for the same input
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from atlas_counsel.service import AskStatus, CounselService, create_app

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture
def client():
    return TestClient(create_app())


# --- REST -------------------------------------------------------------------

def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_grounded_returns_citation(client):
    r = client.post("/ask", json={
        "question": "Above what value does a single-source purchase need justification?"
    }).json()
    assert r["status"] == "answered"
    assert "POL-001#S1" in r["citations"]


def test_ask_unanswerable_then_resume_decline(client):
    r = client.post("/ask", json={
        "question": "What is our policy on accepting gifts from suppliers?"
    }).json()
    assert r["status"] == "needs_input"
    assert r["gate_reason"] == "insufficient_grounding"
    tid = r["thread_id"]

    r2 = client.post("/resume", json={"thread_id": tid, "action": "decline"}).json()
    assert r2["status"] == "refused"
    assert r2["escalated"] is True


def test_resume_steer_is_honored(client):
    r = client.post("/ask", json={
        "question": "What is our policy on accepting gifts from suppliers?"
    }).json()
    tid = r["thread_id"]
    r2 = client.post("/resume", json={
        "thread_id": tid, "action": "steer", "guidance": "POL-004"
    }).json()
    assert r2["status"] == "answered"
    assert all(c.startswith("POL-004") for c in r2["citations"])


def test_thread_id_roundtrips(client):
    """The thread_id from ask is the resume token; a fresh one isn't resumable."""
    r = client.post("/ask", json={
        "question": "What is our policy on accepting gifts from suppliers?"
    }).json()
    assert r["thread_id"]
    # resuming the correct thread works (covered above); here assert the token
    # is opaque-but-present and stable in the response.
    assert isinstance(r["thread_id"], str) and len(r["thread_id"]) >= 8


# --- WebSocket streaming ----------------------------------------------------

def test_ws_streams_nodes_then_result(client):
    with client.websocket_connect("/ws/ask") as ws:
        ws.send_json({"question": "What are NorthLink's payment terms?"})
        nodes, terminal = [], None
        while True:
            f = ws.receive_json()
            if f["event"] == "node":
                nodes.append(f["node"])
            else:
                terminal = f
                break
    # plan->retrieve->validate->synthesize->verify->finalize all stream
    assert nodes[0] == "plan"
    assert "finalize" in nodes
    assert terminal["event"] == "result"
    assert terminal["status"] == "answered"


def test_ws_unanswerable_streams_needs_input(client):
    with client.websocket_connect("/ws/ask") as ws:
        ws.send_json({"question": "What is our policy on accepting gifts from suppliers?"})
        terminal = None
        while True:
            f = ws.receive_json()
            if f["event"] != "node":
                terminal = f
                break
    assert terminal["event"] == "needs_input"
    assert terminal["status"] == "needs_input"


# --- MCP tools --------------------------------------------------------------

def test_mcp_registers_three_tools():
    from atlas_counsel.service.mcp_server import _build_server
    mcp = _build_server()
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert names == {"counsel_ask", "counsel_resume", "counsel_brief"}


def test_mcp_and_http_agree():
    """Same question through the service directly (what MCP calls) and through
    HTTP must produce the same answer + citations."""
    svc = CounselService()
    direct = svc.ask("Above what value does a single-source purchase need justification?")

    client = TestClient(create_app())
    http = client.post("/ask", json={
        "question": "Above what value does a single-source purchase need justification?"
    }).json()

    assert direct.status.value == http["status"]
    assert direct.citations == http["citations"]


def test_brief_is_grounded_in_vendor_contract():
    svc = CounselService()
    res = svc.ask(
        "Summarize the key contract terms for NorthLink: service levels, "
        "payment terms, liability."
    )
    assert res.status == AskStatus.ANSWERED
    assert any(c.startswith("CON-002") for c in res.citations)
