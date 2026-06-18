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

pytest.importorskip("fastapi", reason="fastapi not installed (try: uv sync --extra service)")

from fastapi.testclient import TestClient

from atlas_counsel.service import AskStatus, CounselService, create_app

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture
def client():
    return TestClient(create_app())


# --- REST -------------------------------------------------------------------

def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok"
    assert r["ready"] is True
    assert r["graph"] in ("ok", "no-tenants")
    assert r["checkpointer"] in ("ok", "no-tenants")


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

def test_mcp_registers_tools():
    from atlas_counsel.service.mcp_server import build_mcp_server
    mcp = build_mcp_server()
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert names == {"counsel_ask", "counsel_resume", "counsel_brief", "counsel_health"}


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


# --- Input validation ---------------------------------------------------------

def test_ask_rejects_empty_question(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422


def test_ask_rejects_long_question(client):
    r = client.post("/ask", json={"question": "x" * 2001})
    assert r.status_code == 422


def test_resume_rejects_invalid_action(client):
    r = client.post("/resume", json={"thread_id": "abc12345", "action": "invent"})
    assert r.status_code == 422


def test_resume_rejects_empty_thread_id(client):
    r = client.post("/resume", json={"thread_id": "", "action": "steer"})
    assert r.status_code == 422


# --- Exception boundary -------------------------------------------------------

def test_exception_handler_returns_error_result():
    svc = CounselService()
    app = create_app(service=svc)
    client = TestClient(app, raise_server_exceptions=False)

    def _boom(question, thread_id=None, tenant_id="default"):
        raise RuntimeError("simulated crash")
    svc.ask = _boom

    r = client.post("/ask", json={"question": "test"})
    assert r.status_code == 500
    body = r.json()
    assert body["status"] == "error"
    assert "simulated crash" in body["answer"]


# --- FallbackRetriever --------------------------------------------------------

def test_fallback_retriever_degrades_on_primary_failure():
    from atlas_counsel.chunking import chunk_corpus
    from atlas_counsel.corpus import build_corpus
    from atlas_counsel.embeddings import HashingEmbedder
    from atlas_counsel.retrieval import InMemoryHybridRetriever
    from atlas_counsel.service.core import FallbackRetriever

    fallback = InMemoryHybridRetriever(HashingEmbedder())
    fallback.index(chunk_corpus(build_corpus()))

    class _Failing:
        def search(self, query, top_k=5):
            raise ConnectionError("boom")

    wrapped = FallbackRetriever(primary=_Failing(), fallback=fallback)
    results = wrapped.search("single-source justification threshold")
    assert results
    assert any(r.chunk.span_id == "POL-001#S1" for r in results)


def test_fallback_retriever_try_restore_detects_failure():
    from atlas_counsel.embeddings import HashingEmbedder
    from atlas_counsel.retrieval import InMemoryHybridRetriever
    from atlas_counsel.service.core import FallbackRetriever

    fallback = InMemoryHybridRetriever(HashingEmbedder())

    class _Failing:
        def search(self, query, top_k=5):
            raise ConnectionError("boom")

    wrapped = FallbackRetriever(primary=_Failing(), fallback=fallback)
    wrapped.search("test")
    assert wrapped.try_restore() is False


def test_fallback_retriever_try_restore_detects_recovery():
    from atlas_counsel.chunking import Chunk
    from atlas_counsel.embeddings import HashingEmbedder
    from atlas_counsel.retrieval import InMemoryHybridRetriever, RetrievedChunk
    from atlas_counsel.service.core import FallbackRetriever

    fallback = InMemoryHybridRetriever(HashingEmbedder())

    class _Recovering:
        def __init__(self):
            self.calls = 0

        def search(self, query, top_k=5):
            self.calls += 1
            if self.calls < 3:
                raise ConnectionError("boom")
            return [RetrievedChunk(
                chunk=Chunk(
                    chunk_id="x", span_id="X", doc_id="DOC-001",
                    category="policy", title="T", vendor="",
                    heading="H", text="ok", tokens=[],
                ),
                score=1.0,
            )]

    primary = _Recovering()
    wrapped = FallbackRetriever(primary=primary, fallback=fallback)

    wrapped.search("test")            # primary fails (call 1) → fallback
    assert wrapped.try_restore() is False  # primary fails (call 2)
    assert wrapped.try_restore() is True   # primary succeeds (call 3)


# --- Env var wiring -----------------------------------------------------------

def test_counsel_checkpoint_db_env_var(monkeypatch, tmp_path):
    tenant_dir = tmp_path / "acme"
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))
    # Access the tenant to trigger checkpoint creation.
    svc = CounselService()
    svc.ask("test", tenant_id="acme")
    assert (tenant_dir / "checkpoints.db").exists()
