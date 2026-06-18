"""Transport-agnostic service layer.

Both the FastAPI routes and the MCP tools call THESE functions — the HTTP and
MCP layers are pure transport, never logic. That guarantees `counsel.ask` over
MCP and `POST /ask` over HTTP behave identically.

The central problem solved here: the agent graph pauses at the human-gate via
LangGraph `interrupt()`, but HTTP and MCP tool calls are request/response — a
pause can't block the connection. So a paused run is surfaced as a *result*
(`status="needs_input"` + a `thread_id` resume token), and a second call
resumes it. The graph's checkpointer is what carries state across those two
otherwise-independent calls.

State that must persist across calls lives in the checkpointer, keyed by
thread_id; the service itself stays stateless apart from holding the compiled
graph.
"""

from __future__ import annotations

import logging
import uuid
from enum import Enum

from pydantic import BaseModel, Field
from langgraph.types import Command

from ..agent import build_counsel_graph
from ..agent.llm import LLMProvider
from ..chunking import chunk_corpus
from ..corpus import build_corpus
from ..decompose import QueryDecomposer
from ..embeddings import HashingEmbedder
from ..retrieval import InMemoryHybridRetriever, RetrievedChunk, Retriever

logger = logging.getLogger(__name__)

DEFAULT_RECURSION_LIMIT = 20


class AskStatus(str, Enum):
    ANSWERED = "answered"
    REFUSED = "refused"
    NEEDS_INPUT = "needs_input"   # paused at human-gate; resume required
    ERROR = "error"               # unrecoverable failure


class Citation(BaseModel):
    span_id: str


class AskResult(BaseModel):
    status: AskStatus
    thread_id: str = Field(..., description="Resume token; opaque to callers")
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    # Present only when status == needs_input.
    gate_reason: str | None = None
    retrieved_span_ids: list[str] = Field(default_factory=list)
    attempts: int = 0
    escalated: bool = False


class FallbackRetriever:
    """Wraps a primary retriever with an in-memory fallback for resilience.

    If the primary (e.g. Qdrant) fails, this silently degrades to the
    in-memory index so requests still get answers instead of errors.
    """

    def __init__(self, primary: Retriever, fallback: Retriever) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_healthy = True

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self._primary_healthy:
            try:
                return self._primary.search(query, top_k=top_k)
            except Exception:
                logger.warning("primary retriever failed, falling back to in-memory",
                               exc_info=True)
                self._primary_healthy = False
        return self._fallback.search(query, top_k=top_k)

    def try_restore(self) -> bool:
        """Probe the primary; return True if it recovered."""
        if self._primary_healthy:
            return True
        try:
            self._primary.search("health-check probe", top_k=1)
            self._primary_healthy = True
            logger.info("primary retriever recovered")
            return True
        except Exception:
            return False


class CounselService:
    """Holds one compiled graph and runs ask/resume against it."""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        llm: LLMProvider | None = None,
        decomposer: QueryDecomposer | None = None,
        checkpointer=None,
        checkpoint_db: str | None = None,
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    ) -> None:
        if retriever is None:
            retriever = InMemoryHybridRetriever(HashingEmbedder())
            retriever.index(chunk_corpus(build_corpus()))
        else:
            # Wrap with in-memory fallback so transient Qdrant failures don't
            # block requests. The fallback is pre-indexed with the same corpus.
            fallback = InMemoryHybridRetriever(HashingEmbedder())
            fallback.index(chunk_corpus(build_corpus()))
            retriever = FallbackRetriever(primary=retriever, fallback=fallback)
        if checkpointer is None:
            import os
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
            db_path = checkpoint_db or os.environ.get("COUNSEL_CHECKPOINT_DB", "checkpoints.db")
            conn = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(conn)
        self._checkpointer = checkpointer
        self._recursion_limit = recursion_limit
        self._graph = build_counsel_graph(
            retriever, llm=llm, decomposer=decomposer, checkpointer=checkpointer
        )
        self._ready = True

    # -- public API used by both transports ---------------------------------

    def ask(self, question: str, thread_id: str | None = None) -> AskResult:
        cfg = self._make_config(thread_id)
        out = self._graph.invoke({"question": question}, cfg)
        return self._interpret(out, cfg["configurable"]["thread_id"])

    def resume(self, thread_id: str, action: str, guidance: str | None = None) -> AskResult:
        """Resume a paused run. action ∈ {"steer","decline"}."""
        cfg = self._make_config(thread_id)
        payload = {"action": action}
        if guidance is not None:
            payload["guidance"] = guidance
        out = self._graph.invoke(Command(resume=payload), cfg)
        return self._interpret(out, cfg["configurable"]["thread_id"])

    async def astream(self, question: str, thread_id: str | None = None):
        """Yield a JSON-able frame as each node completes, then a terminal
        frame. Used by the WebSocket for streaming progress.

        Frame shapes:
          {"event": "node", "node": "<name>"}                 progress
          {"event": "result", ...AskResult fields}            done
          {"event": "needs_input", ...}                       paused at gate
        """
        cfg = self._make_config(thread_id)
        # stream_mode="updates" yields {node_name: partial_state} per step.
        for step in self._graph.stream({"question": question}, cfg, stream_mode="updates"):
            for node_name in step:
                if node_name != "__interrupt__":
                    yield {"event": "node", "node": node_name}
        # Build the terminal frame from the final checkpointed state.
        tid = cfg["configurable"]["thread_id"]
        result = self._result_from_snapshot(cfg, tid)
        event = "needs_input" if result.status == AskStatus.NEEDS_INPUT else "result"
        yield {"event": event, **result.model_dump()}

    def deep_health(self) -> dict:
        """Health check that verifies real dependencies, not just liveness."""
        result: dict = {"status": "ok", "ready": self._ready}
        # Probe the graph with a minimal invoke.
        try:
            cfg = self._make_config("health-check")
            self._graph.invoke({"question": "health check"}, cfg)
            result["graph"] = "ok"
        except Exception as exc:
            result["status"] = "degraded"
            result["graph"] = str(exc)
        # Probe the checkpointer.
        try:
            snapshot = self._graph.get_state(self._make_config("health-check"))
            result["checkpointer"] = "ok" if snapshot is not None else "no-state"
        except Exception as exc:
            result["status"] = "degraded"
            result["checkpointer"] = str(exc)
        return result

    # -- helpers ------------------------------------------------------------

    def _make_config(self, thread_id: str | None = None) -> dict:
        return {
            "configurable": {"thread_id": thread_id or uuid.uuid4().hex},
            "recursion_limit": self._recursion_limit,
        }

    def _result_from_snapshot(self, cfg: dict, thread_id: str) -> AskResult:
        """Interpret the graph's current checkpoint into an AskResult, whether
        it paused at the gate or ran to completion."""
        snapshot = self._graph.get_state(cfg)
        for task in snapshot.tasks:
            interrupts = getattr(task, "interrupts", None)
            if interrupts:
                intr = interrupts[0].value
                return AskResult(
                    status=AskStatus.NEEDS_INPUT,
                    thread_id=thread_id,
                    gate_reason=intr.get("reason"),
                    retrieved_span_ids=intr.get("retrieved_span_ids", []),
                )
        ans = snapshot.values["answer"]
        return AskResult(
            status=AskStatus.REFUSED if ans.refused else AskStatus.ANSWERED,
            thread_id=thread_id,
            answer=ans.text,
            citations=ans.citations,
            attempts=ans.attempts,
            escalated=ans.escalated,
        )

    # -- helpers ------------------------------------------------------------

    def _interpret(self, out: dict, thread_id: str) -> AskResult:
        if "__interrupt__" in out:
            intr = out["__interrupt__"][0].value
            return AskResult(
                status=AskStatus.NEEDS_INPUT,
                thread_id=thread_id,
                gate_reason=intr.get("reason"),
                retrieved_span_ids=intr.get("retrieved_span_ids", []),
            )
        ans = out["answer"]
        return AskResult(
            status=AskStatus.REFUSED if ans.refused else AskStatus.ANSWERED,
            thread_id=thread_id,
            answer=ans.text,
            citations=ans.citations,
            attempts=ans.attempts,
            escalated=ans.escalated,
        )
