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

Multi-tenancy: each tenant gets its own SqliteSaver at
`{CHECKPOINT_DIR}/{tenant_id}/checkpoints.db`. Threads are scoped per-tenant
automatically since each tenant has its own database. The retriever is shared
(same procurement corpus, read-only).
"""

from __future__ import annotations

import logging
import uuid
from enum import Enum

from pydantic import BaseModel, Field
from langgraph.types import Command

from ..agent.llm import LLMProvider
from ..chunking import chunk_corpus
from ..corpus import build_corpus
from ..decompose import QueryDecomposer
from ..embeddings import HashingEmbedder
from ..retrieval import InMemoryHybridRetriever, RetrievedChunk, Retriever
from ..telemetry import get_tracer
from .tenants import DEFAULT_TENANT, TenantRegistry

logger = logging.getLogger(__name__)

DEFAULT_RECURSION_LIMIT = 20


class AskStatus(str, Enum):
    ANSWERED = "answered"
    REFUSED = "refused"
    NEEDS_INPUT = "needs_input"   # paused at human-gate; resume required
    ERROR = "error"               # unrecoverable failure


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
    """Holds a TenantRegistry and runs ask/resume against it."""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        llm: LLMProvider | None = None,
        decomposer: QueryDecomposer | None = None,
        checkpointer=None,  # deprecated; retained for test compat
        checkpoint_db: str | None = None,
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    ) -> None:
        if retriever is None:
            retriever = _default_retriever()
        else:
            fallback = _default_retriever()
            retriever = FallbackRetriever(primary=retriever, fallback=fallback)
        self._registry = TenantRegistry(
            retriever=retriever, llm=llm, decomposer=decomposer,
        )
        self._recursion_limit = recursion_limit
        self._ready = True

    # -- public API used by both transports ---------------------------------

    def ask(self, question: str, thread_id: str | None = None,
            tenant_id: str = DEFAULT_TENANT) -> AskResult:
        tracer = get_tracer()
        with tracer.start_as_current_span("counsel.ask") as span:
            span.set_attribute("tenant_id", tenant_id)
            span.set_attribute("question_len", len(question))
            tenant = self._registry.get(tenant_id)
            cfg = self._make_config(thread_id)
            span.set_attribute("thread_id", cfg["configurable"]["thread_id"])
            out = tenant.compiled_graph.invoke({"question": question}, cfg)
            return self._interpret(out, cfg["configurable"]["thread_id"])

    def resume(self, thread_id: str, action: str,
               guidance: str | None = None,
               tenant_id: str = DEFAULT_TENANT) -> AskResult:
        """Resume a paused run. action ∈ {"steer","decline"}."""
        tracer = get_tracer()
        with tracer.start_as_current_span("counsel.resume") as span:
            span.set_attribute("tenant_id", tenant_id)
            span.set_attribute("thread_id", thread_id)
            span.set_attribute("action", action)
            tenant = self._registry.get(tenant_id)
            cfg = self._make_config(thread_id)
            snapshot = tenant.compiled_graph.get_state(cfg)
            if snapshot is None or snapshot.values == {}:
                return AskResult(
                    status=AskStatus.ERROR,
                    thread_id=thread_id,
                    answer="thread not found — it may belong to a different tenant "
                           "or have expired",
                )
            payload = {"action": action}
            if guidance is not None:
                payload["guidance"] = guidance
            out = tenant.compiled_graph.invoke(Command(resume=payload), cfg)
            return self._interpret(out, cfg["configurable"]["thread_id"])

    async def astream(self, question: str, thread_id: str | None = None,
                      tenant_id: str = DEFAULT_TENANT):
        """Yield a JSON-able frame as each node completes, then a terminal
        frame. Used by the WebSocket for streaming progress.

        Frame shapes:
          {"event": "node", "node": "<name>"}                 progress
          {"event": "result", ...AskResult fields}            done
          {"event": "needs_input", ...}                       paused at gate
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("counsel.astream") as span:
            span.set_attribute("tenant_id", tenant_id)
            span.set_attribute("question_len", len(question))
            tenant = self._registry.get(tenant_id)
            cfg = self._make_config(thread_id)
            span.set_attribute("thread_id", cfg["configurable"]["thread_id"])
            for step in tenant.compiled_graph.stream(
                {"question": question}, cfg, stream_mode="updates"
            ):
                for node_name in step:
                    if node_name != "__interrupt__":
                        yield {"event": "node", "node": node_name}
            tid = cfg["configurable"]["thread_id"]
            result = self._result_from_snapshot(tenant.compiled_graph, cfg, tid)
            event = "needs_input" if result.status == AskStatus.NEEDS_INPUT else "result"
            yield {"event": event, **result.model_dump()}

    def deep_health(self) -> dict:
        """Health check that verifies real dependencies, not just liveness."""
        result: dict = {"status": "ok", "ready": self._ready}
        registry_health = self._registry.deep_health()
        # No-tenants-yet is healthy — tenants are created lazily on first use.
        graph = registry_health.get("graph", "unknown")
        ckp = registry_health.get("checkpointer", "unknown")
        if graph != "no-tenants" and graph != "ok":
            result["status"] = "degraded"
        elif ckp != "no-tenants" and ckp != "ok":
            result["status"] = "degraded"
        result["graph"] = graph
        result["checkpointer"] = ckp
        result["tenants"] = registry_health.get("tenants", 0)
        return result

    # -- helpers ------------------------------------------------------------

    def _make_config(self, thread_id: str | None = None) -> dict:
        return {
            "configurable": {"thread_id": thread_id or uuid.uuid4().hex},
            "recursion_limit": self._recursion_limit,
        }

    @staticmethod
    def _result_from_snapshot(graph, cfg: dict, thread_id: str) -> AskResult:
        """Interpret the graph's current checkpoint into an AskResult, whether
        it paused at the gate or ran to completion."""
        snapshot = graph.get_state(cfg)
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

    @staticmethod
    def _interpret(out: dict, thread_id: str) -> AskResult:
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


def _default_retriever() -> Retriever:
    r = InMemoryHybridRetriever(HashingEmbedder())
    r.index(chunk_corpus(build_corpus()))
    return r
