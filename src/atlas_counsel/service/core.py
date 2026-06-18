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
from ..retrieval import InMemoryHybridRetriever, Retriever


class AskStatus(str, Enum):
    ANSWERED = "answered"
    REFUSED = "refused"
    NEEDS_INPUT = "needs_input"   # paused at human-gate; resume required


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


class CounselService:
    """Holds one compiled graph and runs ask/resume against it."""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        llm: LLMProvider | None = None,
        decomposer: QueryDecomposer | None = None,
        checkpointer=None,
    ) -> None:
        if retriever is None:
            retriever = InMemoryHybridRetriever(HashingEmbedder())
            retriever.index(chunk_corpus(build_corpus()))
        if checkpointer is None:
            # In-memory default; prod injects Sqlite/Postgres saver.
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
        self._graph = build_counsel_graph(
            retriever, llm=llm, decomposer=decomposer, checkpointer=checkpointer
        )

    # -- public API used by both transports ---------------------------------

    def ask(self, question: str, thread_id: str | None = None) -> AskResult:
        thread_id = thread_id or uuid.uuid4().hex
        cfg = {"configurable": {"thread_id": thread_id}}
        out = self._graph.invoke({"question": question}, cfg)
        return self._interpret(out, thread_id)

    def resume(self, thread_id: str, action: str, guidance: str | None = None) -> AskResult:
        """Resume a paused run. action ∈ {"steer","decline"}."""
        cfg = {"configurable": {"thread_id": thread_id}}
        payload = {"action": action}
        if guidance is not None:
            payload["guidance"] = guidance
        out = self._graph.invoke(Command(resume=payload), cfg)
        return self._interpret(out, thread_id)

    async def astream(self, question: str, thread_id: str | None = None):
        """Yield a JSON-able frame as each node completes, then a terminal
        frame. Used by the WebSocket for streaming progress.

        Frame shapes:
          {"event": "node", "node": "<name>"}                 progress
          {"event": "result", ...AskResult fields}            done
          {"event": "needs_input", ...}                       paused at gate
        """
        thread_id = thread_id or uuid.uuid4().hex
        cfg = {"configurable": {"thread_id": thread_id}}
        # stream_mode="updates" yields {node_name: partial_state} per step.
        for step in self._graph.stream({"question": question}, cfg, stream_mode="updates"):
            for node_name in step:
                if node_name != "__interrupt__":
                    yield {"event": "node", "node": node_name}
        # Build the terminal frame from the final checkpointed state.
        result = self._result_from_snapshot(cfg, thread_id)
        event = "needs_input" if result.status == AskStatus.NEEDS_INPUT else "result"
        yield {"event": event, **result.model_dump()}

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
