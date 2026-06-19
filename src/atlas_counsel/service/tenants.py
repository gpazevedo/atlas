"""Tenant-scoped resource management.

Each tenant gets its own SqliteSaver (isolated checkpoints) while sharing the
retriever (same procurement corpus, read-only). Graphs are compiled lazily and
cached — compilation is cheap (~7 nodes) so this is fine.

Tenant ids must match `^[a-z0-9]([a-z0-9-]*[a-z0-9])?$` and are capped at 64
characters to prevent path traversal and filesystem issues.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field

from langgraph.checkpoint.sqlite import SqliteSaver

from ..agent import build_counsel_graph
from ..agent.llm import LLMProvider, TemplateLLM
from ..chunking import chunk_corpus
from ..corpus import build_corpus
from ..decompose import HeuristicDecomposer, QueryDecomposer
from ..embeddings import HashingEmbedder
from ..retrieval import InMemoryHybridRetriever, Retriever
from ..telemetry import get_tracer

DEFAULT_TENANT = "default"
_DEFAULT_CHECKPOINT_DIR = "data"
_TENANT_ID_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_TENANT_ID_MAX = 64

logger = logging.getLogger(__name__)


@dataclass
class Tenant:
    tenant_id: str
    checkpointer: SqliteSaver
    compiled_graph: object  # CompiledStateGraph


class TenantRegistry:
    """Thread-safe registry of per-tenant checkpointer + compiled graph."""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        llm: LLMProvider | None = None,
        decomposer: QueryDecomposer | None = None,
        checkpoint_dir: str | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm or TemplateLLM()
        self._decomposer = decomposer if decomposer is not None else HeuristicDecomposer()
        self._checkpoint_dir = checkpoint_dir or os.environ.get(
            "CHECKPOINT_DIR", _DEFAULT_CHECKPOINT_DIR
        )
        self._tenants: dict[str, Tenant] = {}
        self._lock = threading.Lock()

    def get(self, tenant_id: str) -> Tenant:
        TenantRegistry._validate(tenant_id)
        with self._lock:
            if tenant_id not in self._tenants:
                self._tenants[tenant_id] = self._create(tenant_id)
            return self._tenants[tenant_id]

    def deep_health(self) -> dict:
        with self._lock:
            tenant_count = len(self._tenants)
        result: dict = {"tenants": tenant_count}
        if tenant_count > 0:
            with self._lock:
                sample = next(iter(self._tenants.values()))
            try:
                cfg = {"configurable": {"thread_id": "health-check"}}
                sample.compiled_graph.invoke({"question": "health check"}, cfg)
                result["graph"] = "ok"
            except Exception as exc:
                result["graph"] = str(exc)
            try:
                snapshot = sample.compiled_graph.get_state(
                    {"configurable": {"thread_id": "health-check"}}
                )
                result["checkpointer"] = "ok" if snapshot is not None else "no-state"
            except Exception as exc:
                result["checkpointer"] = str(exc)
        else:
            result["graph"] = "no-tenants"
            result["checkpointer"] = "no-tenants"
        return result

    # -- internal ---------------------------------------------------------

    def _create(self, tenant_id: str) -> Tenant:
        tracer = get_tracer()
        with tracer.start_as_current_span("tenant.create") as span:
            span.set_attribute("tenant_id", tenant_id)
            tenant_dir = os.path.join(self._checkpoint_dir, tenant_id)
            os.makedirs(tenant_dir, exist_ok=True)
            db_path = os.path.join(tenant_dir, "checkpoints.db")
            conn = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(conn)
            retriever = self._retriever or _default_retriever()
            graph = build_counsel_graph(
                retriever, llm=self._llm, decomposer=self._decomposer, checkpointer=checkpointer
            )
            logger.info("tenant %s: created (db=%s)", tenant_id, db_path)
            return Tenant(tenant_id=tenant_id, checkpointer=checkpointer, compiled_graph=graph)

    @staticmethod
    def _validate(tenant_id: str) -> None:
        if not tenant_id or not isinstance(tenant_id, str):
            raise ValueError("tenant_id must be a non-empty string")
        if len(tenant_id) > _TENANT_ID_MAX:
            raise ValueError(
                f"tenant_id must be <= {_TENANT_ID_MAX} characters, got {len(tenant_id)}"
            )
        if not _TENANT_ID_RE.match(tenant_id):
            raise ValueError(
                "tenant_id must match [a-z0-9]([a-z0-9-]*[a-z0-9])? — "
                f"got {tenant_id!r}"
            )


def _default_retriever() -> Retriever:
    retriever = InMemoryHybridRetriever(HashingEmbedder())
    retriever.index(chunk_corpus(build_corpus()))
    return retriever
