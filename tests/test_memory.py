"""Multi-tier memory tests.

Covers:
  * InMemoryMemoryStore CRUD for all three tiers
  * SqliteMemoryStore CRUD (temp file, verifies persistence)
  * Tenant isolation (two tenants don't leak)
  * TemplateLLM.reflect() deterministic output
  * Graph: load_memory populates context, save_memory persists
  * Graph: backward compatible without memory_store
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from atlas_counsel.agent import build_counsel_graph
from atlas_counsel.agent.llm import TemplateLLM
from atlas_counsel.chunking import chunk_corpus
from atlas_counsel.corpus import build_corpus
from atlas_counsel.embeddings import HashingEmbedder
from atlas_counsel.memory import (
    EpisodicEntry,
    InMemoryMemoryStore,
    ProceduralSkill,
    SemanticRecord,
    SqliteMemoryStore,
)
from atlas_counsel.retrieval import InMemoryHybridRetriever


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def retriever(embedder):
    r = InMemoryHybridRetriever(embedder)
    r.index(chunk_corpus(build_corpus()))
    return r


@pytest.fixture
def mem_store(embedder):
    return InMemoryMemoryStore(embedder)


@pytest.fixture
def sqlite_store(embedder):
    tmp = tempfile.mkdtemp()
    db = Path(tmp) / "mem.db"
    store = SqliteMemoryStore(str(db), embedder)
    yield store
    # SqliteMemoryStore owns the connection; nothing to explicitly close


# -- InMemoryMemoryStore ---------------------------------------------------

class TestInMemoryStore:
    def test_semantic_write_and_search(self, mem_store):
        sid = mem_store.semantic_write(
            "POL-001 requires VP approval above $50K",
            thread_id="t1", tenant_id="acme",
        )
        assert sid.startswith("sem-")

        results = mem_store.semantic_search(
            "purchase approval threshold", top_k=3, tenant_id="acme",
        )
        assert len(results) == 1
        assert results[0].text == "POL-001 requires VP approval above $50K"
        assert results[0].score > 0

    def test_semantic_empty_tenant(self, mem_store):
        results = mem_store.semantic_search("anything", top_k=3, tenant_id="empty")
        assert results == []

    def test_episodic_upsert_and_search(self, mem_store):
        mem_store.episodic_upsert(
            "User asked about purchase approvals", thread_id="t1", tenant_id="acme",
        )
        results = mem_store.episodic_search(
            "purchase approval thresholds", top_k=3, tenant_id="acme",
        )
        assert len(results) == 1
        assert "purchase approvals" in results[0].summary

    def test_episodic_upsert_replaces(self, mem_store):
        mem_store.episodic_upsert("First summary", thread_id="t1", tenant_id="acme")
        mem_store.episodic_upsert("Updated summary", thread_id="t1", tenant_id="acme")
        results = mem_store.episodic_search("summary", top_k=5, tenant_id="acme")
        # Single entry per thread — upsert, not append
        assert len(results) == 1
        assert "Updated" in results[0].summary

    def test_procedural_save_and_search(self, mem_store):
        skill = ProceduralSkill(
            name="compare_policies",
            fragment="When comparing two policies, cite exact dollar thresholds from each.",
            when_to_use="user asks about differences between policies",
        )
        mem_store.procedural_save(skill, tenant_id="acme")
        results = mem_store.procedural_search(
            "what is the difference between POL-001 and POL-002?", top_k=3,
            tenant_id="acme",
        )
        assert len(results) == 1
        assert results[0].name == "compare_policies"

    def test_procedural_save_increments_usage(self, mem_store):
        skill = ProceduralSkill(
            name="inc_test", fragment="test", when_to_use="testing",
        )
        mem_store.procedural_save(skill, tenant_id="acme")
        mem_store.procedural_save(skill, tenant_id="acme")
        top = mem_store.procedural_top(5, tenant_id="acme")
        assert top[0].usage_count == 2

    def test_procedural_top_ordering(self, mem_store):
        mem_store.procedural_save(
            ProceduralSkill(name="low", fragment="a", score=1.0, usage_count=10),
            tenant_id="acme",
        )
        mem_store.procedural_save(
            ProceduralSkill(name="high", fragment="b", score=5.0, usage_count=1),
            tenant_id="acme",
        )
        top = mem_store.procedural_top(5, tenant_id="acme")
        assert top[0].name == "high"
        assert top[1].name == "low"


# -- Tenant isolation ------------------------------------------------------

class TestTenantIsolation:
    def test_semantic_isolation(self, mem_store):
        mem_store.semantic_write("Acme secret", thread_id="t1", tenant_id="acme")
        mem_store.semantic_write("Globex secret", thread_id="t1", tenant_id="globex")
        acme = mem_store.semantic_search("secret", top_k=5, tenant_id="acme")
        globex = mem_store.semantic_search("secret", top_k=5, tenant_id="globex")
        assert len(acme) == 1
        assert len(globex) == 1
        assert acme[0].text == "Acme secret"
        assert globex[0].text == "Globex secret"

    def test_episodic_isolation(self, mem_store):
        mem_store.episodic_upsert("Acme thread", thread_id="t1", tenant_id="acme")
        mem_store.episodic_upsert("Globex thread", thread_id="t1", tenant_id="globex")
        acme = mem_store.episodic_search("thread", top_k=5, tenant_id="acme")
        globex = mem_store.episodic_search("thread", top_k=5, tenant_id="globex")
        assert len(acme) == 1
        assert len(globex) == 1

    def test_procedural_isolation(self, mem_store):
        mem_store.procedural_save(
            ProceduralSkill(name="acme_skill", fragment="acme", when_to_use="acme"),
            tenant_id="acme",
        )
        mem_store.procedural_save(
            ProceduralSkill(name="globex_skill", fragment="globex", when_to_use="globex"),
            tenant_id="globex",
        )
        assert len(mem_store.procedural_top(10, tenant_id="acme")) == 1
        assert len(mem_store.procedural_top(10, tenant_id="globex")) == 1


# -- SqliteMemoryStore -----------------------------------------------------

class TestSqliteStore:
    def test_semantic_persistence(self, sqlite_store, embedder):
        sid = sqlite_store.semantic_write(
            "POL-001 cap is $50K", thread_id="t1", tenant_id="acme",
        )
        assert sid.startswith("sem-")
        results = sqlite_store.semantic_search("purchase cap", top_k=3, tenant_id="acme")
        assert len(results) == 1
        assert "$50K" in results[0].text

    def test_episodic_upsert(self, sqlite_store):
        sqlite_store.episodic_upsert("Thread about approvals", thread_id="t1",
                                     tenant_id="acme")
        sqlite_store.episodic_upsert("Updated thread about approvals", thread_id="t1",
                                     tenant_id="acme")
        results = sqlite_store.episodic_search("approvals", top_k=5, tenant_id="acme")
        assert len(results) == 1
        assert "Updated" in results[0].summary

    def test_procedural_save_and_top(self, sqlite_store):
        sqlite_store.procedural_save(
            ProceduralSkill(name="test_skill", fragment="do X", when_to_use="testing"),
            tenant_id="acme",
        )
        top = sqlite_store.procedural_top(5, tenant_id="acme")
        assert len(top) == 1
        assert top[0].name == "test_skill"

    def test_idempotent_reopen(self, embedder):
        tmp = tempfile.mkdtemp()
        db = str(Path(tmp) / "mem.db")
        s1 = SqliteMemoryStore(db, embedder)
        s1.semantic_write("fact one", thread_id="t1", tenant_id="acme")

        s2 = SqliteMemoryStore(db, embedder)
        results = s2.semantic_search("fact", top_k=3, tenant_id="acme")
        assert len(results) == 1
        assert results[0].text == "fact one"


# -- TemplateLLM.reflect() ------------------------------------------------

class TestReflection:
    def test_produces_semantic_facts(self):
        llm = TemplateLLM()
        result = llm.reflect(
            "Who approves purchases over $50K?",
            "According to POL-001, the VP of Finance must approve purchases "
            "exceeding $50,000. The CFO must be notified for amounts over $100,000.",
            "thread-1",
        )
        assert len(result.semantic_facts) >= 1
        assert any("VP of Finance" in f for f in result.semantic_facts)

    def test_produces_episodic_summary(self):
        llm = TemplateLLM()
        result = llm.reflect(
            "What is the liability cap for AcmeCloud?",
            "The AcmeCloud MSA caps liability at $500K.",
            "thread-2",
        )
        assert "What is the liability" in result.episodic_summary
        assert "AcmeCloud MSA" in result.episodic_summary

    def test_skills_empty_by_default(self):
        llm = TemplateLLM()
        result = llm.reflect("q?", "answer.", "t1")
        assert result.skills == []

    def test_deterministic(self):
        llm = TemplateLLM()
        r1 = llm.reflect("Who approves $60K?", "POL-001: VP approves over $50K.", "t1")
        r2 = llm.reflect("Who approves $60K?", "POL-001: VP approves over $50K.", "t1")
        assert r1.semantic_facts == r2.semantic_facts
        assert r1.episodic_summary == r2.episodic_summary


# -- Graph integration ----------------------------------------------------

class TestGraphWithMemory:
    def test_load_memory_populates_context(self, retriever, mem_store):
        mem_store.semantic_write(
            "POL-001: VP approves over $50K", thread_id="t0", tenant_id="default",
        )
        mem_store.episodic_upsert(
            "Past question about approval thresholds", thread_id="t0",
            tenant_id="default",
        )
        graph = build_counsel_graph(
            retriever, checkpointer=MemorySaver(), memory_store=mem_store,
            hitl_enabled=False,
        )
        result = graph.invoke(
            {"question": "who approves a $60,000 purchase?",
             "tenant_id": "default", "thread_id": "t1"},
            {"configurable": {"thread_id": "t1"}},
        )
        ctx = result.get("memory_context", "")
        assert "POL-001" in ctx
        assert "Known facts" in ctx
        assert "Related past conversations" in ctx

    def test_save_memory_persists_after_answer(self, retriever, mem_store):
        graph = build_counsel_graph(
            retriever, checkpointer=MemorySaver(), memory_store=mem_store,
            hitl_enabled=False,
        )
        result = graph.invoke(
            {"question": "who approves a $60,000 purchase?",
             "tenant_id": "default", "thread_id": "t2"},
            {"configurable": {"thread_id": "t2"}},
        )
        # If the answer was not refused, facts should have been persisted
        ans = result["answer"]
        if not ans.refused:
            facts = mem_store.semantic_search("approves", top_k=5, tenant_id="default")
            assert len(facts) > 0

    def test_graph_without_memory_backward_compat(self, retriever):
        """Graph compiled without memory_store must work exactly as before."""
        graph = build_counsel_graph(
            retriever, checkpointer=MemorySaver(), hitl_enabled=False,
        )
        result = graph.invoke(
            {"question": "who approves a $60,000 purchase?",
             "tenant_id": "default", "thread_id": "t1"},
            {"configurable": {"thread_id": "t1"}},
        )
        assert "answer" in result
        assert "memory_context" not in result  # no memory nodes added

    def test_memory_does_not_leak_across_tenants(self, retriever, embedder):
        store_acme = InMemoryMemoryStore(embedder)
        store_globex = InMemoryMemoryStore(embedder)

        store_acme.semantic_write("Acme fact", thread_id="t0", tenant_id="acme")
        store_globex.semantic_write("Globex fact", thread_id="t0", tenant_id="globex")

        g_acme = build_counsel_graph(
            retriever, checkpointer=MemorySaver(), memory_store=store_acme,
            hitl_enabled=False,
        )
        result = g_acme.invoke(
            {"question": "tell me about the fact", "tenant_id": "acme",
             "thread_id": "t1"},
            {"configurable": {"thread_id": "t1"}},
        )
        ctx = result.get("memory_context", "")
        assert "Acme" in ctx
        assert "Globex" not in ctx
