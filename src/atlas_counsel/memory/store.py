"""Multi-tier memory stores.

Three memory types, each with a distinct access pattern:

  Semantic  — facts as NL strings; written explicitly, retrieved by similarity
  Episodic  — one rolling summary per thread; upserted, retrieved by similarity
  Procedural — learned prompt fragments with when_to_use cues; JIT retrieval

MemoryStore is the Protocol. Two implementations:

  InMemoryMemoryStore — dict-based, uses EmbeddingProvider for similarity (CI)
  SqliteMemoryStore   — SQLite with JSON embeddings, per-tenant databases (prod)
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
import uuid
from typing import Protocol

from pydantic import BaseModel, Field

from ..embeddings import EmbeddingProvider


# -- data models -----------------------------------------------------------

class SemanticRecord(BaseModel):
    id: str = Field(default_factory=lambda: f"sem-{uuid.uuid4().hex[:12]}")
    text: str
    thread_id: str
    tenant_id: str
    created_at: str = ""
    score: float = 0.0  # populated during search


class EpisodicEntry(BaseModel):
    summary: str
    thread_id: str
    tenant_id: str
    score: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    id: str = ""


class ProceduralSkill(BaseModel):
    name: str
    fragment: str
    when_to_use: str = ""
    usage_count: int = 0
    score: float = 0.0
    created_at: str = ""


class ReflectionResult(BaseModel):
    semantic_facts: list[str] = Field(default_factory=list)
    episodic_summary: str = ""
    skills: list[ProceduralSkill] = Field(default_factory=list)


# -- protocol --------------------------------------------------------------

class MemoryStore(Protocol):
    def semantic_write(self, text: str, *, thread_id: str,
                       tenant_id: str) -> str: ...
    def semantic_search(self, query: str, *, top_k: int,
                        tenant_id: str) -> list[SemanticRecord]: ...
    def episodic_upsert(self, summary: str, *, thread_id: str,
                        tenant_id: str, score: float = 0.0) -> None: ...
    def episodic_search(self, query: str, *, top_k: int,
                        tenant_id: str) -> list[EpisodicEntry]: ...
    def procedural_save(self, skill: ProceduralSkill, *,
                        tenant_id: str) -> None: ...
    def procedural_search(self, query: str, *, top_k: int,
                          tenant_id: str) -> list[ProceduralSkill]: ...
    def procedural_top(self, n: int, *, tenant_id: str) -> list[ProceduralSkill]: ...


# -- helpers ---------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# -- in-memory store -------------------------------------------------------

class InMemoryMemoryStore:
    """Dict-based store for CI and offline testing.

    Uses an EmbeddingProvider (e.g. HashingEmbedder) for similarity search so
    retrieval is deterministic and dependency-free.
    """

    def __init__(self, embedder: EmbeddingProvider) -> None:
        self._embedder = embedder
        self._semantic: dict[str, dict[str, SemanticRecord]] = {}  # tenant -> id -> record
        self._episodic: dict[str, dict[str, EpisodicEntry]] = {}   # tenant -> thread_id -> entry
        self._procedural: dict[str, dict[str, ProceduralSkill]] = {}  # tenant -> name -> skill

    # -- semantic ----------------------------------------------------------

    def semantic_write(self, text: str, *, thread_id: str,
                       tenant_id: str) -> str:
        record = SemanticRecord(
            text=text, thread_id=thread_id, tenant_id=tenant_id,
            created_at=_now_iso(),
        )
        if tenant_id not in self._semantic:
            self._semantic[tenant_id] = {}
        self._semantic[tenant_id][record.id] = record
        return record.id

    def semantic_search(self, query: str, *, top_k: int = 3,
                        tenant_id: str = "default") -> list[SemanticRecord]:
        records = list(self._semantic.get(tenant_id, {}).values())
        if not records:
            return []
        q = self._embedder.embed([query])[0]
        scored: list[tuple[float, SemanticRecord]] = []
        for r in records:
            e = self._embedder.embed([r.text])[0]
            s = _cosine(q.dense, e.dense)
            scored.append((s, r))
        scored.sort(key=lambda x: -x[0])
        results: list[SemanticRecord] = []
        for s, r in scored[:top_k]:
            r.score = s
            results.append(r)
        return results

    # -- episodic ----------------------------------------------------------

    def episodic_upsert(self, summary: str, *, thread_id: str,
                        tenant_id: str, score: float = 0.0) -> None:
        if tenant_id not in self._episodic:
            self._episodic[tenant_id] = {}
        now = _now_iso()
        existing = self._episodic[tenant_id].get(thread_id)
        entry = EpisodicEntry(
            summary=summary, thread_id=thread_id, tenant_id=tenant_id,
            score=score,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            id=existing.id if existing else f"ep-{uuid.uuid4().hex[:12]}",
        )
        self._episodic[tenant_id][thread_id] = entry

    def episodic_search(self, query: str, *, top_k: int = 3,
                        tenant_id: str = "default") -> list[EpisodicEntry]:
        entries = list(self._episodic.get(tenant_id, {}).values())
        if not entries:
            return []
        q = self._embedder.embed([query])[0]
        scored: list[tuple[float, EpisodicEntry]] = []
        for e in entries:
            emb = self._embedder.embed([e.summary])[0]
            s = _cosine(q.dense, emb.dense)
            scored.append((s, e))
        scored.sort(key=lambda x: -x[0])
        return [s[1] for s in scored[:top_k]]

    # -- procedural --------------------------------------------------------

    def procedural_save(self, skill: ProceduralSkill, *,
                        tenant_id: str) -> None:
        if tenant_id not in self._procedural:
            self._procedural[tenant_id] = {}
        existing = self._procedural[tenant_id].get(skill.name)
        if existing:
            skill.usage_count = existing.usage_count + 1
            if not skill.created_at:
                skill.created_at = existing.created_at
        else:
            skill.usage_count = max(skill.usage_count, 1)
            if not skill.created_at:
                skill.created_at = _now_iso()
        self._procedural[tenant_id][skill.name] = skill

    def procedural_search(self, query: str, *, top_k: int = 3,
                          tenant_id: str = "default") -> list[ProceduralSkill]:
        skills = list(self._procedural.get(tenant_id, {}).values())
        if not skills:
            return []
        q = self._embedder.embed([query])[0]
        scored: list[tuple[float, ProceduralSkill]] = []
        for s in skills:
            cue = s.when_to_use or s.name
            e = self._embedder.embed([cue])[0]
            sim = _cosine(q.dense, e.dense)
            scored.append((sim, s))
        scored.sort(key=lambda x: -x[0])
        return [s[1] for s in scored[:top_k]]

    def procedural_top(self, n: int = 5, *,
                       tenant_id: str = "default") -> list[ProceduralSkill]:
        skills = list(self._procedural.get(tenant_id, {}).values())
        skills.sort(key=lambda s: (s.score, s.usage_count), reverse=True)
        return skills[:n]


# -- SQLite store ----------------------------------------------------------

class SqliteMemoryStore:
    """SQLite-backed memory store for production.

    Each tenant gets its own tables in a shared SQLite database (or per-tenant
    database when used with TenantRegistry). Embeddings are stored as JSON arrays
    and similarity is computed in Python at query time — sufficient for per-tenant
    scale (hundreds, not millions of entries).
    """

    def __init__(self, db_path: str, embedder: EmbeddingProvider) -> None:
        self._embedder = embedder
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SQLITE_SCHEMA)
        self._conn.commit()

    # -- semantic ----------------------------------------------------------

    def semantic_write(self, text: str, *, thread_id: str,
                       tenant_id: str) -> str:
        rid = f"sem-{uuid.uuid4().hex[:12]}"
        now = _now_iso()
        emb = self._embedder.embed([text])[0]
        self._conn.execute(
            "INSERT INTO memory_semantic (id, text, thread_id, tenant_id, "
            "created_at, dense_embedding) VALUES (?, ?, ?, ?, ?, ?)",
            (rid, text, thread_id, tenant_id, now, json.dumps(emb.dense)),
        )
        self._conn.commit()
        return rid

    def semantic_search(self, query: str, *, top_k: int = 3,
                        tenant_id: str = "default") -> list[SemanticRecord]:
        q = self._embedder.embed([query])[0]
        rows = self._conn.execute(
            "SELECT id, text, thread_id, tenant_id, created_at, dense_embedding "
            "FROM memory_semantic WHERE tenant_id = ?", (tenant_id,)
        ).fetchall()
        if not rows:
            return []
        scored: list[tuple[float, SemanticRecord]] = []
        for row in rows:
            emb = json.loads(row["dense_embedding"])
            s = _cosine(q.dense, emb)
            scored.append((s, SemanticRecord(
                id=row["id"], text=row["text"], thread_id=row["thread_id"],
                tenant_id=row["tenant_id"], created_at=row["created_at"],
            )))
        scored.sort(key=lambda x: -x[0])
        results: list[SemanticRecord] = []
        for s, r in scored[:top_k]:
            r.score = s
            results.append(r)
        return results

    # -- episodic ----------------------------------------------------------

    def episodic_upsert(self, summary: str, *, thread_id: str,
                        tenant_id: str, score: float = 0.0) -> None:
        now = _now_iso()
        emb = self._embedder.embed([summary])[0]
        existing = self._conn.execute(
            "SELECT id, created_at FROM memory_episodic "
            "WHERE thread_id = ? AND tenant_id = ?",
            (thread_id, tenant_id),
        ).fetchone()
        if existing:
            self._conn.execute(
                "UPDATE memory_episodic SET summary = ?, score = ?, updated_at = ?, "
                "dense_embedding = ? WHERE thread_id = ? AND tenant_id = ?",
                (summary, score, now, json.dumps(emb.dense), thread_id, tenant_id),
            )
        else:
            rid = f"ep-{uuid.uuid4().hex[:12]}"
            self._conn.execute(
                "INSERT INTO memory_episodic (id, summary, thread_id, tenant_id, "
                "score, created_at, updated_at, dense_embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (rid, summary, thread_id, tenant_id, score, now, now,
                 json.dumps(emb.dense)),
            )
        self._conn.commit()

    def episodic_search(self, query: str, *, top_k: int = 3,
                        tenant_id: str = "default") -> list[EpisodicEntry]:
        q = self._embedder.embed([query])[0]
        rows = self._conn.execute(
            "SELECT id, summary, thread_id, tenant_id, score, created_at, "
            "updated_at, dense_embedding FROM memory_episodic "
            "WHERE tenant_id = ?", (tenant_id,)
        ).fetchall()
        if not rows:
            return []
        scored: list[tuple[float, EpisodicEntry]] = []
        for row in rows:
            emb = json.loads(row["dense_embedding"])
            s = _cosine(q.dense, emb)
            scored.append((s, EpisodicEntry(
                id=row["id"], summary=row["summary"], thread_id=row["thread_id"],
                tenant_id=row["tenant_id"], score=row["score"],
                created_at=row["created_at"], updated_at=row["updated_at"],
            )))
        scored.sort(key=lambda x: -x[0])
        return [s[1] for s in scored[:top_k]]

    # -- procedural --------------------------------------------------------

    def procedural_save(self, skill: ProceduralSkill, *,
                        tenant_id: str) -> None:
        now = _now_iso()
        existing = self._conn.execute(
            "SELECT usage_count, created_at FROM memory_procedural "
            "WHERE name = ? AND tenant_id = ?", (skill.name, tenant_id)
        ).fetchone()
        cue = skill.when_to_use or skill.name
        emb = self._embedder.embed([cue])[0]
        emb_json = json.dumps(emb.dense)
        if existing:
            self._conn.execute(
                "UPDATE memory_procedural SET fragment = ?, when_to_use = ?, "
                "usage_count = ?, score = ?, dense_embedding = ? "
                "WHERE name = ? AND tenant_id = ?",
                (skill.fragment, skill.when_to_use,
                 existing["usage_count"] + 1, skill.score,
                 emb_json, skill.name, tenant_id),
            )
        else:
            self._conn.execute(
                "INSERT INTO memory_procedural (name, fragment, when_to_use, "
                "usage_count, score, created_at, tenant_id, dense_embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (skill.name, skill.fragment, skill.when_to_use,
                 max(skill.usage_count, 1), skill.score, now, tenant_id, emb_json),
            )
        self._conn.commit()

    def procedural_search(self, query: str, *, top_k: int = 3,
                          tenant_id: str = "default") -> list[ProceduralSkill]:
        q = self._embedder.embed([query])[0]
        rows = self._conn.execute(
            "SELECT name, fragment, when_to_use, usage_count, score, created_at, "
            "dense_embedding FROM memory_procedural WHERE tenant_id = ?",
            (tenant_id,)
        ).fetchall()
        if not rows:
            return []
        scored: list[tuple[float, ProceduralSkill]] = []
        for row in rows:
            emb = json.loads(row["dense_embedding"])
            s = _cosine(q.dense, emb)
            scored.append((s, ProceduralSkill(
                name=row["name"], fragment=row["fragment"],
                when_to_use=row["when_to_use"], usage_count=row["usage_count"],
                score=row["score"], created_at=row["created_at"],
            )))
        scored.sort(key=lambda x: -x[0])
        return [s[1] for s in scored[:top_k]]

    def procedural_top(self, n: int = 5, *,
                       tenant_id: str = "default") -> list[ProceduralSkill]:
        rows = self._conn.execute(
            "SELECT name, fragment, when_to_use, usage_count, score, created_at "
            "FROM memory_procedural WHERE tenant_id = ? "
            "ORDER BY score DESC, usage_count DESC LIMIT ?",
            (tenant_id, n),
        ).fetchall()
        return [ProceduralSkill(
            name=row["name"], fragment=row["fragment"],
            when_to_use=row["when_to_use"], usage_count=row["usage_count"],
            score=row["score"], created_at=row["created_at"],
        ) for row in rows]


_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_semantic (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT '',
    dense_embedding TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS memory_episodic (
    id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    dense_embedding TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS memory_procedural (
    name TEXT NOT NULL,
    fragment TEXT NOT NULL,
    when_to_use TEXT NOT NULL DEFAULT '',
    usage_count INTEGER NOT NULL DEFAULT 0,
    score REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT '',
    tenant_id TEXT NOT NULL,
    dense_embedding TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (name, tenant_id)
);

CREATE INDEX IF NOT EXISTS idx_semantic_tenant ON memory_semantic(tenant_id);
CREATE INDEX IF NOT EXISTS idx_episodic_tenant ON memory_episodic(tenant_id);
CREATE INDEX IF NOT EXISTS idx_episodic_thread ON memory_episodic(thread_id, tenant_id);
CREATE INDEX IF NOT EXISTS idx_procedural_tenant ON memory_procedural(tenant_id);
"""
