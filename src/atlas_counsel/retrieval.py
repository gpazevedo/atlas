"""Retrieval layer.

`Retriever` is the protocol the rest of the system codes against. Two
implementations:

  * `InMemoryHybridRetriever` — real dense + sparse scoring with RRF fusion,
    no external services. Used by tests and as a reference for the fusion
    semantics. Runs anywhere.
  * `QdrantHybridRetriever` (qdrant_store.py) — the production path using
    Qdrant's native named vectors + Query API server-side fusion. Needs a
    running Qdrant; covered by an integration test you run locally.

Both return `RetrievedChunk`s carrying the `span_id`, so downstream citation
checking is identical regardless of backend.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from .chunking import Chunk
from .embeddings import Embedding, EmbeddingProvider, SparseVector


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    # Per-channel ranks (1-based) before fusion, for observability/eval.
    dense_rank: int | None = None
    sparse_rank: int | None = None


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sparse_dot(a: SparseVector, b: SparseVector) -> float:
    bmap = dict(zip(b.indices, b.values))
    return sum(v * bmap.get(i, 0.0) for i, v in zip(a.indices, a.values))


def reciprocal_rank_fusion(
    dense_order: list[str],
    sparse_order: list[str],
    k: int = 60,
) -> dict[str, float]:
    """RRF: score(d) = sum over channels of 1/(k + rank_d).

    Rank-based, so it fuses the dense (cosine) and sparse (lexical) channels
    without needing their raw scores to be on the same scale — which they
    never are. k=60 is the standard dampening constant.
    """
    scores: dict[str, float] = {}
    for order in (dense_order, sparse_order):
        for rank, chunk_id in enumerate(order, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]: ...


class InMemoryHybridRetriever:
    """Reference hybrid retriever: dense cosine + sparse dot, fused with RRF."""

    def __init__(self, provider: EmbeddingProvider, rrf_k: int = 60) -> None:
        self._provider = provider
        self._rrf_k = rrf_k
        self._chunks: list[Chunk] = []
        self._embs: list[Embedding] = []

    @property
    def collection_name(self) -> str:
        # Same naming rule as the Qdrant backend: space_id is baked in so two
        # embedding spaces can never share an index.
        return f"counsel_{self._provider.space_id}"

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._embs = self._provider.embed([c.embed_text for c in chunks])

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        q = self._provider.embed([query])[0]

        dense_scores = [(_dot(q.dense, e.dense), i) for i, e in enumerate(self._embs)]
        sparse_scores = [(_sparse_dot(q.sparse, e.sparse), i) for i, e in enumerate(self._embs)]

        # Order each channel best-first.
        dense_ranked = [i for _, i in sorted(dense_scores, key=lambda x: -x[0])]
        sparse_ranked = [i for _, i in sorted(sparse_scores, key=lambda x: -x[0])]

        dense_rank = {idx: r for r, idx in enumerate(dense_ranked, 1)}
        sparse_rank = {idx: r for r, idx in enumerate(sparse_ranked, 1)}

        fused = reciprocal_rank_fusion(
            [self._chunks[i].chunk_id for i in dense_ranked],
            [self._chunks[i].chunk_id for i in sparse_ranked],
            k=self._rrf_k,
        )

        by_id = {c.chunk_id: idx for idx, c in enumerate(self._chunks)}
        ranked_ids = sorted(fused, key=lambda cid: -fused[cid])[:top_k]

        results: list[RetrievedChunk] = []
        for cid in ranked_ids:
            idx = by_id[cid]
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[idx],
                    score=fused[cid],
                    dense_rank=dense_rank.get(idx),
                    sparse_rank=sparse_rank.get(idx),
                )
            )
        return results
