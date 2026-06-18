"""Composed retrieval pipeline.

Wraps a base hybrid retriever with two optional advanced stages:

    query --(decompose?)--> sub-queries --retrieve each--> merge(dedupe)
          --(rerank?)--> top_k

It implements the same `search(query, top_k)` signature as the base retriever,
so it slots directly into the eval harness and any downstream caller. Each
stage is independently toggleable, which is exactly what lets the harness
attribute a metric change to a specific stage (ablation).

Merge strategy: when sub-queries are retrieved separately, a chunk may surface
from more than one. We keep its best (max) score and dedupe by span_id, so a
chunk relevant to multiple sub-queries isn't double-counted but also isn't
lost.
"""

from __future__ import annotations

from .decompose import HeuristicDecomposer, QueryDecomposer
from .rerank import Reranker
from .retrieval import RetrievedChunk, Retriever


class RetrievalPipeline:
    def __init__(
        self,
        base: Retriever,
        *,
        decomposer: QueryDecomposer | None = None,
        reranker: Reranker | None = None,
        fanout_k: int = 8,
    ) -> None:
        self._base = base
        self._decomposer = decomposer
        self._reranker = reranker
        # How many candidates to pull per sub-query before rerank/truncation.
        self._fanout_k = fanout_k

    def _retrieve_merged(self, query: str) -> list[RetrievedChunk]:
        subs = self._decomposer.decompose(query) if self._decomposer else [query]
        best: dict[str, RetrievedChunk] = {}
        for sub in subs:
            for rc in self._base.search(sub, top_k=self._fanout_k):
                cur = best.get(rc.chunk.span_id)
                if cur is None or rc.score > cur.score:
                    best[rc.chunk.span_id] = rc
        # Stable order: by score desc before any rerank.
        return sorted(best.values(), key=lambda rc: -rc.score)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        candidates = self._retrieve_merged(query)
        if self._reranker:
            return self._reranker.rerank(query, candidates, top_k=top_k)
        return candidates[:top_k]


def default_pipeline(base: Retriever, reranker: Reranker) -> RetrievalPipeline:
    """The full advanced-RAG pipeline: conditional decomposition + rerank."""
    return RetrievalPipeline(
        base, decomposer=HeuristicDecomposer(), reranker=reranker
    )
