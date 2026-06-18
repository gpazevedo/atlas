"""Retrieval metrics.

These need no LLM: they score the ranked span ids a retriever returned
against the known `supporting_span_ids` of a golden item. Because the golden
set carries ground-truth spans, every number here is exact and reproducible.

Definitions (all over the retrieved ranked list, best-first):

  hit@k            1.0 if any supporting span appears in the top-k, else 0.0
  context_recall   fraction of supporting spans found in the top-k
  context_precision (AP-style) precision averaged at each rank where a
                    supporting span occurs — rewards ranking the right spans
                    higher, not just including them
  reciprocal_rank  1 / rank of the first supporting span (0 if none)

For UNANSWERABLE items there are no supporting spans, so these metrics do not
apply; the runner scores those items on the refusal dimension instead.
"""

from __future__ import annotations


def hit_at_k(retrieved_ids: list[str], supporting: set[str], k: int) -> float:
    return 1.0 if set(retrieved_ids[:k]) & supporting else 0.0


def context_recall(retrieved_ids: list[str], supporting: set[str], k: int) -> float:
    if not supporting:
        return 0.0
    found = set(retrieved_ids[:k]) & supporting
    return len(found) / len(supporting)


def context_precision(retrieved_ids: list[str], supporting: set[str], k: int) -> float:
    """Average precision: mean of precision@i over the ranks i (1..k) where a
    supporting span is retrieved. 0 if none retrieved."""
    if not supporting:
        return 0.0
    hits = 0
    summed = 0.0
    for i, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in supporting:
            hits += 1
            summed += hits / i
    return summed / len(supporting) if hits else 0.0


def reciprocal_rank(retrieved_ids: list[str], supporting: set[str]) -> float:
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in supporting:
            return 1.0 / i
    return 0.0
