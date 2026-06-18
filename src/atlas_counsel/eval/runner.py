"""Eval runner.

Scores a retriever+answerer+judge against the golden set and aggregates.

Scoring is answer-type aware:

  GROUNDED / MULTI_HOP
     retrieval metrics (hit@k, recall, precision, MRR) vs supporting spans
     + judge metrics (faithfulness, relevancy)
     + wrongly_refused penalty (should NOT refuse)

  UNANSWERABLE
     refusal is the metric: correct iff the system refused. Retrieval/judge
     metrics are not applicable (no ground-truth spans).

Aggregation is sliced by golden tag so you can read, e.g., context_precision
on the 'threshold-precision' items separately from the rest — proving the
planted hard cases are handled, not just the easy ones.

Run twice with two retrievers (two embedding providers) to get the A/B table.
"""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel

from ..corpus.models import AnswerType, GoldenItem
from ..retrieval import Retriever
from .answerer import answer_from_chunks
from .judge import HeuristicJudge, LLMJudge
from .metrics import (
    context_precision,
    context_recall,
    hit_at_k,
    reciprocal_rank,
)


class ItemResult(BaseModel):
    qid: str
    answer_type: AnswerType
    tags: list[str]
    refused: bool
    # retrieval metrics (None for unanswerable)
    hit_at_k: float | None = None
    context_recall: float | None = None
    context_precision: float | None = None
    reciprocal_rank: float | None = None
    # judge metrics (None for unanswerable / refused)
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    # correctness of the refusal decision (always set)
    refusal_correct: bool


class EvalReport(BaseModel):
    label: str
    k: int
    n_items: int
    items: list[ItemResult]
    aggregate: dict[str, float]
    by_tag: dict[str, dict[str, float]]


def _safe_mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def evaluate(
    retriever: Retriever,
    golden: list[GoldenItem],
    *,
    label: str,
    k: int = 5,
    judge: LLMJudge | None = None,
) -> EvalReport:
    judge = judge or HeuristicJudge()
    items: list[ItemResult] = []

    for g in golden:
        retrieved = retriever.search(g.question, top_k=k)
        retrieved_ids = [rc.chunk.span_id for rc in retrieved]
        ans = answer_from_chunks(g.question, retrieved)
        supporting = set(g.supporting_span_ids)

        if g.answer_type == AnswerType.UNANSWERABLE:
            # The only correct behavior is to refuse.
            items.append(ItemResult(
                qid=g.qid, answer_type=g.answer_type, tags=g.tags,
                refused=ans.refused,
                refusal_correct=ans.refused,
            ))
            continue

        # Answerable: should NOT refuse; score retrieval + answer quality.
        refusal_correct = not ans.refused
        jr = None
        if not ans.refused:
            jr = judge.judge(g.question, ans.text,
                             [rc.chunk.text for rc in retrieved])

        items.append(ItemResult(
            qid=g.qid, answer_type=g.answer_type, tags=g.tags,
            refused=ans.refused,
            hit_at_k=hit_at_k(retrieved_ids, supporting, k),
            context_recall=context_recall(retrieved_ids, supporting, k),
            context_precision=context_precision(retrieved_ids, supporting, k),
            reciprocal_rank=reciprocal_rank(retrieved_ids, supporting),
            faithfulness=jr.faithfulness if jr else None,
            answer_relevancy=jr.answer_relevancy if jr else None,
            refusal_correct=refusal_correct,
        ))

    aggregate = _aggregate(items)
    by_tag = _aggregate_by_tag(items)
    return EvalReport(
        label=label, k=k, n_items=len(items),
        items=items, aggregate=aggregate, by_tag=by_tag,
    )


_RETRIEVAL_FIELDS = ("hit_at_k", "context_recall", "context_precision", "reciprocal_rank")
_JUDGE_FIELDS = ("faithfulness", "answer_relevancy")


def _aggregate(items: list[ItemResult]) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in _RETRIEVAL_FIELDS + _JUDGE_FIELDS:
        vals = [getattr(it, f) for it in items if getattr(it, f) is not None]
        out[f] = _safe_mean(vals)
    # Refusal correctness over ALL items (answerable + unanswerable).
    out["refusal_accuracy"] = _safe_mean([1.0 if it.refusal_correct else 0.0 for it in items])
    return out


def _aggregate_by_tag(items: list[ItemResult]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[ItemResult]] = defaultdict(list)
    for it in items:
        for tag in it.tags:
            buckets[tag].append(it)
    return {tag: _aggregate(group) for tag, group in sorted(buckets.items())}
