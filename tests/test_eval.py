"""Eval harness tests.

Covers metric math, the grounding-based refusal decision (and why it isn't
score-based), and a regression gate on aggregate numbers so future changes
can't silently degrade quality.
"""

from __future__ import annotations

from atlas_counsel.chunking import chunk_corpus
from atlas_counsel.corpus import build_corpus
from atlas_counsel.embeddings import HashingEmbedder
from atlas_counsel.eval import evaluate
from atlas_counsel.eval.answerer import answer_from_chunks, grounding_overlap
from atlas_counsel.eval.metrics import (
    context_precision,
    context_recall,
    hit_at_k,
    reciprocal_rank,
)
from atlas_counsel.retrieval import InMemoryHybridRetriever


# --- metric math (exact, hand-checked) --------------------------------------

def test_hit_at_k():
    assert hit_at_k(["a", "b", "c"], {"b"}, k=3) == 1.0
    assert hit_at_k(["a", "b", "c"], {"z"}, k=3) == 0.0
    assert hit_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0  # c is at rank 3


def test_context_recall():
    assert context_recall(["a", "b"], {"a", "b"}, k=2) == 1.0
    assert context_recall(["a", "x"], {"a", "b"}, k=2) == 0.5


def test_context_precision_rewards_higher_ranks():
    # supporting span at rank 1 => AP 1.0
    assert context_precision(["a", "x", "y"], {"a"}, k=3) == 1.0
    # same span at rank 3 => AP 1/3
    assert round(context_precision(["x", "y", "a"], {"a"}, k=3), 4) == round(1 / 3, 4)


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "y"], {"a"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


# --- refusal is grounding-based, not score-based ----------------------------

def _retriever():
    r = InMemoryHybridRetriever(HashingEmbedder())
    r.index(chunk_corpus(build_corpus()))
    return r


def test_unanswerable_question_is_refused():
    r = _retriever()
    hits = r.search("What is our policy on accepting gifts from suppliers?", top_k=5)
    ans = answer_from_chunks(
        "What is our policy on accepting gifts from suppliers?", hits
    )
    assert ans.refused
    assert ans.citations == []


def test_answerable_question_is_not_refused():
    r = _retriever()
    q = "Above what value does a single-source purchase need justification?"
    hits = r.search(q, top_k=5)
    ans = answer_from_chunks(q, hits)
    assert not ans.refused
    assert ans.citations


def test_grounding_overlap_separates_answerable_from_not():
    r = _retriever()
    answerable = "Above what value does a single-source purchase need justification?"
    unanswerable = "What is our policy on accepting gifts from suppliers?"
    o_yes = grounding_overlap(answerable, r.search(answerable, 5), top_n=2)
    o_no = grounding_overlap(unanswerable, r.search(unanswerable, 5), top_n=2)
    assert o_yes > o_no
    # The whole point: the unanswerable one has materially lower grounding.
    assert o_no < 0.25 <= o_yes


# --- end-to-end regression gate ---------------------------------------------

def test_eval_regression_gate():
    """Locks in current offline-harness quality. If a future change drops
    these, the test fails loudly rather than degrading silently."""
    r = _retriever()
    report = evaluate(r, build_corpus().golden, label="ci", k=5)
    agg = report.aggregate

    assert agg["refusal_accuracy"] == 1.0, "must refuse exactly the unanswerable items"
    assert agg["hit_at_k"] >= 0.80
    assert agg["context_precision"] >= 0.80
    assert agg["reciprocal_rank"] >= 0.80
    assert agg["faithfulness"] >= 0.75


def test_per_tag_hard_cases_present_in_report():
    r = _retriever()
    report = evaluate(r, build_corpus().golden, label="ci", k=5)
    # The planted hard-case tags must surface as their own slices.
    for tag in ("threshold-precision", "contradiction", "unanswerable"):
        assert tag in report.by_tag
    # Unanswerable slice must show perfect refusal.
    assert report.by_tag["unanswerable"]["refusal_accuracy"] == 1.0
