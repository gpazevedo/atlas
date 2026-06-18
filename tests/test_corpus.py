"""Tests for the corpus generator.

These guard the three properties the eval harness depends on:
  1. determinism      — same seed => identical corpus
  2. integrity        — every golden span id exists in some document
  3. hard cases       — the planted traps are actually present and tagged
"""

from __future__ import annotations

import pytest

from atlas_counsel.corpus import build_corpus
from atlas_counsel.corpus.models import AnswerType


def test_determinism():
    a = build_corpus(seed=42).model_dump_json()
    b = build_corpus(seed=42).model_dump_json()
    assert a == b


def test_referential_integrity_holds():
    # build_corpus would raise on construction if violated; assert explicitly.
    corpus = build_corpus()
    known = {s.span_id for d in corpus.documents for s in d.spans}
    for item in corpus.golden:
        assert set(item.supporting_span_ids) <= known


def test_span_ids_are_stable_and_unique():
    corpus = build_corpus()
    ids = [s.span_id for d in corpus.documents for s in d.spans]
    assert len(ids) == len(set(ids)), "span ids must be unique"
    assert "POL-001#S1" in ids


def test_unanswerable_item_has_no_spans():
    corpus = build_corpus()
    unanswerable = [g for g in corpus.golden if g.answer_type == AnswerType.UNANSWERABLE]
    assert unanswerable, "corpus must contain at least one unanswerable item"
    for item in unanswerable:
        assert item.supporting_span_ids == []


def test_contradiction_pair_present():
    """AcmeCloud (99.9%) vs NorthLink (99.5%) must both exist and disagree."""
    from atlas_counsel.corpus.models import DocCategory

    corpus = build_corpus()
    # Key on (vendor, category): a vendor can own both a contract and a log,
    # so vendor name alone does not identify a document.
    contracts = {
        d.vendor: d
        for d in corpus.documents
        if d.category == DocCategory.CONTRACT
    }
    acme = contracts["AcmeCloud"].spans[0].text
    north = contracts["NorthLink"].spans[0].text
    assert "99.9%" in acme and "99.5%" in north


def test_threshold_precision_trap_present():
    """POL-001 ($50k) and POL-002 ($25k) are surface-similar but distinct."""
    corpus = build_corpus()
    by_id = {d.doc_id: d for d in corpus.documents}
    assert "$50,000" in by_id["POL-001"].spans[1].text
    assert "$25,000" in by_id["POL-002"].spans[1].text


def test_golden_covers_each_answer_type():
    corpus = build_corpus()
    types = {g.answer_type for g in corpus.golden}
    assert AnswerType.GROUNDED in types
    assert AnswerType.UNANSWERABLE in types
    assert AnswerType.MULTI_HOP in types


def test_invalid_golden_rejected():
    """A grounded item with no spans must fail validation."""
    from atlas_counsel.corpus.models import GoldenItem

    with pytest.raises(ValueError):
        GoldenItem(
            qid="Q-999",
            question="x",
            answer_type=AnswerType.GROUNDED,
            supporting_span_ids=[],
            reference_answer="y",
        )
