"""Tests for rerank + decomposition.

The corpus-level ablation shows the offline proxies don't beat a near-saturated
first stage (that's reported honestly, not hidden). These tests instead prove
each stage is *mechanically correct* on constructed inputs where the effect is
unambiguous — and that decomposition never harms simple queries.
"""

from __future__ import annotations

from atlas_counsel.chunking import Chunk
from atlas_counsel.corpus.models import DocCategory
from atlas_counsel.decompose import HeuristicDecomposer
from atlas_counsel.rerank import TokenInteractionReranker
from atlas_counsel.retrieval import RetrievedChunk


def _rc(span_id: str, text: str, score: float, title: str = "T") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=span_id, span_id=span_id, doc_id=span_id.split("#")[0],
            category=DocCategory.POLICY, title=title, text=text,
        ),
        score=score,
    )


# --- reranker mechanics ------------------------------------------------------

def test_rerank_promotes_phrase_match_over_scattered_words():
    """Two candidates with the SAME query-term coverage, but one reproduces the
    query's exact phrasing and the other scatters the words. The phrase match
    should win — that adjacency signal is precisely what a first-stage
    bag-of-words retriever cannot see."""
    query = "net 45 days"
    cands = [
        # scattered: has 'net', '45', 'days' but never adjacent
        _rc("A#S0", "Net amounts differ. We counted 45 items over many days.", 0.9),
        # exact phrase
        _rc("B#S0", "Invoices are payable net 45 days from receipt.", 0.1),
    ]
    out = TokenInteractionReranker().rerank(query, cands, top_k=2)
    assert out[0].chunk.span_id == "B#S0"


def test_rerank_is_stable_and_truncates():
    query = "liability cap"
    cands = [
        _rc("A#S0", "Liability cap is twelve months of fees.", 0.5),
        _rc("B#S0", "Unrelated termination clause text.", 0.4),
        _rc("C#S0", "Another unrelated payment clause.", 0.3),
    ]
    out = TokenInteractionReranker().rerank(query, cands, top_k=1)
    assert len(out) == 1
    assert out[0].chunk.span_id == "A#S0"


def test_rerank_empty_input():
    assert TokenInteractionReranker().rerank("q", [], top_k=5) == []


# --- decomposition mechanics -------------------------------------------------

def test_decompose_splits_comparison_over_two_vendors():
    d = HeuristicDecomposer()
    subs = d.decompose("Compare the uptime guarantees of AcmeCloud and NorthLink")
    assert len(subs) == 2
    assert any("AcmeCloud" in s for s in subs)
    assert any("NorthLink" in s for s in subs)
    # aspect carried through, junk stripped
    assert all("compare" not in s.lower() for s in subs)


def test_decompose_leaves_simple_query_untouched():
    d = HeuristicDecomposer()
    q = "What are NorthLink's payment terms?"
    assert d.decompose(q) == [q]


def test_decompose_requires_both_entities_and_cue():
    d = HeuristicDecomposer()
    # two entities but no comparison cue -> no split
    assert d.decompose("AcmeCloud signed before NorthLink was contacted") == [
        "AcmeCloud signed before NorthLink was contacted"
    ]
    # one entity with a cue -> no split
    assert len(d.decompose("Compare AcmeCloud pricing tiers")) == 1


def test_decompose_aspect_has_no_trailing_prepositions():
    d = HeuristicDecomposer()
    subs = d.decompose("Compare the liability caps of AcmeCloud and NorthLink")
    for s in subs:
        assert not s.lower().strip().endswith(" of")
        assert not s.lower().strip().endswith(" the")
