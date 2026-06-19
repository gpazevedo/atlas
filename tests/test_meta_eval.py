"""Meta-evaluation tests: perturbation logic and end-to-end bias report."""

from __future__ import annotations

from atlas_counsel.chunking import chunk_corpus
from atlas_counsel.corpus import build_corpus
from atlas_counsel.embeddings import HashingEmbedder
from atlas_counsel.eval import evaluate, meta_evaluate
from atlas_counsel.eval.judge import HeuristicJudge
from atlas_counsel.eval.meta_eval import (
    _CONFIDENCE_MARKERS,
    _CONNECTORS,
    _HEDGING_MARKERS,
    perturb_confidence,
    perturb_fluency,
)
from atlas_counsel.retrieval import InMemoryHybridRetriever


_SAMPLE = (
    "POL-001#S1 states that purchases above $50,000 require VP approval. "
    "The policy also covers single-source justifications."
)


# --- perturbation unit tests --------------------------------------------------

def test_perturb_fluency_high_adds_connectors():
    result = perturb_fluency(_SAMPLE, mode="high")
    assert any(conn in result for conn in _CONNECTORS)


def test_perturb_fluency_low_strips_connectors():
    # Inject connectors first, then strip them.
    injected = perturb_fluency(_SAMPLE, mode="high")
    stripped = perturb_fluency(injected, mode="low")
    for conn in _CONNECTORS:
        assert conn not in stripped


def test_perturb_confidence_high_adds_markers():
    result = perturb_confidence(_SAMPLE, mode="high")
    assert "clearly" in result.lower()


def test_perturb_confidence_low_adds_hedging():
    result = perturb_confidence(_SAMPLE, mode="low")
    assert "appears" in result.lower()


def test_perturbations_preserve_span_ids():
    text = "According to POL-001#S1, single-source purchases need VP approval."
    assert "POL-001#S1" in perturb_fluency(text, mode="high")
    assert "POL-001#S1" in perturb_fluency(text, mode="low")
    assert "POL-001#S1" in perturb_confidence(text, mode="high")
    assert "POL-001#S1" in perturb_confidence(text, mode="low")


# --- end-to-end ---------------------------------------------------------------

def test_meta_evaluate_returns_valid_bias_report():
    corpus = build_corpus()
    retriever = InMemoryHybridRetriever(HashingEmbedder())
    retriever.index(chunk_corpus(corpus))
    report = evaluate(retriever, corpus.golden, label="ci", k=5)
    bias = meta_evaluate(HeuristicJudge(), retriever, corpus.golden, report)

    # HeuristicJudge is token-overlap based — perturbation adds/removes
    # surface tokens (connectors, markers) that aren't in context, so
    # faithfulness ratios shift modestly. Relevancy (answer vs question)
    # should stay near zero since the question doesn't change.
    assert abs(bias.fluency_delta_faithfulness) <= 0.25
    assert abs(bias.fluency_delta_relevancy) <= 0.05
    assert abs(bias.confidence_delta_faithfulness) <= 0.05
    assert abs(bias.confidence_delta_relevancy) <= 0.05

    # Correlations are bounded in [-1, 1].
    assert -1.0 <= bias.retrieval_faithfulness_corr <= 1.0
    assert -1.0 <= bias.retrieval_relevancy_corr <= 1.0

    assert bias.n_pairs >= 1
