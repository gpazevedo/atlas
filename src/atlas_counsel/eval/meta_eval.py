"""LLM-as-Judge meta-evaluation.

Detects three judge bias dimensions without requiring an external LLM:

1. Fluency bias — does the judge overrate well-written answers?
2. Confidence formatting bias — is the judge swayed by confident phrasing?
3. Retrieval-judge correlation — do judge scores track retrieval quality?

Approach: create perturbed answer variants (fluent/choppy, confident/hedging)
that preserve factual content but change surface features. A biased judge shows
large score deltas between variants. A fair judge shows near-zero deltas.

The HeuristicJudge (token overlap) should show ~0 deltas — proving perturbation
preserves content. A real LLM judge would reveal actual bias.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..corpus.models import AnswerType, GoldenItem
from ..retrieval import Retriever
from .answerer import answer_from_chunks
from .judge import LLMJudge
from .runner import EvalReport

# Words/phrases that signal fluency or lack thereof.
_CONNECTORS = (
    "Furthermore", "Moreover", "Additionally", "In addition", "Specifically",
    "In particular", "Notably",
)
# Words/phrases that signal confidence or hedging.
_CONFIDENCE_MARKERS = (
    "clearly", "definitively", "absolutely", "undoubtedly", "without question",
    "it is established that",
)
_HEDGING_MARKERS = (
    "might", "may", "could", "seems to", "appears to", "possibly", "perhaps",
    "it is possible that", "I believe", "likely",
)


def perturb_fluency(answer: str, *, mode: str = "high") -> str:
    """Adjust surface fluency while preserving factual content and span refs.

    mode="high": adds connectors and smooth transitions.
    mode="low":  strips connectors, shortens sentences.
    """
    if mode == "high":
        return _fluency_high(answer)
    return _fluency_low(answer)


def perturb_confidence(answer: str, *, mode: str = "high") -> str:
    """Adjust confidence tone while preserving factual content and span refs.

    mode="high": adds conviction markers.
    mode="low":  adds hedging markers.
    """
    if mode == "high":
        return _confidence_high(answer)
    return _confidence_low(answer)


# -- internal perturbation helpers --------------------------------------------

def _fluency_high(text: str) -> str:
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return "Based on the available information, " + text[0].lower() + text[1:]
    result = [sentences[0]]
    connectors = list(_CONNECTORS)
    for i, s in enumerate(sentences[1:], start=1):
        conn = connectors[(i - 1) % len(connectors)]
        result.append(f"{conn}, {s[0].lower() + s[1:] if s[0].isupper() else s}")
    return ". ".join(result)


def _fluency_low(text: str) -> str:
    # Remove connectors and transition phrases, simplify sentence starts.
    for conn in _CONNECTORS:
        text = re.sub(rf"\b{conn},\s*", "", text)
    # Collapse multiple spaces.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _confidence_high(text: str) -> str:
    marker = _CONFIDENCE_MARKERS[0]
    sentences = _split_sentences(text)
    if not sentences:
        return text
    # Add a confidence prefix to the first sentence.
    sentences[0] = f"It is {marker} that " + sentences[0][0].lower() + sentences[0][1:]
    # Replace hedging with confidence markers in later sentences.
    for i in range(1, len(sentences)):
        for h in _HEDGING_MARKERS:
            sentences[i] = re.sub(rf"\b{re.escape(h)}\b", marker, sentences[i],
                                  flags=re.IGNORECASE)
    return ". ".join(sentences)


def _confidence_low(text: str) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return text
    # Add a hedging prefix to the first sentence.
    sentences[0] = "It appears that " + sentences[0][0].lower() + sentences[0][1:]
    # Add hedging to confident assertions.
    for i in range(1, len(sentences)):
        for c in _CONFIDENCE_MARKERS:
            sentences[i] = re.sub(rf"\b{re.escape(c)}\b", "likely", sentences[i],
                                  flags=re.IGNORECASE)
    return ". ".join(sentences)


def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries while keeping span refs like POL-001#S1 intact."""
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


# -- Spearman rank correlation (pure Python, no scipy dependency) --------------

def spearman_rank(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation coefficient. Returns 0.0 if < 3 pairs."""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return 0.0
    rank_x = _ranks(xs)
    rank_y = _ranks(ys)
    d2_sum = sum((rx - ry) ** 2 for rx, ry in zip(rank_x, rank_y))
    rho = 1.0 - (6.0 * d2_sum) / (n * (n * n - 1))
    return round(rho, 4)


def _ranks(vals: list[float]) -> list[float]:
    """Return ranks (1-based, average for ties) for a list of values."""
    indexed = sorted(enumerate(vals), key=lambda x: x[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = (i + j + 1) / 2.0  # 1-based average rank
        for k in range(i, j):
            ranks[indexed[k][0]] = avg
        i = j
    return ranks


# -- bias report ---------------------------------------------------------------

@dataclass
class BiasReport:
    fluency_delta_faithfulness: float
    fluency_delta_relevancy: float
    confidence_delta_faithfulness: float
    confidence_delta_relevancy: float
    retrieval_faithfulness_corr: float
    retrieval_relevancy_corr: float
    n_pairs: int


def meta_evaluate(
    judge: LLMJudge,
    retriever: Retriever,
    golden: list[GoldenItem],
    report: EvalReport,
) -> BiasReport:
    """Run perturbation-based bias tests and retrieval-judge correlation.

    Generates answers for each answerable golden item, creates perturbed
    variants, and measures judge score deltas. Uses the EvalReport for
    retrieval-judge correlation scores.
    """
    answerable = [g for g in golden if g.answer_type != AnswerType.UNANSWERABLE]
    n_pairs = len(answerable)

    # Context for judge calls: retrieve actual chunks so the answer is real.
    empty_ctx: list[str] = []

    f_deltas_fluency: list[float] = []
    r_deltas_fluency: list[float] = []
    f_deltas_confidence: list[float] = []
    r_deltas_confidence: list[float] = []

    for g in answerable:
        retrieved = retriever.search(g.question, top_k=5)
        ans = answer_from_chunks(g.question, retrieved)
        if ans.refused:
            continue
        ctx = [rc.chunk.text for rc in retrieved]

        # Fluency perturbation.
        fluent_ans = perturb_fluency(ans.text, mode="high")
        choppy_ans = perturb_fluency(ans.text, mode="low")
        jf = judge.judge(g.question, fluent_ans, ctx)
        jc = judge.judge(g.question, choppy_ans, ctx)
        f_deltas_fluency.append(jf.faithfulness - jc.faithfulness)
        r_deltas_fluency.append(jf.answer_relevancy - jc.answer_relevancy)

        # Confidence perturbation.
        conf_ans = perturb_confidence(ans.text, mode="high")
        hedg_ans = perturb_confidence(ans.text, mode="low")
        jcf = judge.judge(g.question, conf_ans, ctx)
        jch = judge.judge(g.question, hedg_ans, ctx)
        f_deltas_confidence.append(jcf.faithfulness - jch.faithfulness)
        r_deltas_confidence.append(jcf.answer_relevancy - jch.answer_relevancy)

    # Retrieval-judge correlation from the EvalReport.
    answerable_items = [
        it for it in report.items
        if it.faithfulness is not None and it.answer_relevancy is not None
    ]
    precisions = [it.context_precision for it in answerable_items
                  if it.context_precision is not None]
    recalls = [it.context_recall for it in answerable_items
               if it.context_recall is not None]
    faith_scores = [it.faithfulness for it in answerable_items
                    if it.faithfulness is not None]
    relev_scores = [it.answer_relevancy for it in answerable_items
                    if it.answer_relevancy is not None]

    f_corr = _safe_spearman(faith_scores[:len(precisions)],
                            precisions[:len(faith_scores)])
    r_corr = _safe_spearman(relev_scores[:len(recalls)],
                            recalls[:len(relev_scores)])

    return BiasReport(
        fluency_delta_faithfulness=_safe_mean(f_deltas_fluency),
        fluency_delta_relevancy=_safe_mean(r_deltas_fluency),
        confidence_delta_faithfulness=_safe_mean(f_deltas_confidence),
        confidence_delta_relevancy=_safe_mean(r_deltas_confidence),
        retrieval_faithfulness_corr=f_corr,
        retrieval_relevancy_corr=r_corr,
        n_pairs=n_pairs,
    )


def render_bias_report(br: BiasReport) -> str:
    lines = [
        "Meta-evaluation: LLM-as-Judge Bias Report",
        f"  perturbation pairs: {br.n_pairs}",
        "",
        "  Fluency bias (Δ high - low):",
        f"    faithfulness: {br.fluency_delta_faithfulness:+.4f}",
        f"    relevancy:    {br.fluency_delta_relevancy:+.4f}",
        "",
        "  Confidence bias (Δ high - low):",
        f"    faithfulness: {br.confidence_delta_faithfulness:+.4f}",
        f"    relevancy:    {br.confidence_delta_relevancy:+.4f}",
        "",
        "  Retrieval-judge correlation (Spearman ρ):",
        f"    faithfulness ~ precision: {br.retrieval_faithfulness_corr:+.4f}",
        f"    relevancy    ~ recall:    {br.retrieval_relevancy_corr:+.4f}",
        "",
        "  (deltas near 0 = unbiased; large |ρ| = judge tracks retrieval)",
    ]
    return "\n".join(lines)


# -- helpers -------------------------------------------------------------------

def _safe_mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def _safe_spearman(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    return spearman_rank(xs[:n], ys[:n])
