"""Reranking.

A reranker rescores the top-N candidates from hybrid retrieval using a richer
query-document interaction signal than the first-stage retriever can afford at
corpus scale. This is the standard two-stage RAG pattern: cheap recall-oriented
retrieval, then expensive precision-oriented rerank over a short list.

`Reranker` is the protocol. Two implementations:

  * `TokenInteractionReranker` — deterministic, offline, for CI. It scores a
    (query, chunk) pair using ONLY signals a real cross-encoder would also see:
    weighted token overlap with an exact-phrase bonus and a coverage term. It
    NEVER consults the golden spans — if it did, the eval would be a lie. It is
    a lower-fidelity proxy whose job is to validate the *plumbing* and the
    measured effect direction, not to match a real model's quality.

  * `CrossEncoderReranker` — wraps a sentence-transformers CrossEncoder
    (e.g. BAAI/bge-reranker-v2-m3). Injected locally; same protocol.

The pipeline measures rerank's effect via the eval harness, so we don't have
to take its value on faith.
"""

from __future__ import annotations

import re
from typing import Protocol

from .retrieval import RetrievedChunk

_WORD = re.compile(r"[a-z0-9$%.,]+")
_STOP = {
    "the", "a", "an", "of", "to", "for", "and", "or", "is", "are", "what",
    "which", "who", "does", "do", "above", "value", "need", "our", "with",
    "at", "in", "on", "be", "by", "this", "that", "i", "can", "from", "into",
}


def _tokens(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def _content(s: str) -> set[str]:
    return {t for t in _tokens(s) if t not in _STOP and len(t) > 2}


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


class TokenInteractionReranker:
    """Offline cross-encoder proxy. Honest: uses only (query, doc) text.

    Score = coverage(query terms found in doc)
          + 0.5 * idf-ish rarity weight of matched terms
          + phrase_bonus if a 2-gram from the query appears verbatim in the doc

    The phrase bonus is what a bag-of-words first stage misses and a real
    cross-encoder captures: term *adjacency*. That's the mechanism by which
    rerank is expected to help multi-term queries.

    KNOWN LIMITATION (measured): this lexical proxy cannot do the semantic /
    numeric-range reasoning a real cross-encoder does. E.g. for "who approves a
    $60,000 purchase" it cannot infer that $60,000 falls in the "$10k-$100k"
    tier span, and is misled by lexical "approve" overlap in an unrelated
    conflict-of-interest span. On the current small corpus this makes the proxy
    a net-neutral-to-negative reranker — see the ablation in `scripts`/README.
    It is shipped as correct, tested *plumbing*; the production
    `CrossEncoderReranker` is the implementation expected to win, and the eval
    harness is how you confirm that locally. We do NOT tune the proxy toward
    the golden spans to manufacture a win.
    """

    def __init__(self, phrase_bonus: float = 1.0) -> None:
        self._phrase_bonus = phrase_bonus

    def _doc_freq(self, candidates: list[RetrievedChunk]) -> dict[str, int]:
        df: dict[str, int] = {}
        for c in candidates:
            for tok in _content(c.chunk.text) | _content(c.chunk.title):
                df[tok] = df.get(tok, 0) + 1
        return df

    def _pair_score(
        self, q_terms: set[str], q_bigrams: set[str],
        chunk_text: str, chunk_title: str, df: dict[str, int], n: int,
    ) -> float:
        doc_terms = _content(chunk_text) | _content(chunk_title)
        if not q_terms:
            return 0.0
        matched = q_terms & doc_terms
        coverage = len(matched) / len(q_terms)
        # rarer matched terms (low doc freq) weigh more
        rarity = sum(1.0 / (1 + df.get(t, 0)) for t in matched)
        rarity_norm = rarity / len(q_terms)

        doc_bigrams = set(zip(_tokens(chunk_text), _tokens(chunk_text)[1:]))
        # Phrase signal scales with the FRACTION of query bigrams matched, so a
        # passage that reproduces the query's phrasing is rewarded in
        # proportion to how much structure it shares — strong adjacency can
        # then overcome a passage that merely stuffs the individual keywords.
        if q_bigrams:
            phrase = self._phrase_bonus * (len(q_bigrams & doc_bigrams) / len(q_bigrams))
        else:
            phrase = 0.0
        return coverage + 0.5 * rarity_norm + phrase

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        q_terms = _content(query)
        q_tokens = _tokens(query)
        q_bigrams = set(zip(q_tokens, q_tokens[1:]))
        df = self._doc_freq(candidates)

        rescored: list[RetrievedChunk] = []
        for c in candidates:
            s = self._pair_score(
                q_terms, q_bigrams, c.chunk.text, c.chunk.title, df, len(candidates)
            )
            rescored.append(c.model_copy(update={"score": s}))

        rescored.sort(key=lambda rc: -rc.score)
        return rescored[:top_k]


class CrossEncoderReranker:
    """Production reranker over a sentence-transformers CrossEncoder
    (default BAAI/bge-reranker-v2-m3). Injected locally; needs the optional
    `rerank` extra. Same protocol as the offline proxy.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", model=None) -> None:
        self._model_name = model_name
        self._model = model  # injectable for testing

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # optional dep

            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        model = self._ensure_model()
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = model.predict(pairs)
        rescored = [
            c.model_copy(update={"score": float(s)})
            for c, s in zip(candidates, scores)
        ]
        rescored.sort(key=lambda rc: -rc.score)
        return rescored[:top_k]
