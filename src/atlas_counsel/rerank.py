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

from typing import Protocol

from ._tokenize import STOPWORDS, content_tokens, tokenize
from .retrieval import RetrievedChunk


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

    @staticmethod
    def _doc_content(c: RetrievedChunk) -> set[str]:
        """Content tokens from chunk text (precomputed) + title (cheap, short)."""
        text_tokens = {t for t in c.chunk.tokens if t not in STOPWORDS and len(t) >= 3}
        title_tokens = content_tokens(c.chunk.title, min_len=3)
        return text_tokens | title_tokens

    @staticmethod
    def _doc_bigrams(c: RetrievedChunk) -> set[tuple[str, str]]:
        return set(zip(c.chunk.tokens, c.chunk.tokens[1:]))

    def _doc_freq(self, docs: list[set[str]]) -> dict[str, int]:
        df: dict[str, int] = {}
        for terms in docs:
            for tok in terms:
                df[tok] = df.get(tok, 0) + 1
        return df

    def _pair_score(
        self, q_terms: set[str], q_bigrams: set[tuple[str, str]],
        doc_terms: set[str], doc_bigrams: set[tuple[str, str]],
        df: dict[str, int], n: int,
    ) -> float:
        if not q_terms:
            return 0.0
        matched = q_terms & doc_terms
        coverage = len(matched) / len(q_terms)
        rarity = sum(1.0 / (1 + df.get(t, 0)) for t in matched)
        rarity_norm = rarity / len(q_terms)

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
        q_terms = content_tokens(query, min_len=3)
        q_tokens = tokenize(query)
        q_bigrams = set(zip(q_tokens, q_tokens[1:]))

        # Precompute doc-side data once per candidate — avoids re-running
        # regex/tokenization for every (query, chunk) pair.
        doc_terms_list = [self._doc_content(c) for c in candidates]
        doc_bigrams_list = [self._doc_bigrams(c) for c in candidates]
        df = self._doc_freq(doc_terms_list)

        rescored: list[RetrievedChunk] = []
        for i, c in enumerate(candidates):
            s = self._pair_score(
                q_terms, q_bigrams,
                doc_terms_list[i], doc_bigrams_list[i],
                df, len(candidates),
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
