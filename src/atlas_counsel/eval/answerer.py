"""A minimal, deterministic answerer for end-to-end eval.

This is NOT the production synthesizer (that arrives with the LangGraph PR and
uses an LLM with Pydantic-enforced citations). It exists so the eval harness
can score a full retrieve->answer loop offline and deterministically.

Refusal design (important): RRF fusion scores are rank-based and carry NO
absolute relevance signal — the top hit always scores ~1/(k+1) whether or not
it is actually on-topic. So a raw-score threshold cannot tell answerable from
unanswerable. Instead we refuse based on *lexical grounding*: how much of the
query's distinctive content actually appears in the retrieved spans. An
absent-from-corpus question ("supplier gifts") has near-zero overlap; a
covered one ("$50,000 single-source") overlaps strongly. The production
synthesizer replaces this with an LLM faithfulness self-check.
"""

from __future__ import annotations

from pydantic import BaseModel

from .._tokenize import STOPWORDS, content_tokens
from ..retrieval import RetrievedChunk


class Answer(BaseModel):
    text: str
    citations: list[str]   # span ids
    refused: bool


REFUSAL_TEXT = (
    "I cannot answer this from the available procurement documents; the "
    "information is not covered by the corpus."
)

_DOMAIN_STOP = {"policy", "purchase", "agreement"}


def grounding_overlap(question: str, retrieved: list[RetrievedChunk], top_n: int) -> float:
    """Fraction of the query's distinctive content tokens that appear in the
    top-n retrieved spans. 0 => nothing in the corpus addresses the query."""
    q = content_tokens(question, min_len=3, extra_stop=_DOMAIN_STOP)
    if not q:
        return 0.0
    ctx: set[str] = set()
    for rc in retrieved[:top_n]:
        ctx |= {t for t in rc.chunk.tokens if t not in STOPWORDS | _DOMAIN_STOP and len(t) >= 3}
        if rc.chunk.heading:
            ctx |= content_tokens(rc.chunk.heading, min_len=3, extra_stop=_DOMAIN_STOP)
        ctx |= content_tokens(rc.chunk.title, min_len=3, extra_stop=_DOMAIN_STOP)
    return len(q & ctx) / len(q)


def answer_from_chunks(
    question: str,
    retrieved: list[RetrievedChunk],
    grounding_threshold: float = 0.25,
    top_n: int = 2,
) -> Answer:
    if not retrieved:
        return Answer(text=REFUSAL_TEXT, citations=[], refused=True)

    overlap = grounding_overlap(question, retrieved, top_n)
    if overlap < grounding_threshold:
        return Answer(text=REFUSAL_TEXT, citations=[], refused=True)

    top = retrieved[:top_n]
    parts = [rc.chunk.text for rc in top]
    cites = [rc.chunk.span_id for rc in top]
    text = " ".join(parts) + " [" + ", ".join(cites) + "]"
    return Answer(text=text, citations=cites, refused=False)
