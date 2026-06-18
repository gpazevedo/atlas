"""Shared tokenization used by chunking, reranker, judge, and answerer.

All consumers that need tokenized forms of chunk text reuse the tokens
precomputed on `Chunk` rather than running regex again — this eliminates
repeated regex from the hot path.
"""

from __future__ import annotations

import re

WORD_RE = re.compile(r"[a-z0-9$%.,]+")

STOPWORDS: set[str] = {
    "the", "a", "an", "of", "to", "for", "and", "or", "is", "are", "what",
    "which", "who", "does", "do", "above", "value", "need", "our", "with",
    "at", "in", "on", "be", "by", "this", "that", "i", "can", "from", "into",
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens from *text* (regex, no stopword filtering)."""
    return [t.lower() for t in WORD_RE.findall(text)]


def content_tokens(text: str, *, min_len: int = 3, extra_stop: set[str] | None = None) -> set[str]:
    """Distinctive content tokens: stopword-filtered, length-filtered."""
    stop = STOPWORDS | (extra_stop or set())
    return {t for t in tokenize(text) if t not in stop and len(t) >= min_len}
