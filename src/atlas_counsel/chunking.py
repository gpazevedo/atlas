"""Chunking.

The corpus is already authored as spans, and each span is the natural
citation unit. So a chunk maps 1:1 to a span and *carries its span_id
unchanged* — this is what lets a retrieval result cite `POL-001#S1`
end to end. If later we sub-split long spans, the chunk id becomes
`<span_id>::c<n>` but still resolves back to the owning span.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ._tokenize import tokenize
from .corpus.models import Corpus, DocCategory


class Chunk(BaseModel):
    chunk_id: str = Field(..., description="Stable id; equals span_id for 1:1 chunks")
    span_id: str = Field(..., description="Owning span — the citation target")
    doc_id: str
    category: DocCategory
    title: str
    vendor: str | None = None
    heading: str | None = None
    text: str = Field(..., min_length=1)
    # Precomputed tokens so downstream (reranker, judge, answerer) don't
    # re-run regex on the same text repeatedly.
    tokens: list[str] = Field(default_factory=list)

    @property
    def embed_text(self) -> str:
        """Text handed to the embedder. Prepending title+heading gives the
        dense model document context a bare span would lack."""
        prefix_parts = [self.title]
        if self.heading:
            prefix_parts.append(self.heading)
        return " — ".join(prefix_parts) + "\n" + self.text


def chunk_corpus(corpus: Corpus) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in corpus.documents:
        for span in doc.spans:
            chunks.append(
                Chunk(
                    chunk_id=span.span_id,
                    span_id=span.span_id,
                    doc_id=doc.doc_id,
                    category=doc.category,
                    title=doc.title,
                    vendor=doc.vendor,
                    heading=span.heading,
                    text=span.text,
                    tokens=tokenize(span.text),
                )
            )
    return chunks
