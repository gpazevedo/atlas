"""Schema-driven core for the ATLAS Counsel corpus.

Every artifact the generator emits is a validated Pydantic model. The same
models are re-used by the ingest pipeline and the eval harness, so the
citation contract (stable span IDs) holds end to end.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, computed_field, model_validator


class DocCategory(str, Enum):
    POLICY = "policy"
    CONTRACT = "contract"
    NEGOTIATION_LOG = "negotiation_log"


class Span(BaseModel):
    """A citable unit of text. The (doc_id, ordinal) pair yields a stable,
    human-readable citation id like 'POL-003#S2' that survives re-generation
    as long as the seed is unchanged."""

    ordinal: int = Field(..., ge=0, description="0-based position within its doc")
    heading: str | None = Field(None, description="Section heading, if any")
    text: str = Field(..., min_length=1)

    # Populated by the parent Document so spans are self-describing once
    # detached (e.g. inside a retrieval result).
    doc_id: str = Field(default="", description="Owning document id")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def span_id(self) -> str:
        return f"{self.doc_id}#S{self.ordinal}"


class Document(BaseModel):
    doc_id: str = Field(..., pattern=r"^[A-Z]{3}-\d{3}$")
    category: DocCategory
    title: str
    vendor: str | None = Field(None, description="Set for contracts / logs")
    spans: list[Span] = Field(..., min_length=1)
    # Free-form tags used by eval to slice results (e.g. 'contradiction-pair').
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _stamp_spans(self) -> "Document":
        for i, span in enumerate(self.spans):
            if span.ordinal != i:
                raise ValueError(
                    f"{self.doc_id}: span ordinal {span.ordinal} != position {i}"
                )
            span.doc_id = self.doc_id
        return self

    @property
    def full_text(self) -> str:
        parts = []
        for s in self.spans:
            parts.append(f"## {s.heading}\n{s.text}" if s.heading else s.text)
        return "\n\n".join(parts)


class AnswerType(str, Enum):
    """What kind of grounding the golden answer expects."""

    GROUNDED = "grounded"            # answer lives in one or more spans
    UNANSWERABLE = "unanswerable"    # corpus does NOT contain the answer
    MULTI_HOP = "multi_hop"          # requires combining spans across docs


class GoldenItem(BaseModel):
    """One evaluation question with its known supporting spans."""

    qid: str = Field(..., pattern=r"^Q-\d{3}$")
    question: str = Field(..., min_length=1)
    answer_type: AnswerType
    # Span ids that MUST be retrieved/cited for a correct grounded answer.
    # Empty iff answer_type == UNANSWERABLE.
    supporting_span_ids: list[str] = Field(default_factory=list)
    # A reference answer for faithfulness/answer-relevancy scoring. For
    # unanswerable items this is the expected refusal.
    reference_answer: str = Field(..., min_length=1)
    # Eval slice tags: 'contradiction', 'threshold-precision', 'cross-vendor'…
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_support(self) -> "GoldenItem":
        if self.answer_type == AnswerType.UNANSWERABLE:
            if self.supporting_span_ids:
                raise ValueError(f"{self.qid}: unanswerable item must have no spans")
        elif not self.supporting_span_ids:
            raise ValueError(f"{self.qid}: grounded item needs supporting spans")
        return self


class Corpus(BaseModel):
    documents: list[Document]
    golden: list[GoldenItem]

    @model_validator(mode="after")
    def _referential_integrity(self) -> "Corpus":
        known = {s.span_id for d in self.documents for s in d.spans}
        for item in self.golden:
            missing = set(item.supporting_span_ids) - known
            if missing:
                raise ValueError(f"{item.qid} references unknown spans: {missing}")
        return self
