"""Structured-output schemas for the agent.

Citations are load-bearing: a `Claim` cannot exist without a `span_id` field,
and the final `CounselAnswer` is assembled only from verified claims. This is
the JD's "structured output patterns" + "citation-based grounding" enforced by
the type system rather than by prompt-politeness.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """One assertion in an answer, bound to the span that supports it."""

    text: str = Field(..., min_length=1)
    span_id: str = Field(..., description="Citation target, e.g. POL-001#S1")


class DraftAnswer(BaseModel):
    claims: list[Claim] = Field(default_factory=list)

    @property
    def cited_span_ids(self) -> list[str]:
        return [c.span_id for c in self.claims]


class GroundingVerdict(BaseModel):
    """Output of the verify node."""

    faithful: bool
    unsupported_span_ids: list[str] = Field(default_factory=list)


_REFUSAL_TEXT = (
    "I cannot answer this from the available documents; "
    "the information is not covered by the corpus."
)
_REFUSAL_NO_HITL_TEXT = "I don't know how to answer this question."


class CounselAnswer(BaseModel):
    """The agent's final, user-facing result."""

    text: str
    citations: list[str] = Field(default_factory=list)
    refused: bool = False
    # Provenance for observability / eval.
    attempts: int = 1
    escalated: bool = False

    @classmethod
    def from_claims(cls, draft: DraftAnswer, *, attempts: int, escalated: bool) -> "CounselAnswer":
        text = " ".join(c.text for c in draft.claims)
        cites = draft.cited_span_ids
        if cites:
            text = text + " [" + ", ".join(cites) + "]"
        return cls(text=text, citations=cites, refused=False,
                   attempts=attempts, escalated=escalated)

    @classmethod
    def refusal(cls, *, attempts: int, escalated: bool,
                no_hitl: bool = False) -> "CounselAnswer":
        text = (
            _REFUSAL_NO_HITL_TEXT if no_hitl else _REFUSAL_TEXT
        )
        return cls(text=text, citations=[], refused=True,
                   attempts=attempts, escalated=escalated)
