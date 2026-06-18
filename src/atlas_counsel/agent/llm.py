"""LLM provider abstraction for the agent's generative nodes.

`synthesize` and `verify` are the only nodes that need an LLM. We isolate that
behind `LLMProvider` so the graph topology, routing, checkpointing and
human-gate are all testable offline:

  * `TemplateLLM` — deterministic, offline, dependency-free. It produces a
    grounded answer by extracting sentences from the retrieved spans and
    attaching their real span_ids as citations. It NEVER invents a citation,
    so it models the behavior we want the real LLM to have and lets the verify
    loop be exercised. It is a stand-in for plumbing, not a quality model.

  * Real providers (Ollama for dev, Bedrock for prod) implement the same
    protocol with an actual prompt + structured-output parse.

Both return the same Pydantic types, so nodes are provider-agnostic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .schemas import DraftAnswer, GroundingVerdict
from ..retrieval import RetrievedChunk


@runtime_checkable
class LLMProvider(Protocol):
    def synthesize(
        self, question: str, context: list[RetrievedChunk], guidance: str | None
    ) -> DraftAnswer: ...

    def verify(
        self, question: str, draft: DraftAnswer, context: list[RetrievedChunk]
    ) -> GroundingVerdict: ...


class TemplateLLM:
    """Deterministic offline provider.

    synthesize: builds a claim per top retrieved span, each citing that span's
                real id. Optionally honors human guidance by restricting to a
                preferred span if present in context.
    verify:     re-checks that every claim's cited span is in the retrieved
                set AND that the claim text actually came from that span. Since
                synthesize only ever uses retrieved spans, this passes on the
                first try in the happy path — but the *mechanism* is real, so
                a real LLM that hallucinates a citation would be caught.
    """

    def __init__(self, max_claims: int = 2) -> None:
        self._max_claims = max_claims

    def synthesize(
        self, question: str, context: list[RetrievedChunk], guidance: str | None
    ) -> DraftAnswer:
        from .schemas import Claim

        pool = context[: self._max_claims]
        if guidance:
            # If a human steered toward a specific doc, prefer its spans.
            preferred = [c for c in context if guidance.lower() in c.chunk.doc_id.lower()]
            if preferred:
                pool = preferred[: self._max_claims]

        claims = [
            Claim(text=c.chunk.text.strip(), span_id=c.chunk.span_id)
            for c in pool
        ]
        return DraftAnswer(claims=claims)

    def verify(
        self, question: str, draft: DraftAnswer, context: list[RetrievedChunk]
    ) -> GroundingVerdict:
        retrieved_text = {c.chunk.span_id: c.chunk.text.strip() for c in context}
        unsupported: list[str] = []
        for claim in draft.claims:
            # A claim is faithful iff its cited span was retrieved AND its text
            # matches that span's text (no fabrication / drift).
            src = retrieved_text.get(claim.span_id)
            if src is None or claim.text.strip() not in src and src not in claim.text.strip():
                unsupported.append(claim.span_id or "<no-citation>")
        return GroundingVerdict(
            faithful=not unsupported,
            unsupported_span_ids=unsupported,
        )
