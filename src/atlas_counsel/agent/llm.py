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
from ..memory.store import ProceduralSkill, ReflectionResult
from ..retrieval import RetrievedChunk


@runtime_checkable
class LLMProvider(Protocol):
    def synthesize(
        self, question: str, context: list[RetrievedChunk], guidance: str | None
    ) -> DraftAnswer: ...

    def verify(
        self, question: str, draft: DraftAnswer, context: list[RetrievedChunk]
    ) -> GroundingVerdict: ...

    def gap_analyze(
        self, question: str, context: list[RetrievedChunk]
    ) -> list[str]: ...

    def reflect(
        self, question: str, answer_text: str, thread_id: str
    ) -> ReflectionResult: ...


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
            src = retrieved_text.get(claim.span_id)
            if src is None or claim.text.strip() not in src and src not in claim.text.strip():
                unsupported.append(claim.span_id or "<no-citation>")
        return GroundingVerdict(
            faithful=not unsupported,
            unsupported_span_ids=unsupported,
        )

    def gap_analyze(
        self, question: str, context: list[RetrievedChunk]
    ) -> list[str]:
        """Return follow-up queries targeting question tokens not yet covered."""
        from .._tokenize import STOPWORDS, content_tokens
        _DOMAIN_STOP = {"policy", "purchase", "agreement"}
        q_tokens = content_tokens(question, min_len=3, extra_stop=_DOMAIN_STOP)
        if not q_tokens:
            return []
        ctx_tokens: set[str] = set()
        for rc in context:
            ctx_tokens |= {t for t in rc.chunk.tokens
                           if t not in STOPWORDS | _DOMAIN_STOP and len(t) >= 3}
            if rc.chunk.heading:
                ctx_tokens |= content_tokens(rc.chunk.heading, min_len=3,
                                             extra_stop=_DOMAIN_STOP)
            ctx_tokens |= content_tokens(rc.chunk.title, min_len=3,
                                         extra_stop=_DOMAIN_STOP)
        missing = [t for t in q_tokens if t not in ctx_tokens]
        if not missing:
            return []
        return [" ".join(missing)]

    def reflect(
        self, question: str, answer_text: str, thread_id: str
    ) -> ReflectionResult:
        """Deterministic heuristic reflection.

        Extracts answer sentences as semantic facts, builds a simple episodic
        summary from question + answer, and returns no skills (conservative).
        Real LLM providers override this with structured-output prompting.
        """
        # Semantic facts: split answer on sentence boundaries, keep non-trivial
        facts: list[str] = []
        for part in answer_text.replace("! ", ". ").replace("? ", ". ").split(". "):
            clean = part.strip().rstrip(".")
            if clean and len(clean.split()) >= 4:
                facts.append(clean + ".")
        # Episodic summary: short concatenation
        summary = (
            f"Q: {question[:200]} | "
            f"A: {answer_text[:200]}{'...' if len(answer_text) > 200 else ''}"
        )
        return ReflectionResult(
            semantic_facts=facts,
            episodic_summary=summary,
            skills=[],
        )
