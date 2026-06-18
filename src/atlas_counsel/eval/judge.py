"""Answer-quality judging.

Faithfulness and answer-relevancy are the metrics that genuinely need an LLM
(or at least semantic judgement). We isolate that behind `LLMJudge` so:

  * CI uses `HeuristicJudge` — deterministic, offline, dependency-free. It is
    a *lower-fidelity* proxy (lexical grounding + refusal detection), honest
    about being a proxy, that lets the whole harness run and regress in CI.
  * Locally / in prod you inject an LLM-backed judge implementing the same
    protocol (a thin wrapper over your dev Ollama or Bedrock Claude).

Both return scores in [0, 1] so the report format is identical regardless.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from .._tokenize import STOPWORDS, tokenize


class JudgeResult(BaseModel):
    faithfulness: float    # are the answer's claims supported by the context?
    answer_relevancy: float  # does the answer address the question?


class LLMJudge(Protocol):
    def judge(
        self, question: str, answer: str, context: list[str]
    ) -> JudgeResult: ...


_REFUSAL_MARKERS = (
    "cannot", "can't", "not covered", "no information", "don't have",
    "not in", "unable to", "does not", "insufficient", "not found",
)


def looks_like_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(m in a for m in _REFUSAL_MARKERS)


class HeuristicJudge:
    """Deterministic offline judge — a proxy, not a real grader.

    faithfulness:     fraction of the answer's content tokens that are present
                      in the retrieved context. High when the answer doesn't
                      invent tokens absent from context; low when it does.
    answer_relevancy: token overlap between answer and question (excluding
                      stopwords). Crude but monotonic with on-topic-ness.

    Refusals are handled by the runner, not here.
    """

    # Matches the old stopword set exactly — omits "i", "can", "from", "into"
    # from the shared STOPWORDS to preserve existing score behaviour.
    _STOP = STOPWORDS - {"i", "can", "from", "into"}

    def judge(
        self, question: str, answer: str, context: list[str]
    ) -> JudgeResult:
        ans_tokens = set(tokenize(answer)) - self._STOP
        ctx_tokens = set().union(*[set(tokenize(c)) for c in context]) if context else set()
        q_tokens = set(tokenize(question)) - self._STOP

        faithfulness = (
            len(ans_tokens & ctx_tokens) / len(ans_tokens) if ans_tokens else 0.0
        )
        relevancy = (
            len(ans_tokens & q_tokens) / len(q_tokens) if q_tokens else 0.0
        )
        return JudgeResult(
            faithfulness=round(faithfulness, 4),
            answer_relevancy=round(min(relevancy, 1.0), 4),
        )
