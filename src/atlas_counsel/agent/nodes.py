"""Agent nodes + routing.

Each node is `(state) -> partial state`. Routers are `(state) -> next node
name`. Keeping nodes pure (no hidden I/O beyond the injected retriever/llm)
makes the whole graph deterministic under the offline providers and therefore
unit-testable.

Control flow:

    plan -> retrieve -> validate
    validate --grounded--> synthesize
             --insufficient--> human_gate
    synthesize -> verify
    verify --pass--> finalize -> END
           --unfaithful & attempts left--> synthesize   (retry)
           --unfaithful & exhausted--> human_gate
    human_gate --steer--> synthesize
               --decline--> finalize(refuse) -> END
"""

from __future__ import annotations

from langgraph.types import interrupt

from .llm import LLMProvider
from .schemas import CounselAnswer
from .state import CounselState
from ..decompose import QueryDecomposer
from ..eval.answerer import grounding_overlap
from ..retrieval import Retriever

MAX_ATTEMPTS = 2
GROUNDING_THRESHOLD = 0.25


def make_nodes(
    retriever: Retriever,
    llm: LLMProvider,
    decomposer: QueryDecomposer | None,
    top_k: int = 5,
):
    """Build the node callables closed over the injected dependencies."""

    def plan(state: CounselState) -> CounselState:
        q = state["question"]
        subs = decomposer.decompose(q) if decomposer else [q]
        return {"sub_queries": subs, "attempts": 0, "escalated": False,
                "human_input": None}

    def retrieve(state: CounselState) -> CounselState:
        merged: dict[str, object] = {}
        for sub in state["sub_queries"]:
            for rc in retriever.search(sub, top_k=top_k):
                cur = merged.get(rc.chunk.span_id)
                if cur is None or rc.score > cur.score:  # type: ignore[union-attr]
                    merged[rc.chunk.span_id] = rc
        ranked = sorted(merged.values(), key=lambda rc: -rc.score)  # type: ignore[attr-defined]
        return {"retrieved": ranked}

    def validate(state: CounselState) -> CounselState:
        # Grounding decision on actual retrieved content — this replaces the
        # PR3 score-threshold stub. (RRF scores carry no relevance signal; the
        # query's distinctive tokens appearing in retrieved spans do.)
        overlap = grounding_overlap(state["question"], state["retrieved"], top_n=top_k)
        return {"grounded": overlap >= GROUNDING_THRESHOLD}

    def synthesize(state: CounselState) -> CounselState:
        draft = llm.synthesize(
            state["question"], state["retrieved"], state.get("human_input")
        )
        return {"draft": draft, "attempts": state.get("attempts", 0) + 1}

    def verify(state: CounselState) -> CounselState:
        verdict = llm.verify(state["question"], state["draft"], state["retrieved"])
        return {"verdict": verdict}

    def human_gate(state: CounselState) -> CounselState:
        # Pause the run and surface why. The caller resumes with a decision:
        #   {"action": "steer", "guidance": "<doc hint>"} or {"action": "decline"}
        decision = interrupt({
            "reason": "insufficient_grounding" if not state.get("grounded", True)
                      else "unfaithful_after_retries",
            "question": state["question"],
            "retrieved_span_ids": [rc.chunk.span_id for rc in state["retrieved"]],
        })
        action = (decision or {}).get("action", "decline")
        if action == "steer":
            return {"escalated": True, "human_input": (decision or {}).get("guidance", "")}
        return {"escalated": True, "human_input": "__decline__"}

    def finalize(state: CounselState) -> CounselState:
        attempts = state.get("attempts", 1)
        escalated = state.get("escalated", False)
        human_input = state.get("human_input")
        declined = human_input == "__decline__"
        # A human steer (non-decline guidance after escalation) overrides the
        # original automated grounding verdict — the human chose to proceed.
        steered = escalated and human_input not in (None, "__decline__")
        ungrounded = not state.get("grounded", True) and not steered
        if declined or ungrounded:
            return {"answer": CounselAnswer.refusal(attempts=attempts, escalated=escalated)}
        return {"answer": CounselAnswer.from_claims(
            state["draft"], attempts=attempts, escalated=escalated)}

    return {
        "plan": plan, "retrieve": retrieve, "validate": validate,
        "synthesize": synthesize, "verify": verify,
        "human_gate": human_gate, "finalize": finalize,
    }


# --- routers ----------------------------------------------------------------

def route_after_validate(state: CounselState) -> str:
    return "synthesize" if state.get("grounded") else "human_gate"


def route_after_verify(state: CounselState) -> str:
    if state["verdict"].faithful:
        return "finalize"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "human_gate"      # exhausted retries -> escalate
    return "synthesize"          # bounded retry


def route_after_human(state: CounselState) -> str:
    # Declined -> finalize as refusal; steered -> try synthesizing again.
    if state.get("human_input") == "__decline__":
        return "finalize"
    return "synthesize"
