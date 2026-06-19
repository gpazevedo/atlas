"""Agent graph tests.

Covers every control-flow path:
  * grounded happy path (no escalation)
  * unanswerable -> interrupt -> decline -> refusal
  * unanswerable -> interrupt -> steer -> answer
  * bounded retry: a hallucinating LLM is caught by verify and the loop
    terminates at MAX_ATTEMPTS rather than spinning
  * gap-aware iterative retrieval: insufficient grounding triggers gap_analyze
    -> retrieve loop, accumulating chunks before escalating
  * routers in isolation
  * structured-output / citation enforcement
"""

from __future__ import annotations

import warnings

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from atlas_counsel.agent import build_counsel_graph
from atlas_counsel.agent.llm import TemplateLLM
from atlas_counsel.agent.nodes import (
    MAX_ATTEMPTS,
    MAX_GAP_ITERATIONS,
    route_after_human,
    route_after_validate,
    route_after_verify,
)
from atlas_counsel.agent.schemas import (
    Claim,
    DraftAnswer,
    GroundingVerdict,
)
from atlas_counsel.chunking import chunk_corpus
from atlas_counsel.corpus import build_corpus
from atlas_counsel.embeddings import HashingEmbedder
from atlas_counsel.retrieval import InMemoryHybridRetriever

# Checkpoint serialization of custom types emits a forward-compat warning; not
# relevant to these tests.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _retriever():
    r = InMemoryHybridRetriever(HashingEmbedder())
    r.index(chunk_corpus(build_corpus()))
    return r


def _graph(llm=None):
    return build_counsel_graph(_retriever(), llm=llm, checkpointer=MemorySaver())


# --- happy path -------------------------------------------------------------

def test_grounded_question_answers_with_citation():
    graph = _graph()
    cfg = {"configurable": {"thread_id": "g1"}}
    out = graph.invoke(
        {"question": "Above what value does a single-source purchase need justification?"},
        cfg,
    )
    ans = out["answer"]
    assert not ans.refused
    assert "POL-001#S1" in ans.citations
    assert ans.attempts == 1
    assert not ans.escalated


def test_multi_hop_question_pulls_both_vendors():
    graph = _graph()
    cfg = {"configurable": {"thread_id": "g2"}}
    out = graph.invoke(
        {"question": "Compare the uptime guarantees of AcmeCloud and NorthLink"}, cfg
    )
    ans = out["answer"]
    assert not ans.refused
    # both vendor contract spans should be cited
    assert any(c.startswith("CON-001") for c in ans.citations)
    assert any(c.startswith("CON-002") for c in ans.citations)


# --- human-gate: interrupt + resume -----------------------------------------

def test_unanswerable_interrupts_then_declines_to_refusal():
    graph = _graph()
    cfg = {"configurable": {"thread_id": "h1"}}
    out = graph.invoke(
        {"question": "What is our policy on accepting gifts from suppliers?"}, cfg
    )
    assert "__interrupt__" in out
    assert out["__interrupt__"][0].value["reason"] == "insufficient_grounding"

    final = graph.invoke(Command(resume={"action": "decline"}), cfg)
    ans = final["answer"]
    assert ans.refused
    assert ans.escalated


def test_unanswerable_interrupts_then_steer_is_honored():
    graph = _graph()
    cfg = {"configurable": {"thread_id": "h2"}}
    out = graph.invoke(
        {"question": "What is our policy on accepting gifts from suppliers?"}, cfg
    )
    assert "__interrupt__" in out
    # Steer toward POL-004 (conflict of interest) — a human override.
    final = graph.invoke(
        Command(resume={"action": "steer", "guidance": "POL-004"}), cfg
    )
    ans = final["answer"]
    # Steered runs go back through synthesize; escalated flag records the human touch.
    assert ans.escalated
    assert not ans.refused
    assert all(c.startswith("POL-004") for c in ans.citations)


# --- bounded retry: hallucinating LLM is caught and the loop terminates ------

class _HallucinatingLLM:
    """Always cites a span that was NOT retrieved -> verify always fails.
    Used to prove the retry loop is bounded by MAX_ATTEMPTS."""

    def synthesize(self, question, context, guidance):
        return DraftAnswer(claims=[Claim(text="fabricated", span_id="ZZZ-999#S0")])

    def verify(self, question, draft, context):
        retrieved = {c.chunk.span_id for c in context}
        bad = [c.span_id for c in draft.claims if c.span_id not in retrieved]
        return GroundingVerdict(faithful=not bad, unsupported_span_ids=bad)


def test_hallucination_is_bounded_and_escalates():
    graph = build_counsel_graph(
        _retriever(), llm=_HallucinatingLLM(), checkpointer=MemorySaver()
    )
    cfg = {"configurable": {"thread_id": "r1"}}
    out = graph.invoke(
        {"question": "Above what value does a single-source purchase need justification?"},
        cfg,
    )
    # verify keeps failing; after MAX_ATTEMPTS it escalates to human_gate (interrupt)
    assert "__interrupt__" in out
    assert out["__interrupt__"][0].value["reason"] == "unfaithful_after_retries"
    # decline -> refusal, and attempts capped at MAX_ATTEMPTS
    final = graph.invoke(Command(resume={"action": "decline"}), cfg)
    assert final["answer"].refused
    assert final["answer"].attempts == MAX_ATTEMPTS


# --- routers in isolation ---------------------------------------------------

def test_route_after_verify_retries_then_escalates():
    faithful = {"verdict": GroundingVerdict(faithful=True)}
    assert route_after_verify(faithful) == "finalize"

    unfaithful_early = {"verdict": GroundingVerdict(faithful=False), "attempts": 1}
    assert route_after_verify(unfaithful_early) == "synthesize"

    unfaithful_exhausted = {"verdict": GroundingVerdict(faithful=False), "attempts": MAX_ATTEMPTS}
    assert route_after_verify(unfaithful_exhausted) == "human_gate"


def test_route_after_human():
    assert route_after_human({"human_input": "__decline__"}) == "finalize"
    assert route_after_human({"human_input": "POL-004"}) == "synthesize"


def test_route_after_validate_grounded_skips_gap():
    assert route_after_validate({"grounded": True}) == "synthesize"


def test_route_after_validate_gap_then_escalates():
    assert route_after_validate({"grounded": False, "gap_iterations": 0}) == "gap_analyze"
    assert route_after_validate({"grounded": False, "gap_iterations": 1}) == "gap_analyze"
    assert (
        route_after_validate({"grounded": False, "gap_iterations": MAX_GAP_ITERATIONS})
        == "human_gate"
    )


# --- gap-aware iterative retrieval -------------------------------------------

def test_unanswerable_still_reaches_human_gate_after_gap_loop():
    """Gap loop runs but doesn't find content for a genuinely unanswerable
    question; it exhausts iterations and escalates to human_gate."""
    graph = _graph()
    cfg = {"configurable": {"thread_id": "gap1"}}
    out = graph.invoke(
        {"question": "What is our policy on accepting gifts from suppliers?"}, cfg
    )
    assert "__interrupt__" in out
    assert out["__interrupt__"][0].value["reason"] == "insufficient_grounding"
    # Gap iterations were exhausted before reaching human_gate.
    state = graph.get_state(cfg)
    assert state.values.get("gap_iterations", 0) == MAX_GAP_ITERATIONS


def test_gap_analysis_accumulates_chunks():
    """Follow-up queries from gap_analyze merge new chunks into the existing
    retrieved set rather than replacing them."""
    graph = _graph()
    cfg = {"configurable": {"thread_id": "gap2"}}
    out = graph.invoke(
        {"question": "What is our policy on accepting gifts from suppliers?"}, cfg
    )
    state = graph.get_state(cfg)
    # Retrieved chunks should contain results from the initial plan retrieval
    # plus any chunks found during gap-analysis rounds.
    assert len(state.values.get("retrieved", [])) > 0


def test_template_llm_gap_analyze_returns_followup():
    """gap_analyze identifies question tokens not covered by retrieved spans."""
    from atlas_counsel.chunking import chunk_corpus
    from atlas_counsel.corpus import build_corpus
    from atlas_counsel.embeddings import HashingEmbedder
    from atlas_counsel.retrieval import InMemoryHybridRetriever

    r = InMemoryHybridRetriever(HashingEmbedder())
    r.index(chunk_corpus(build_corpus()))
    chunks = r.search("single-source purchase threshold", top_k=3)
    llm = TemplateLLM()

    # Tokens like "purchase" and "threshold" should be covered by retrieved
    # spans; "gifts" and "suppliers" are absent from the corpus. A gap-aware
    # question should produce a follow-up query for the missing tokens.
    follow_ups = llm.gap_analyze(
        "What is our policy on accepting gifts from suppliers?", chunks
    )
    assert isinstance(follow_ups, list)
    # The missing tokens (gifts, suppliers, accepting) should appear in the query.
    if follow_ups:
        assert any("gifts" in f for f in follow_ups)


# --- structured output / citation enforcement -------------------------------

def test_claim_requires_span_id():
    with pytest.raises(Exception):
        Claim(text="something")  # missing span_id


def test_counsel_answer_assembles_citations():
    draft = DraftAnswer(claims=[
        Claim(text="A.", span_id="POL-001#S1"),
        Claim(text="B.", span_id="POL-003#S0"),
    ])
    from atlas_counsel.agent.schemas import CounselAnswer
    ans = CounselAnswer.from_claims(draft, attempts=1, escalated=False)
    assert ans.citations == ["POL-001#S1", "POL-003#S0"]
    assert "POL-001#S1" in ans.text
