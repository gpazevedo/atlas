"""Agent state.

A single TypedDict threaded through every node and persisted by the
checkpointer. Nodes return partial updates; LangGraph merges them. Keeping the
state flat and explicit is what makes checkpointing, interrupt/resume, and
time-travel debugging work.
"""

from __future__ import annotations

from typing import TypedDict

from .schemas import CounselAnswer, DraftAnswer, GroundingVerdict
from ..retrieval import RetrievedChunk


class CounselState(TypedDict, total=False):
    # --- inputs ---
    question: str

    # --- plan ---
    sub_queries: list[str]

    # --- retrieve ---
    retrieved: list[RetrievedChunk]

    # --- validate ---
    grounded: bool          # did retrieval surface enough to answer?

    # --- synthesize / verify loop ---
    draft: DraftAnswer
    verdict: GroundingVerdict
    attempts: int           # synthesize attempts so far (bounds the retry loop)

    # --- gap-aware retrieval ---
    gap_iterations: int     # how many gap-analysis rounds completed

    # --- human-gate ---
    hitl_enabled: bool       # if False, skip the gate and refuse instead
    escalated: bool
    human_input: str | None  # guidance / decision provided on resume

    # --- output ---
    answer: CounselAnswer
