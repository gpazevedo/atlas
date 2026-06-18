"""Query decomposition.

Multi-hop questions ("compare AcmeCloud and NorthLink uptime") need spans from
multiple documents. A single retrieval often surfaces one entity's spans and
starves the other. Decomposing into sub-queries, retrieving each, and merging
recovers the missing side.

Risk: over-splitting a simple query degrades it. So decomposition is
CONDITIONAL — a query is only split when it shows multi-part structure
(coordinating "and"/"vs"/"compare" over distinct named entities). Simple
queries pass through untouched. The eval harness measures both slices so we
can confirm multi-hop improves without simple-query regression.

`QueryDecomposer` is the protocol. The offline `HeuristicDecomposer` uses
surface cues + the known vendor list; a production decomposer swaps in an LLM
behind the same protocol.
"""

from __future__ import annotations

import re
from typing import Protocol

from .corpus.generator import VENDORS

_VENDOR_NAMES = [v["name"] for v in VENDORS]
_COMPARE_CUES = ("compare", " vs ", " versus ", "difference between", "both")


class QueryDecomposer(Protocol):
    def decompose(self, query: str) -> list[str]: ...


class HeuristicDecomposer:
    """Deterministic decomposer.

    Splits only when the query mentions >= 2 known entities AND carries a
    comparison cue (or an 'and' joining the entities). Returns the original
    query unchanged otherwise, so single-hop queries are never harmed.
    """

    def decompose(self, query: str) -> list[str]:
        mentioned = [v for v in _VENDOR_NAMES if v.lower() in query.lower()]
        lower = query.lower()
        has_cue = any(cue in lower for cue in _COMPARE_CUES) or (
            len(mentioned) >= 2 and " and " in lower
        )
        if len(mentioned) >= 2 and has_cue:
            # Build a focused sub-query per entity, preserving the asked aspect.
            aspect = self._aspect(query)
            return [f"{v} {aspect}".strip() for v in mentioned]
        return [query]

    @staticmethod
    def _aspect(query: str) -> str:
        """Strip entity names and comparison scaffolding to leave the aspect
        (e.g. 'uptime guarantee'). Best-effort; falls back to the full query."""
        aspect = query
        for v in _VENDOR_NAMES:
            aspect = re.sub(re.escape(v), "", aspect, flags=re.IGNORECASE)
        junk_words = (
            "compare", "the", "and", "vs", "versus", "difference", "between",
            "of", "both", "guarantees", "caps",
        )
        for junk in junk_words:
            aspect = re.sub(rf"\b{re.escape(junk)}\b", " ", aspect, flags=re.IGNORECASE)
        aspect = re.sub(r"[?.]", " ", aspect)
        aspect = re.sub(r"\s+", " ", aspect).strip()
        return aspect or query
