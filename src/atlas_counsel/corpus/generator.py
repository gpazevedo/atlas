"""Deterministic synthetic procurement corpus generator.

Design contract:
  * Fully seeded — same seed in, byte-identical corpus out.
  * No network, no LLM calls. The "real-ish" content is template-driven so
    the planted hard cases land in known places and the repo is public-safe.
  * Hard cases are explicit and tagged, so eval can prove the retriever and
    grounding logic actually handle them:
      - contradiction:        two vendors with conflicting SLA uptime
      - threshold-precision:  near-duplicate policies differing by one number
      - unanswerable:         a question whose answer is absent by design
      - cross-vendor:         multi-hop question spanning two contracts
"""

from __future__ import annotations

import random

from .models import (
    AnswerType,
    Corpus,
    DocCategory,
    Document,
    GoldenItem,
    Span,
)

# --- vendor fixtures ---------------------------------------------------------
# Uptime values are deliberately chosen so AcmeCloud and NorthLink CONFLICT
# (99.9 vs 99.5) — that pair is the planted contradiction.
VENDORS = [
    {"name": "AcmeCloud", "uptime": "99.9%", "net_terms": 30, "liability_cap": "12 months of fees"},
    {"name": "NorthLink", "uptime": "99.5%", "net_terms": 45, "liability_cap": "6 months of fees"},
    {"name": "Vertex Supply", "uptime": "99.95%", "net_terms": 60, "liability_cap": "fees paid in prior 12 months"},
]


def _policies() -> list[Document]:
    docs: list[Document] = []

    # POL-001: single-source justification threshold = $50,000
    docs.append(Document(
        doc_id="POL-001",
        category=DocCategory.POLICY,
        title="Single-Source Procurement Justification Policy",
        tags=["threshold-precision"],
        spans=[
            Span(ordinal=0, heading="Purpose",
                 text="This policy governs when a purchase may proceed without "
                      "competitive bidding."),
            Span(ordinal=1, heading="Threshold",
                 text="Any single-source purchase with a total contract value at "
                      "or above $50,000 requires a written justification approved "
                      "by the Procurement Director."),
            Span(ordinal=2, heading="Documentation",
                 text="The justification must state why no alternative supplier "
                      "can meet the requirement and be retained for seven years."),
        ],
    ))

    # POL-002: near-duplicate of POL-001 but for a DIFFERENT category and a
    # DIFFERENT threshold ($25,000). This is the reranker-precision trap:
    # surface-similar, but only one is correct for a given question.
    docs.append(Document(
        doc_id="POL-002",
        category=DocCategory.POLICY,
        title="Sole-Source IT Software Justification Policy",
        tags=["threshold-precision"],
        spans=[
            Span(ordinal=0, heading="Purpose",
                 text="This policy governs sole-source purchases of IT software "
                      "licenses specifically."),
            Span(ordinal=1, heading="Threshold",
                 text="Any sole-source IT software purchase with a total contract "
                      "value at or above $25,000 requires a written justification "
                      "approved by the Procurement Director."),
            Span(ordinal=2, heading="Documentation",
                 text="The justification must include a compatibility assessment "
                      "and be retained for seven years."),
        ],
    ))

    # POL-003: approval matrix
    docs.append(Document(
        doc_id="POL-003",
        category=DocCategory.POLICY,
        title="Procurement Approval Authority Matrix",
        spans=[
            Span(ordinal=0, heading="Tiers",
                 text="Purchases up to $10,000 may be approved by a Team Lead. "
                      "Purchases from $10,000 up to $100,000 require Procurement "
                      "Director approval. Purchases of $100,000 or more require "
                      "CFO approval."),
            Span(ordinal=1, heading="Splitting",
                 text="Splitting a purchase into multiple orders to stay below an "
                      "approval tier is prohibited and treated as a policy "
                      "violation."),
        ],
    ))

    # POL-004: conflict of interest
    docs.append(Document(
        doc_id="POL-004",
        category=DocCategory.POLICY,
        title="Procurement Conflict-of-Interest Policy",
        spans=[
            Span(ordinal=0, heading="Disclosure",
                 text="Any buyer with a personal or financial relationship to a "
                      "supplier must disclose it before participating in a "
                      "sourcing decision involving that supplier."),
            Span(ordinal=1, heading="Recusal",
                 text="A buyer who has disclosed a conflict must recuse themselves "
                      "from scoring or awarding the related contract."),
        ],
    ))
    return docs


def _contracts() -> list[Document]:
    docs: list[Document] = []
    for i, v in enumerate(VENDORS, start=1):
        docs.append(Document(
            doc_id=f"CON-{i:03d}",
            category=DocCategory.CONTRACT,
            title=f"Master Services Agreement — {v['name']}",
            vendor=v["name"],
            tags=["contract"] + (["contradiction"] if v["name"] in {"AcmeCloud", "NorthLink"} else []),
            spans=[
                Span(ordinal=0, heading="Service Level",
                     text=f"{v['name']} guarantees a monthly uptime of "
                          f"{v['uptime']}. Failure to meet this level entitles the "
                          f"buyer to service credits as described in Schedule B."),
                Span(ordinal=1, heading="Payment Terms",
                     text=f"Invoices are payable net {v['net_terms']} days from "
                          f"the date of receipt."),
                Span(ordinal=2, heading="Termination",
                     text=f"Either party may terminate this agreement for "
                          f"convenience with 90 days written notice."),
                Span(ordinal=3, heading="Liability",
                     text=f"{v['name']}'s aggregate liability is capped at "
                          f"{v['liability_cap']}."),
            ],
        ))
    return docs


def _negotiation_logs() -> list[Document]:
    return [Document(
        doc_id="NEG-001",
        category=DocCategory.NEGOTIATION_LOG,
        title="Negotiation Log — NorthLink Renewal 2025",
        vendor="NorthLink",
        tags=["precedent"],
        spans=[
            Span(ordinal=0, heading="Opening",
                 text="NorthLink opened with a 12% price increase citing "
                      "infrastructure costs."),
            Span(ordinal=1, heading="Outcome",
                 text="Buyer secured a 4% increase in exchange for extending the "
                      "term by 12 months and improving payment terms to net 45."),
        ],
    )]


def _golden(docs: list[Document]) -> list[GoldenItem]:
    """Hand-authored golden set referencing real span ids."""
    return [
        GoldenItem(
            qid="Q-001",
            question="Above what value does a single-source purchase need written "
                     "justification?",
            answer_type=AnswerType.GROUNDED,
            supporting_span_ids=["POL-001#S1"],
            reference_answer="A single-source purchase needs written justification "
                             "at or above $50,000 (POL-001).",
            tags=["threshold-precision"],
        ),
        GoldenItem(
            qid="Q-002",
            question="What is the justification threshold for a sole-source IT "
                     "software license purchase?",
            answer_type=AnswerType.GROUNDED,
            supporting_span_ids=["POL-002#S1"],
            reference_answer="$25,000 or above for sole-source IT software (POL-002) "
                             "— distinct from the general $50,000 single-source rule.",
            tags=["threshold-precision"],
        ),
        GoldenItem(
            qid="Q-003",
            question="Who must approve a $60,000 purchase?",
            answer_type=AnswerType.GROUNDED,
            supporting_span_ids=["POL-003#S0"],
            reference_answer="The Procurement Director, since $60,000 falls in the "
                             "$10k–$100k tier (POL-003).",
            tags=["approval"],
        ),
        GoldenItem(
            qid="Q-004",
            question="Compare the uptime guarantees of AcmeCloud and NorthLink.",
            answer_type=AnswerType.MULTI_HOP,
            supporting_span_ids=["CON-001#S0", "CON-002#S0"],
            reference_answer="AcmeCloud guarantees 99.9% (CON-001) while NorthLink "
                             "guarantees 99.5% (CON-002); AcmeCloud is higher.",
            tags=["contradiction", "cross-vendor"],
        ),
        GoldenItem(
            qid="Q-005",
            question="What are NorthLink's payment terms?",
            answer_type=AnswerType.GROUNDED,
            supporting_span_ids=["CON-002#S1"],
            reference_answer="Net 45 days from receipt of invoice (CON-002).",
            tags=["contract"],
        ),
        GoldenItem(
            qid="Q-006",
            question="What is our policy on accepting gifts from suppliers?",
            answer_type=AnswerType.UNANSWERABLE,
            supporting_span_ids=[],
            reference_answer="The corpus does not cover a supplier-gift policy; the "
                             "agent should decline to answer rather than guess.",
            tags=["unanswerable", "refusal"],
        ),
        GoldenItem(
            qid="Q-007",
            question="Can I split a $120,000 purchase into two $60,000 orders to "
                     "avoid CFO approval?",
            answer_type=AnswerType.GROUNDED,
            supporting_span_ids=["POL-003#S1", "POL-003#S0"],
            reference_answer="No — splitting to stay under a tier is prohibited "
                             "(POL-003); $120,000 requires CFO approval.",
            tags=["approval", "policy-reasoning"],
        ),
        GoldenItem(
            qid="Q-008",
            question="What liability cap did we agree to with Vertex Supply?",
            answer_type=AnswerType.GROUNDED,
            supporting_span_ids=["CON-003#S3"],
            reference_answer="Fees paid in the prior 12 months (CON-003).",
            tags=["contract"],
        ),
    ]


def build_corpus(seed: int = 42) -> Corpus:
    """Assemble and validate the full corpus. `seed` reserved for future
    randomized expansion; ordering is deterministic regardless."""
    random.seed(seed)
    docs = _policies() + _contracts() + _negotiation_logs()
    golden = _golden(docs)
    return Corpus(documents=docs, golden=golden)
