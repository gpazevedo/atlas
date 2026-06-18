"""Reporting: A/B table rendering + optional Langfuse export.

`render_ab_table` produces a plain-text comparison of two EvalReports (e.g.
local vs Bedrock provider) — the artifact the whole provider abstraction
exists to produce.

`maybe_log_to_langfuse` is a no-op unless langfuse is installed and configured
via env (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST). It pushes
per-item scores as a Langfuse trace so the same numbers show up in the
observability UI. Kept optional so CI never depends on it.
"""

from __future__ import annotations

import os

from .runner import EvalReport

_METRIC_ORDER = (
    "hit_at_k", "context_recall", "context_precision", "reciprocal_rank",
    "faithfulness", "answer_relevancy", "refusal_accuracy",
)


def render_ab_table(a: EvalReport, b: EvalReport) -> str:
    lines = []
    lines.append(f"A/B eval  (k={a.k})")
    lines.append(f"  A = {a.label}")
    lines.append(f"  B = {b.label}")
    lines.append("")
    w = max(len(m) for m in _METRIC_ORDER) + 2
    header = f"{'metric':<{w}}{'A':>8}{'B':>8}{'Δ(B-A)':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for m in _METRIC_ORDER:
        av = a.aggregate.get(m, 0.0)
        bv = b.aggregate.get(m, 0.0)
        lines.append(f"{m:<{w}}{av:>8.3f}{bv:>8.3f}{bv - av:>+10.3f}")
    return "\n".join(lines)


def render_by_tag(report: EvalReport) -> str:
    lines = [f"Per-tag breakdown — {report.label} (k={report.k})", ""]
    for tag, agg in report.by_tag.items():
        cp = agg.get("context_precision", 0.0)
        rr = agg.get("reciprocal_rank", 0.0)
        ra = agg.get("refusal_accuracy", 0.0)
        lines.append(f"  {tag:<22} ctx_prec={cp:.3f}  mrr={rr:.3f}  refusal_acc={ra:.3f}")
    return "\n".join(lines)


def maybe_log_to_langfuse(report: EvalReport) -> bool:
    """Push per-item scores to Langfuse if configured. Returns True if logged.
    Silent no-op otherwise so CI and offline runs never break."""
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return False
    try:
        from langfuse import Langfuse  # optional dep
    except ImportError:
        return False

    client = Langfuse()
    for it in report.items:
        trace = client.trace(name="atlas-counsel-eval", metadata={
            "label": report.label, "qid": it.qid,
            "answer_type": it.answer_type.value, "tags": it.tags,
        })
        for field in ("context_precision", "context_recall", "faithfulness",
                      "answer_relevancy", "reciprocal_rank"):
            val = getattr(it, field)
            if val is not None:
                trace.score(name=field, value=val)
        trace.score(name="refusal_correct", value=1.0 if it.refusal_correct else 0.0)
    client.flush()
    return True
