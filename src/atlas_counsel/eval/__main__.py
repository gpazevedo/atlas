"""Run the eval harness offline (in-memory retriever) and print the report.

    python -m atlas_counsel.eval
    python -m atlas_counsel.eval --ab     # A/B two embedding spaces
    python -m atlas_counsel.eval --meta   # meta-evaluation: judge bias report
"""
from __future__ import annotations

import argparse

from ..chunking import chunk_corpus
from ..corpus import build_corpus
from ..embeddings import HashingEmbedder
from ..retrieval import InMemoryHybridRetriever
from . import (
    evaluate, meta_evaluate, render_ab_table, render_bias_report,
    render_by_tag, maybe_log_to_langfuse,
)
from .judge import HeuristicJudge


def _build(space_id: str, dense_dim: int):
    chunks = chunk_corpus(build_corpus())
    r = InMemoryHybridRetriever(HashingEmbedder(space_id=space_id, dense_dim=dense_dim))
    r.index(chunks)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ab", action="store_true",
                    help="compare two embedding configs")
    ap.add_argument("--meta", action="store_true",
                    help="run LLM-as-judge meta-evaluation (bias report)")
    args = ap.parse_args()

    golden = build_corpus().golden

    rA = _build("hashing-256", 256)
    repA = evaluate(rA, golden, label="hashing-256", k=args.k)

    print("Aggregate:", repA.aggregate)
    print()
    print(render_by_tag(repA))

    if args.ab:
        rB = _build("hashing-512", 512)
        repB = evaluate(rB, golden, label="hashing-512", k=args.k)
        print()
        print(render_ab_table(repA, repB))

    if args.meta:
        print()
        bias = meta_evaluate(HeuristicJudge(), rA, golden, repA)
        print(render_bias_report(bias))

    if maybe_log_to_langfuse(repA):
        print("\n(scores pushed to Langfuse)")


if __name__ == "__main__":
    main()
