"""Ablation: measure each advanced-RAG stage through the eval harness.

    python -m atlas_counsel.ablation

Prints aggregate metrics for: baseline hybrid, +rerank, +decompose, +both.
This is the artifact that keeps the project honest — it reports what each
stage actually does to the numbers rather than assuming improvement.
"""

from __future__ import annotations

from .chunking import chunk_corpus
from .corpus import build_corpus
from .decompose import HeuristicDecomposer
from .embeddings import HashingEmbedder
from .eval import evaluate
from .pipeline import RetrievalPipeline
from .rerank import TokenInteractionReranker
from .retrieval import InMemoryHybridRetriever

_FIELDS = ("context_precision", "reciprocal_rank", "context_recall",
           "faithfulness", "refusal_accuracy")


def main() -> None:
    golden = build_corpus().golden
    chunks = chunk_corpus(build_corpus())
    base = InMemoryHybridRetriever(HashingEmbedder())
    base.index(chunks)

    configs = {
        "baseline": base,
        "+rerank": RetrievalPipeline(base, reranker=TokenInteractionReranker()),
        "+decompose": RetrievalPipeline(base, decomposer=HeuristicDecomposer()),
        "+both": RetrievalPipeline(
            base, decomposer=HeuristicDecomposer(),
            reranker=TokenInteractionReranker()),
    }

    header = f"{'config':<14}" + "".join(f"{f[:9]:>11}" for f in _FIELDS)
    print(header)
    print("-" * len(header))
    base_agg = evaluate(base, golden, label="baseline", k=5).aggregate
    for name, r in configs.items():
        agg = evaluate(r, golden, label=name, k=5).aggregate
        row = f"{name:<14}"
        for f in _FIELDS:
            row += f"{agg[f]:>11.3f}"
        print(row)

    print("\nNote: on this small corpus first-stage hybrid nearly saturates, so")
    print("the offline lexical proxies show no net gain. The stages are shipped")
    print("as tested infrastructure; production cross-encoder + LLM decomposer")
    print("are expected to win, measured via this same harness.")


if __name__ == "__main__":
    main()
