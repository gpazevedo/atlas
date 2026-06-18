"""Ingest CLI: build the corpus, chunk it, and index into Qdrant.

    python -m atlas_counsel.ingest --url http://localhost:6333

With --dry-run, runs the in-memory retriever instead (no Qdrant needed) and
prints a sample search, so you can sanity-check the pipeline offline.
"""
from __future__ import annotations

import argparse

from .chunking import chunk_corpus
from .corpus import build_corpus
from .embeddings import HashingEmbedder
from .retrieval import InMemoryHybridRetriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--space-id", default="hashing-v1")
    ap.add_argument("--dry-run", action="store_true",
                    help="use in-memory retriever, no Qdrant")
    ap.add_argument("--query", default="single-source justification threshold")
    args = ap.parse_args()

    chunks = chunk_corpus(build_corpus())
    provider = HashingEmbedder(space_id=args.space_id)

    if args.dry_run:
        r = InMemoryHybridRetriever(provider)
        r.index(chunks)
        print(f"[dry-run] indexed {len(chunks)} chunks into '{r.collection_name}' (in-memory)")
        hits = r.search(args.query, top_k=3)
    else:
        from .qdrant_store import QdrantHybridRetriever
        r = QdrantHybridRetriever(provider, url=args.url)
        r.index(chunks)
        print(f"indexed {len(chunks)} chunks into Qdrant collection '{r.collection_name}'")
        hits = r.search(args.query, top_k=3)

    print(f"\nquery: {args.query}")
    for h in hits:
        print(f"  {h.score:.4f}  {h.chunk.span_id:14} {h.chunk.title}")


if __name__ == "__main__":
    main()
