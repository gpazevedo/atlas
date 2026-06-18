"""Qdrant integration test.

Skipped automatically unless a Qdrant server is reachable (env QDRANT_URL or
localhost:6333) and qdrant-client is installed. Run locally with:

    docker run -p 6333:6333 qdrant/qdrant
    uv sync --extra qdrant
    uv run pytest tests/test_qdrant_integration.py -v
"""

from __future__ import annotations

import os

import pytest

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def _qdrant_available() -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen(f"{QDRANT_URL}/readyz", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


pytest.importorskip("qdrant_client", reason="qdrant-client not installed")

pytestmark = pytest.mark.skipif(
    not _qdrant_available(), reason="no Qdrant server reachable"
)


def test_qdrant_hybrid_roundtrip():
    from atlas_counsel.chunking import chunk_corpus
    from atlas_counsel.corpus import build_corpus
    from atlas_counsel.embeddings import HashingEmbedder
    from atlas_counsel.qdrant_store import QdrantHybridRetriever

    provider = HashingEmbedder(space_id="itest")
    retriever = QdrantHybridRetriever(provider, url=QDRANT_URL)

    # Clean slate for a deterministic test.
    client = retriever._ensure_client()
    if client.collection_exists(retriever.collection_name):
        client.delete_collection(retriever.collection_name)

    chunks = chunk_corpus(build_corpus())
    retriever.index(chunks)

    results = retriever.search(
        "single-source justification threshold", top_k=3
    )
    assert results
    assert any(r.chunk.span_id == "POL-001#S1" for r in results)
    # Citations survive the round-trip through Qdrant payloads.
    for r in results:
        assert "#S" in r.chunk.span_id
