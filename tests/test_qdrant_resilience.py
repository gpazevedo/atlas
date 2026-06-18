"""Qdrant resilience tests — retry/timeout without a real Qdrant server.

These test the tenacity retry decorator and QDRANT_TIMEOUT wiring by injecting
mock clients. No Qdrant server required.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client", reason="qdrant-client not installed")

from atlas_counsel.embeddings import HashingEmbedder
from atlas_counsel.qdrant_store import QDRANT_TIMEOUT, QdrantHybridRetriever


def test_qdrant_retries_on_connection_error_then_succeeds():
    class _FlakyClient:
        def __init__(self):
            self.attempts = 0

        def collection_exists(self, name):
            self.attempts += 1
            if self.attempts < 2:
                raise ConnectionError("transient")
            return False

        def create_collection(self, **kwargs):
            pass

    provider = HashingEmbedder(space_id="retry-ok")
    retriever = QdrantHybridRetriever(provider, client=_FlakyClient())
    retriever.ensure_collection()
    assert retriever._client.attempts == 2


def test_qdrant_retry_exhausts_and_raises():
    class _AlwaysFailingClient:
        def __init__(self):
            self.attempts = 0

        def collection_exists(self, name):
            self.attempts += 1
            raise ConnectionError("persistent")

    provider = HashingEmbedder(space_id="retry-exhaust")
    retriever = QdrantHybridRetriever(provider, client=_AlwaysFailingClient())

    with pytest.raises(ConnectionError):
        retriever.ensure_collection()
    assert retriever._client.attempts == 3


def test_qdrant_does_not_retry_on_unexpected_error():
    class _ValueErrorClient:
        def __init__(self):
            self.attempts = 0

        def collection_exists(self, name):
            self.attempts += 1
            raise ValueError("not retryable")

    provider = HashingEmbedder(space_id="retry-nope")
    retriever = QdrantHybridRetriever(provider, client=_ValueErrorClient())

    with pytest.raises(ValueError):
        retriever.ensure_collection()
    assert retriever._client.attempts == 1


def test_qdrant_client_is_created_with_timeout():
    provider = HashingEmbedder()
    retriever = QdrantHybridRetriever(provider)
    client = retriever._ensure_client()
    assert client._client._timeout == QDRANT_TIMEOUT
