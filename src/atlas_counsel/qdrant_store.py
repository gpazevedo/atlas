"""Production hybrid retriever backed by Qdrant.

Uses Qdrant's native hybrid search: one collection holds a named dense vector
and a named sparse vector per point, and the Query API fuses them server-side
with RRF via prefetch. This is the path you run locally / in prod.

The collection name is derived from the provider's `space_id` so a bge-m3
index and a Titan index are physically separate collections — you cannot
accidentally query one embedding space against the other.

Requires: `pip install qdrant-client` and a running Qdrant
(e.g. `docker run -p 6333:6333 qdrant/qdrant`).
"""

from __future__ import annotations

from .chunking import Chunk
from .embeddings import EmbeddingProvider
from .retrieval import RetrievedChunk

DENSE_VEC = "dense"
SPARSE_VEC = "sparse"


class QdrantHybridRetriever:
    def __init__(
        self,
        provider: EmbeddingProvider,
        url: str = "http://localhost:6333",
        rrf_k: int = 60,
        client=None,
    ) -> None:
        self._provider = provider
        self._rrf_k = rrf_k
        self._url = url
        self._client = client  # injectable for testing

    @property
    def collection_name(self) -> str:
        return f"counsel_{self._provider.space_id}"

    def _ensure_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient  # local import: optional dep

            self._client = QdrantClient(url=self._url)
        return self._client

    def ensure_collection(self) -> None:
        from qdrant_client import models as qm

        client = self._ensure_client()
        name = self.collection_name
        if client.collection_exists(name):
            return
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VEC: qm.VectorParams(
                    size=self._provider.dense_dim,
                    distance=qm.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VEC: qm.SparseVectorParams(
                    index=qm.SparseIndexParams()
                )
            },
        )

    def index(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        from qdrant_client import models as qm

        client = self._ensure_client()
        self.ensure_collection()
        embs = self._provider.embed([c.embed_text for c in chunks])

        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embs)):
            points.append(
                qm.PointStruct(
                    id=i,
                    vector={
                        DENSE_VEC: emb.dense,
                        SPARSE_VEC: qm.SparseVector(
                            indices=emb.sparse.indices,
                            values=emb.sparse.values,
                        ),
                    },
                    # Store the chunk so retrieval reconstructs citations.
                    payload=chunk.model_dump(mode="json"),
                )
            )
        for start in range(0, len(points), batch_size):
            client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + batch_size],
            )

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        from qdrant_client import models as qm

        client = self._ensure_client()
        q = self._provider.embed([query])[0]

        # Prefetch each channel, then fuse server-side with RRF.
        response = client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                qm.Prefetch(
                    query=q.dense,
                    using=DENSE_VEC,
                    limit=max(top_k * 4, 20),
                ),
                qm.Prefetch(
                    query=qm.SparseVector(
                        indices=q.sparse.indices,
                        values=q.sparse.values,
                    ),
                    using=SPARSE_VEC,
                    limit=max(top_k * 4, 20),
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        results: list[RetrievedChunk] = []
        for point in response.points:
            results.append(
                RetrievedChunk(
                    chunk=Chunk.model_validate(point.payload),
                    score=point.score,
                )
            )
        return results
