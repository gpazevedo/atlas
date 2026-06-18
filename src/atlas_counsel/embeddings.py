"""Embedding provider abstraction.

Design goals, in priority order:

1. **Hybrid-native.** A provider yields BOTH a dense vector and a sparse
   vector per text. Sparse (lexical) catches exact tokens like "$25,000"
   that dense embeddings smear together — the threshold-precision trap.

2. **Vector-space safety.** Local (bge-m3, 1024-d) and Bedrock (Titan,
   1024-d but a *different space*) are NOT interchangeable even when dims
   match. We never let them share a collection. Each provider declares a
   `space_id`, and the retriever derives the Qdrant collection name from
   it, so cross-space contamination is structurally impossible, not a
   thing you have to remember.

3. **Offline-testable.** `HashingEmbedder` is a deterministic, dependency-
   free provider used by unit tests and CI. It is real enough to exercise
   fusion and ranking logic without a model download or network.

Prod/dev swap is config: pick the provider, the collection name follows.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class SparseVector(BaseModel):
    """Qdrant-style sparse vector: parallel indices/values arrays."""

    indices: list[int]
    values: list[float]


class Embedding(BaseModel):
    dense: list[float]
    sparse: SparseVector


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def space_id(self) -> str:
        """Stable identifier of this embedding space, e.g. 'bge-m3' or
        'titan-v2'. Drives collection naming. Two providers with the same
        space_id are assumed interchangeable; different => never mixed."""
        ...

    @property
    def dense_dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[Embedding]: ...


_TOKEN_RE = re.compile(r"[A-Za-z0-9$%.,]+")


def _tokenize(text: str) -> list[str]:
    # Keep $ % . , so "$25,000" and "99.9%" survive as single lexical units —
    # exactly the tokens the threshold/contradiction traps hinge on.
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _hash_index(token: str, vocab_size: int) -> int:
    h = hashlib.blake2b(token.encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % vocab_size


class HashingEmbedder:
    """Deterministic, offline embedder for tests and CI.

    Dense:  hashed bag-of-tokens projected into `dense_dim`, L2-normalized.
            Cosine similarity then tracks lexical overlap closely enough to
            validate ranking and fusion behavior.
    Sparse: hashed-token term frequencies (a BM25-flavored lexical signal).

    NOT for production quality — it's a stand-in so the pipeline runs and is
    tested without a real model. Real providers (bge-m3, Titan) implement the
    same Protocol.
    """

    def __init__(self, dense_dim: int = 256, vocab_size: int = 2**16,
                 space_id: str = "hashing-v1") -> None:
        self._dim = dense_dim
        self._vocab = vocab_size
        self._space_id = space_id

    @property
    def space_id(self) -> str:
        return self._space_id

    @property
    def dense_dim(self) -> int:
        return self._dim

    def _dense_one(self, tokens: list[str]) -> list[float]:
        vec = [0.0] * self._dim
        for tok in tokens:
            idx = _hash_index(tok, self._dim)
            # signed contribution so distinct tokens can cancel/reinforce
            sign = 1.0 if _hash_index(tok + "#s", 2) == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _sparse_one(self, tokens: list[str]) -> SparseVector:
        tf: dict[int, float] = {}
        for tok in tokens:
            idx = _hash_index(tok, self._vocab)
            tf[idx] = tf.get(idx, 0.0) + 1.0
        items = sorted(tf.items())
        return SparseVector(indices=[i for i, _ in items],
                            values=[v for _, v in items])

    def embed(self, texts: list[str]) -> list[Embedding]:
        out: list[Embedding] = []
        for t in texts:
            toks = _tokenize(t)
            out.append(Embedding(dense=self._dense_one(toks),
                                 sparse=self._sparse_one(toks)))
        return out
