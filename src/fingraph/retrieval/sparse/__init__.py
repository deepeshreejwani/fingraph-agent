"""Sparse retrieval."""

from fingraph.retrieval.sparse.bm25 import (
    DEFAULT_SPARSE_MODEL,
    BM25Embedder,
)

__all__ = [
    "BM25Embedder",
    "DEFAULT_SPARSE_MODEL",
]
