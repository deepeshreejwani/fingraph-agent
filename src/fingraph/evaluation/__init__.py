"""Evaluation utilities for FinGraph."""

from fingraph.evaluation.failures import (
    FailureCategory,
    classify_failures,
)
from fingraph.evaluation.retrieval import (
    RelevanceSpec,
    ndcg_at_k,
    precision_at_k,
    recall_at_k_explicit,
    reciprocal_rank,
)

__all__ = [
    "FailureCategory",
    "RelevanceSpec",
    "classify_failures",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k_explicit",
    "reciprocal_rank",
]
