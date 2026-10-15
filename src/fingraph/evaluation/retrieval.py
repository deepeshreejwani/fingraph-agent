"""Deterministic retrieval evaluation metrics."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from fingraph.retrieval.models import RetrievalResult


@dataclass(frozen=True)
class RelevanceSpec:
    """Bootstrap or explicit relevance definition for one retrieval question."""

    tickers: frozenset[str] = frozenset()
    fiscal_years: frozenset[int] = frozenset()
    section_names: frozenset[str] = frozenset()
    relevant_chunk_ids: frozenset[str] = frozenset()


def is_relevant(
    result: RetrievalResult,
    relevance: RelevanceSpec,
) -> bool:
    """Determine whether a retrieval result satisfies the relevance contract."""

    if relevance.relevant_chunk_ids:
        return result.chunk_id in relevance.relevant_chunk_ids

    if relevance.tickers and result.ticker not in relevance.tickers:
        return False

    if relevance.fiscal_years and result.fiscal_year not in relevance.fiscal_years:
        return False

    return not (relevance.section_names and result.section_name not in relevance.section_names)


def precision_at_k(
    results: Sequence[RetrievalResult],
    relevance: RelevanceSpec,
    *,
    k: int,
) -> float:
    """Compute fixed-cutoff Precision@K."""

    if k <= 0:
        raise ValueError("k must be positive.")

    retrieved = results[:k]

    relevant_count = sum(
        is_relevant(
            result,
            relevance,
        )
        for result in retrieved
    )

    return relevant_count / k


def reciprocal_rank(
    results: Sequence[RetrievalResult],
    relevance: RelevanceSpec,
    *,
    k: int,
) -> float:
    """Compute reciprocal rank of the first relevant result."""

    if k <= 0:
        raise ValueError("k must be positive.")

    for rank, result in enumerate(
        results[:k],
        start=1,
    ):
        if is_relevant(
            result,
            relevance,
        ):
            return 1.0 / rank

    return 0.0


def dcg_at_k(
    results: Sequence[RetrievalResult],
    relevance: RelevanceSpec,
    *,
    k: int,
) -> float:
    """Compute binary-relevance discounted cumulative gain."""

    if k <= 0:
        raise ValueError("k must be positive.")

    score = 0.0

    for rank, result in enumerate(
        results[:k],
        start=1,
    ):
        gain = (
            1.0
            if is_relevant(
                result,
                relevance,
            )
            else 0.0
        )

        if gain:
            score += gain / math.log2(rank + 1)

    return score


def ndcg_at_k(
    results: Sequence[RetrievalResult],
    relevance: RelevanceSpec,
    *,
    k: int,
) -> float | None:
    """
    Compute binary NDCG@K when explicit chunk-level qrels exist.

    Metadata-only bootstrap relevance does not define the complete relevant
    document universe, so a valid ideal DCG cannot be constructed.
    """

    if k <= 0:
        raise ValueError("k must be positive.")

    if not relevance.relevant_chunk_ids:
        return None

    actual = dcg_at_k(
        results,
        relevance,
        k=k,
    )

    ideal_relevant = min(
        len(relevance.relevant_chunk_ids),
        k,
    )

    if ideal_relevant == 0:
        return 0.0

    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_relevant + 1,
        )
    )

    return actual / ideal


def recall_at_k_explicit(
    results: Sequence[RetrievalResult],
    relevance: RelevanceSpec,
    *,
    k: int,
) -> float | None:
    """
    Compute Recall@K only when explicit chunk-level judgments exist.

    Metadata-only bootstrap judgments do not define the complete number
    of relevant chunks, so reporting Recall@K for them would be misleading.
    """

    if k <= 0:
        raise ValueError("k must be positive.")

    if not relevance.relevant_chunk_ids:
        return None

    retrieved_ids = {result.chunk_id for result in results[:k]}

    hits = len(retrieved_ids & relevance.relevant_chunk_ids)

    return hits / len(relevance.relevant_chunk_ids)
