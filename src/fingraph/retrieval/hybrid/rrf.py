"""Reciprocal Rank Fusion."""

from collections import defaultdict

from fingraph.retrieval.models import (
    RetrievalResult,
)


def reciprocal_rank_fusion(
    sparse_results: list[RetrievalResult],
    dense_results: list[RetrievalResult],
    *,
    k: int = 60,
    limit: int = 10,
) -> list[RetrievalResult]:
    """Fuse independent rankings using Reciprocal Rank Fusion."""

    if k <= 0:
        raise ValueError("RRF k must be positive.")

    if limit <= 0:
        raise ValueError("limit must be positive.")

    scores: dict[str, float] = defaultdict(float)

    results_by_id: dict[
        str,
        RetrievalResult,
    ] = {}

    for result_set in (
        sparse_results,
        dense_results,
    ):
        for result in result_set:
            scores[result.chunk_id] += 1.0 / (k + result.rank)

            results_by_id[result.chunk_id] = result

    ranked_ids = sorted(
        scores,
        key=scores.__getitem__,
        reverse=True,
    )[:limit]

    fused: list[RetrievalResult] = []

    for rank, chunk_id in enumerate(
        ranked_ids,
        start=1,
    ):
        original = results_by_id[chunk_id]

        fused.append(
            original.model_copy(
                update={
                    "score": scores[chunk_id],
                    "method": "hybrid",
                    "rank": rank,
                }
            )
        )

    return fused
