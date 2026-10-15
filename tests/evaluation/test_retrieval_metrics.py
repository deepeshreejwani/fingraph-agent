"""Regression tests for retrieval evaluation metric correctness."""

import math
from types import SimpleNamespace
from typing import cast

import pytest

from fingraph.evaluation.retrieval import (
    RelevanceSpec,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k_explicit,
    reciprocal_rank,
)
from fingraph.retrieval.models import RetrievalResult


def _result(
    chunk_id: str,
    *,
    rank: int,
    ticker: str = "MSFT",
    fiscal_year: int = 2026,
    section_name: str = "Item 1A. Risk Factors",
) -> RetrievalResult:
    """Create the minimal retrieval-result shape required by metric tests."""

    return cast(
        RetrievalResult,
        SimpleNamespace(
            chunk_id=chunk_id,
            rank=rank,
            ticker=ticker,
            fiscal_year=fiscal_year,
            section_name=section_name,
        ),
    )


def test_precision_at_k_uses_fixed_k_denominator() -> None:
    relevance = RelevanceSpec(
        relevant_chunk_ids=frozenset(
            {
                "relevant-1",
            }
        )
    )

    results = [
        _result(
            "relevant-1",
            rank=1,
        )
    ]

    assert precision_at_k(
        results,
        relevance,
        k=5,
    ) == pytest.approx(0.2)


def test_precision_at_k_empty_results_is_zero() -> None:
    relevance = RelevanceSpec(
        relevant_chunk_ids=frozenset(
            {
                "relevant-1",
            }
        )
    )

    assert (
        precision_at_k(
            [],
            relevance,
            k=5,
        )
        == 0.0
    )


def test_precision_at_k_counts_only_top_k() -> None:
    relevance = RelevanceSpec(
        relevant_chunk_ids=frozenset(
            {
                "relevant-1",
                "relevant-3",
            }
        )
    )

    results = [
        _result(
            "relevant-1",
            rank=1,
        ),
        _result(
            "irrelevant-2",
            rank=2,
        ),
        _result(
            "relevant-3",
            rank=3,
        ),
    ]

    assert precision_at_k(
        results,
        relevance,
        k=2,
    ) == pytest.approx(0.5)


def test_explicit_qrels_define_ideal_dcg() -> None:
    relevance = RelevanceSpec(
        relevant_chunk_ids=frozenset(
            {
                "relevant-1",
                "relevant-2",
                "relevant-3",
            }
        )
    )

    results = [
        _result(
            "relevant-1",
            rank=1,
        )
    ]

    expected_ideal = 1.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4)

    expected_ndcg = 1.0 / expected_ideal

    score = ndcg_at_k(
        results,
        relevance,
        k=3,
    )

    assert score is not None

    assert score == pytest.approx(expected_ndcg)

    assert score < 1.0


def test_perfect_explicit_qrel_ranking_has_ndcg_one() -> None:
    relevance = RelevanceSpec(
        relevant_chunk_ids=frozenset(
            {
                "relevant-1",
                "relevant-2",
                "relevant-3",
            }
        )
    )

    results = [
        _result(
            "relevant-1",
            rank=1,
        ),
        _result(
            "relevant-2",
            rank=2,
        ),
        _result(
            "relevant-3",
            rank=3,
        ),
    ]

    assert ndcg_at_k(
        results,
        relevance,
        k=3,
    ) == pytest.approx(1.0)


def test_dcg_discount_penalizes_later_relevant_result() -> None:
    relevance = RelevanceSpec(
        relevant_chunk_ids=frozenset(
            {
                "relevant",
            }
        )
    )

    results = [
        _result(
            "irrelevant",
            rank=1,
        ),
        _result(
            "relevant",
            rank=2,
        ),
    ]

    assert dcg_at_k(
        results,
        relevance,
        k=2,
    ) == pytest.approx(1.0 / math.log2(3))


def test_metadata_only_ndcg_is_undefined() -> None:
    relevance = RelevanceSpec(
        tickers=frozenset(
            {
                "MSFT",
            }
        ),
        fiscal_years=frozenset(
            {
                2026,
            }
        ),
    )

    results = [
        _result(
            "msft-2026",
            rank=1,
        )
    ]

    assert (
        ndcg_at_k(
            results,
            relevance,
            k=5,
        )
        is None
    )


def test_metadata_only_recall_is_undefined() -> None:
    relevance = RelevanceSpec(
        tickers=frozenset(
            {
                "MSFT",
            }
        )
    )

    results = [
        _result(
            "msft-result",
            rank=1,
        )
    ]

    assert (
        recall_at_k_explicit(
            results,
            relevance,
            k=5,
        )
        is None
    )


def test_explicit_recall_uses_complete_qrel_set() -> None:
    relevance = RelevanceSpec(
        relevant_chunk_ids=frozenset(
            {
                "relevant-1",
                "relevant-2",
                "relevant-3",
                "relevant-4",
            }
        )
    )

    results = [
        _result(
            "relevant-1",
            rank=1,
        ),
        _result(
            "irrelevant",
            rank=2,
        ),
        _result(
            "relevant-3",
            rank=3,
        ),
    ]

    assert recall_at_k_explicit(
        results,
        relevance,
        k=3,
    ) == pytest.approx(0.5)


def test_metadata_relevance_still_supports_precision_and_mrr() -> None:
    relevance = RelevanceSpec(
        tickers=frozenset(
            {
                "MSFT",
            }
        ),
        fiscal_years=frozenset(
            {
                2026,
            }
        ),
        section_names=frozenset(
            {
                "Item 1A. Risk Factors",
            }
        ),
    )

    results = [
        _result(
            "wrong-company",
            rank=1,
            ticker="NVDA",
        ),
        _result(
            "correct",
            rank=2,
        ),
    ]

    assert precision_at_k(
        results,
        relevance,
        k=5,
    ) == pytest.approx(0.2)

    assert reciprocal_rank(
        results,
        relevance,
        k=5,
    ) == pytest.approx(0.5)


def test_cutoff_metrics_reject_non_positive_k() -> None:
    relevance = RelevanceSpec(
        relevant_chunk_ids=frozenset(
            {
                "relevant",
            }
        )
    )

    with pytest.raises(
        ValueError,
        match="k must be positive",
    ):
        precision_at_k(
            [],
            relevance,
            k=0,
        )

    with pytest.raises(
        ValueError,
        match="k must be positive",
    ):
        reciprocal_rank(
            [],
            relevance,
            k=0,
        )

    with pytest.raises(
        ValueError,
        match="k must be positive",
    ):
        dcg_at_k(
            [],
            relevance,
            k=0,
        )

    with pytest.raises(
        ValueError,
        match="k must be positive",
    ):
        ndcg_at_k(
            [],
            relevance,
            k=0,
        )

    with pytest.raises(
        ValueError,
        match="k must be positive",
    ):
        recall_at_k_explicit(
            [],
            relevance,
            k=0,
        )
