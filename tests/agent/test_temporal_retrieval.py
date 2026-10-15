"""Temporal semantics for narrative filing retrieval."""

from typing import Any

from fingraph.agent.tools import (
    VectorEvidenceTool,
)
from fingraph.retrieval.models import (
    RetrievalFilter,
)


class FakeStore:
    """Minimal retrieval store recording applied metadata filters."""

    def __init__(
        self,
        *,
        latest_year: int | None = 2026,
    ) -> None:
        self.latest_year = latest_year

        self.latest_calls: list[str] = []

        self.sparse_filters: list[RetrievalFilter | None] = []

        self.dense_filters: list[RetrievalFilter | None] = []

    def latest_fiscal_year(
        self,
        ticker: str,
    ) -> int | None:
        self.latest_calls.append(ticker)

        return self.latest_year

    def sparse_search(
        self,
        query: str,
        *,
        filters: RetrievalFilter | None = None,
        limit: int = 10,
    ) -> list[Any]:
        del query
        del limit

        self.sparse_filters.append(filters)

        return []

    def dense_search(
        self,
        query: str,
        *,
        filters: RetrievalFilter | None = None,
        limit: int = 10,
    ) -> list[Any]:
        del query
        del limit

        self.dense_filters.append(filters)

        return []


def _filter_for(
    *,
    question: str,
    latest_year: int | None = 2026,
) -> tuple[
    FakeStore,
    RetrievalFilter,
]:
    store = FakeStore(latest_year=(latest_year))

    tool = VectorEvidenceTool(
        store,  # type: ignore[arg-type]
        limit=5,
    )

    result = tool(question)

    assert result == []

    assert len(store.sparse_filters) == 1

    assert len(store.dense_filters) == 1

    sparse_filter = store.sparse_filters[0]

    dense_filter = store.dense_filters[0]

    assert sparse_filter == dense_filter

    assert sparse_filter is not None

    return (
        store,
        sparse_filter,
    )


def test_unqualified_single_company_query_uses_latest_year() -> None:
    (
        store,
        retrieval_filter,
    ) = _filter_for(question=("What risks does NVIDIA describe in its 10-K?"))

    assert retrieval_filter.tickers == ("NVDA",)

    assert retrieval_filter.fiscal_years == (2026,)

    assert retrieval_filter.section_names == ("Item 1A. Risk Factors",)

    assert store.latest_calls == ["NVDA"]


def test_explicit_year_overrides_latest() -> None:
    (
        store,
        retrieval_filter,
    ) = _filter_for(question=("What risks did NVIDIA describe in its 2022 10-K?"))

    assert retrieval_filter.fiscal_years == (2022,)

    assert store.latest_calls == []


def test_explicit_year_range_expands_inclusively() -> None:
    (
        store,
        retrieval_filter,
    ) = _filter_for(question=("How did Microsoft's strategy change from 2024 to 2026?"))

    assert retrieval_filter.fiscal_years == (
        2024,
        2025,
        2026,
    )

    assert store.latest_calls == []


def test_between_range_expands_inclusively() -> None:
    (
        _,
        retrieval_filter,
    ) = _filter_for(question=("Compare NVIDIA risks between 2023 and 2025."))

    assert retrieval_filter.fiscal_years == (
        2023,
        2024,
        2025,
    )


def test_unbounded_multi_year_query_does_not_force_latest() -> None:
    (
        store,
        retrieval_filter,
    ) = _filter_for(question=("How have NVIDIA's risks changed over the years?"))

    assert retrieval_filter.tickers == ("NVDA",)

    assert retrieval_filter.fiscal_years == ()

    assert store.latest_calls == []


def test_latest_year_unavailable_leaves_year_unrestricted() -> None:
    (
        store,
        retrieval_filter,
    ) = _filter_for(
        question=("Describe Amazon's business."),
        latest_year=None,
    )

    assert retrieval_filter.tickers == ("AMZN",)

    assert retrieval_filter.fiscal_years == ()

    assert store.latest_calls == ["AMZN"]
