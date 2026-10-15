"""Longitudinal retrieval diversification tests."""

from fingraph.agent.tools import (
    _diversify_results_by_year,
    _select_representative_years,
)
from fingraph.retrieval.models import (
    RetrievalResult,
)


def _result(
    *,
    year: int,
    rank: int,
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"chunk-{year}-{rank}",
        filing_id=f"filing-{year}",
        section_id=f"section-{year}",
        cik="0001045810",
        ticker="NVDA",
        fiscal_year=year,
        section_name=("Item 1A. Risk Factors"),
        text=f"Risk evidence for {year}",
        source_url=(f"https://example.com/{year}"),
        score=score,
        method="hybrid",
        rank=rank,
    )


def test_representative_years_cover_timeline_endpoints() -> None:
    years = (
        2019,
        2020,
        2021,
        2022,
        2023,
        2024,
        2025,
        2026,
    )

    selected = _select_representative_years(
        years=years,
        limit=5,
    )

    assert selected == (
        2019,
        2021,
        2023,
        2024,
        2026,
    )


def test_representative_years_return_all_when_budget_allows() -> None:
    selected = _select_representative_years(
        years=(
            2024,
            2025,
            2026,
        ),
        limit=5,
    )

    assert selected == (
        2024,
        2025,
        2026,
    )


def test_single_slot_prefers_latest_year() -> None:
    selected = _select_representative_years(
        years=(
            2022,
            2023,
            2024,
        ),
        limit=1,
    )

    assert selected == (2024,)


def test_diversification_keeps_best_result_per_target_year() -> None:
    candidates = [
        _result(
            year=2023,
            rank=1,
            score=0.900,
        ),
        _result(
            year=2023,
            rank=2,
            score=0.850,
        ),
        _result(
            year=2026,
            rank=3,
            score=0.800,
        ),
        _result(
            year=2019,
            rank=4,
            score=0.750,
        ),
        _result(
            year=2021,
            rank=5,
            score=0.700,
        ),
        _result(
            year=2024,
            rank=6,
            score=0.650,
        ),
    ]

    selected = _diversify_results_by_year(
        results=candidates,
        target_years=(
            2019,
            2021,
            2023,
            2024,
            2026,
        ),
        limit=5,
    )

    assert [item.fiscal_year for item in selected] == [
        2019,
        2021,
        2023,
        2024,
        2026,
    ]

    assert [item.rank for item in selected] == [
        1,
        2,
        3,
        4,
        5,
    ]

    chosen_2023 = next(item for item in selected if item.fiscal_year == 2023)

    assert chosen_2023.score == 0.900


def test_diversification_falls_back_to_other_distinct_years() -> None:
    candidates = [
        _result(
            year=2022,
            rank=1,
            score=0.900,
        ),
        _result(
            year=2023,
            rank=2,
            score=0.800,
        ),
        _result(
            year=2024,
            rank=3,
            score=0.700,
        ),
    ]

    selected = _diversify_results_by_year(
        results=candidates,
        target_years=(
            2019,
            2022,
            2026,
        ),
        limit=3,
    )

    assert len(selected) == 3

    assert {item.fiscal_year for item in selected} == {
        2022,
        2023,
        2024,
    }
