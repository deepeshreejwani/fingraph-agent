"""Regression coverage for financial graph result identity and completeness."""

import pytest

from fingraph.graph.qa.validator import (
    CypherValidationError,
    requested_economic_years,
    validate_financial_projection_contract,
    validate_result_rows,
)

QUESTION = "Between Alphabet and Meta, which company had higher revenue in 2025?"


def _revenue_row(
    *,
    ticker: str,
    year: int,
    value: float,
    unit: str = "USD",
    start_date: str | None = None,
    end_date: str | None = None,
    fact_id: str | None = None,
    accession_number: str | None = None,
) -> dict[str, object]:
    """Build one canonical annual revenue result row."""

    return {
        "ticker": ticker,
        "year": year,
        "metric_concept": "RevenueFromContractWithCustomerExcludingAssessedTax",
        "value": value,
        "unit": unit,
        "fact_id": fact_id or f"{ticker.lower()}-revenue-{year}",
        "start_date": start_date or f"{year - 1}-07-01",
        "end_date": end_date or f"{year}-06-30",
        "accession_number": accession_number or f"{ticker.lower()}-{year}-10k",
    }


def test_multi_company_financial_rows_require_economic_year() -> None:
    row = _revenue_row(
        ticker="GOOGL",
        year=2025,
        value=402836000000.0,
    )
    row.pop("year")

    with pytest.raises(
        CypherValidationError,
        match="economic_year",
    ):
        validate_result_rows(
            question=QUESTION,
            rows=[row],
        )


def test_multi_company_financial_rows_with_complete_identity_pass() -> None:
    rows = [
        _revenue_row(
            ticker="GOOGL",
            year=2025,
            value=402836000000.0,
        ),
        _revenue_row(
            ticker="META",
            year=2025,
            value=200966000000.0,
        ),
    ]

    validate_result_rows(
        question=QUESTION,
        rows=rows,
    )


def test_multi_company_financial_rows_still_require_all_tickers() -> None:
    rows = [
        _revenue_row(
            ticker="GOOGL",
            year=2025,
            value=402836000000.0,
        )
    ]

    with pytest.raises(
        CypherValidationError,
        match="missing requested ticker",
    ):
        validate_result_rows(
            question="Between Apple and Alphabet, which company had lower revenue in 2025?",
            rows=rows,
        )


def test_single_company_question_rejects_wrong_company_rows() -> None:
    rows = [
        _revenue_row(
            ticker="AAPL",
            year=2025,
            value=416161000000.0,
        )
    ]

    with pytest.raises(
        CypherValidationError,
        match="unrequested ticker",
    ):
        validate_result_rows(
            question="What was Microsoft revenue in 2025?",
            rows=rows,
        )


def test_annual_revenue_question_rejects_quarterly_duration_fact() -> None:
    rows = [
        _revenue_row(
            ticker="MSFT",
            year=2025,
            value=70000000000.0,
            start_date="2025-01-01",
            end_date="2025-03-31",
        )
    ]

    with pytest.raises(
        CypherValidationError,
        match="not an annual-duration observation",
    ):
        validate_result_rows(
            question="What was Microsoft revenue in 2025?",
            rows=rows,
        )


def test_financial_rows_reject_mixed_units() -> None:
    rows = [
        _revenue_row(
            ticker="GOOGL",
            year=2025,
            value=402836000000.0,
            unit="USD",
        ),
        _revenue_row(
            ticker="META",
            year=2025,
            value=200966000000.0,
            unit="EUR",
        ),
    ]

    with pytest.raises(
        CypherValidationError,
        match="mix incompatible units",
    ):
        validate_result_rows(
            question=QUESTION,
            rows=rows,
        )


def test_financial_rows_require_every_requested_year() -> None:
    rows = [
        _revenue_row(
            ticker="MSFT",
            year=2025,
            value=281724000000.0,
        )
    ]

    with pytest.raises(
        CypherValidationError,
        match="missing requested economic year",
    ):
        validate_result_rows(
            question="How did Microsoft revenue change from 2024 to 2025?",
            rows=rows,
        )


def test_financial_projection_contract_checks_final_return_only() -> None:
    cypher = """
MATCH (company:Company {ticker: $ticker})
MATCH (company)-[:REPORTS]->(fact:FinancialFact)-[:FOR_PERIOD]->(period:FiscalPeriod)
MATCH (filing:Filing)-[:REPORTED_FACT]->(fact)
WITH company, period, fact, filing, fact.value AS value
RETURN company.ticker AS ticker,
       period.period_year AS year,
       fact.concept AS metric_concept,
       999 AS value,
       fact.unit AS unit,
       fact.fact_id AS fact_id,
       fact.start_date AS start_date,
       fact.end_date AS end_date,
       filing.accession_number AS accession_number
""".strip()

    with pytest.raises(
        CypherValidationError,
        match="final RETURN clause",
    ):
        validate_financial_projection_contract(cypher)


def test_financial_rows_require_each_company_for_each_requested_year() -> None:
    rows = [
        _revenue_row(
            ticker="AAPL",
            year=2024,
            value=383285000000.0,
        ),
        _revenue_row(
            ticker="MSFT",
            year=2025,
            value=281724000000.0,
        ),
    ]

    with pytest.raises(
        CypherValidationError,
        match="missing requested company/economic-year coverage",
    ):
        validate_result_rows(
            question=("Compare Apple and Microsoft revenue between 2024 and 2025."),
            rows=rows,
        )


def test_financial_rows_reject_economic_year_end_date_mismatch() -> None:
    rows = [
        _revenue_row(
            ticker="MSFT",
            year=2025,
            value=281724000000.0,
            start_date="2023-07-01",
            end_date="2024-06-30",
        )
    ]

    with pytest.raises(
        CypherValidationError,
        match="does not match end_date year",
    ):
        validate_result_rows(
            question="What was Microsoft revenue in 2025?",
            rows=rows,
        )


def test_financial_rows_reject_unrequested_economic_years() -> None:
    rows = [
        _revenue_row(
            ticker="MSFT",
            year=2025,
            value=281724000000.0,
        ),
        _revenue_row(
            ticker="MSFT",
            year=2024,
            value=245122000000.0,
        ),
    ]

    with pytest.raises(
        CypherValidationError,
        match="unrequested economic year",
    ):
        validate_result_rows(
            question="What was Microsoft revenue in 2025?",
            rows=rows,
        )


def test_financial_rows_complete_company_year_grid_passes() -> None:
    rows = [
        _revenue_row(
            ticker="AAPL",
            year=2024,
            value=383285000000.0,
        ),
        _revenue_row(
            ticker="AAPL",
            year=2025,
            value=391035000000.0,
        ),
        _revenue_row(
            ticker="MSFT",
            year=2024,
            value=245122000000.0,
        ),
        _revenue_row(
            ticker="MSFT",
            year=2025,
            value=281724000000.0,
        ),
    ]

    validate_result_rows(
        question="Compare Apple and Microsoft revenue between 2024 and 2025.",
        rows=rows,
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "Microsoft revenue from 2024 to 2026",
            (2024, 2025, 2026),
        ),
        (
            "Microsoft revenue between 2024 and 2026",
            (2024, 2025, 2026),
        ),
        (
            "Microsoft revenue 2024 through 2026",
            (2024, 2025, 2026),
        ),
        (
            "Microsoft revenue 2024-2026",
            (2024, 2025, 2026),
        ),
        (
            "Microsoft revenue 2024–2026",
            (2024, 2025, 2026),
        ),
        (
            "Microsoft revenue 2024—2026",
            (2024, 2025, 2026),
        ),
    ],
)
def test_requested_economic_years_expands_inclusive_ranges(
    question: str,
    expected: tuple[int, ...],
) -> None:
    assert requested_economic_years(question) == expected


def test_requested_economic_years_does_not_invent_range_for_explicit_year_list() -> None:
    assert requested_economic_years("Compare Microsoft revenue in 2024 and 2026.") == (
        2024,
        2026,
    )


def test_requested_economic_years_preserves_single_explicit_year() -> None:
    assert requested_economic_years("What was Microsoft revenue in 2025?") == (2025,)


def test_three_year_financial_range_accepts_intermediate_year() -> None:
    rows = [
        _revenue_row(
            ticker="MSFT",
            year=2024,
            value=245122000000.0,
        ),
        _revenue_row(
            ticker="MSFT",
            year=2025,
            value=281724000000.0,
        ),
        _revenue_row(
            ticker="MSFT",
            year=2026,
            value=331839000000.0,
        ),
    ]

    validate_result_rows(
        question="What was Microsoft revenue from 2024 to 2026?",
        rows=rows,
    )


def test_three_year_financial_range_requires_intermediate_year() -> None:
    rows = [
        _revenue_row(
            ticker="MSFT",
            year=2024,
            value=245122000000.0,
        ),
        _revenue_row(
            ticker="MSFT",
            year=2026,
            value=331839000000.0,
        ),
    ]

    with pytest.raises(
        CypherValidationError,
        match="missing requested economic year",
    ):
        validate_result_rows(
            question="What was Microsoft revenue from 2024 to 2026?",
            rows=rows,
        )


def test_multi_company_three_year_range_requires_complete_company_year_grid() -> None:
    rows = [
        _revenue_row(
            ticker="AAPL",
            year=2024,
            value=383285000000.0,
        ),
        _revenue_row(
            ticker="AAPL",
            year=2025,
            value=391035000000.0,
        ),
        _revenue_row(
            ticker="AAPL",
            year=2026,
            value=416161000000.0,
        ),
        _revenue_row(
            ticker="MSFT",
            year=2024,
            value=245122000000.0,
        ),
        _revenue_row(
            ticker="MSFT",
            year=2025,
            value=281724000000.0,
        ),
    ]

    with pytest.raises(
        CypherValidationError,
        match="missing requested company/economic-year coverage",
    ):
        validate_result_rows(
            question="Compare Apple and Microsoft revenue from 2024 to 2026.",
            rows=rows,
        )


def test_multi_company_three_year_range_complete_grid_passes() -> None:
    rows = [
        _revenue_row(
            ticker="AAPL",
            year=2024,
            value=383285000000.0,
        ),
        _revenue_row(
            ticker="AAPL",
            year=2025,
            value=391035000000.0,
        ),
        _revenue_row(
            ticker="AAPL",
            year=2026,
            value=416161000000.0,
        ),
        _revenue_row(
            ticker="MSFT",
            year=2024,
            value=245122000000.0,
        ),
        _revenue_row(
            ticker="MSFT",
            year=2025,
            value=281724000000.0,
        ),
        _revenue_row(
            ticker="MSFT",
            year=2026,
            value=331839000000.0,
        ),
    ]

    validate_result_rows(
        question="Compare Apple and Microsoft revenue from 2024 to 2026.",
        rows=rows,
    )
