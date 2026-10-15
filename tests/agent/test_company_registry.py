"""Regression coverage for the centralized supported-company universe."""

from fingraph.agent.tools import (
    _extract_tickers,
    _has_explicit_unsupported_company,
)
from fingraph.company_registry import (
    COMPANIES_BY_TICKER,
    extract_company_tickers,
)
from fingraph.graph.qa.schema import (
    CANONICAL_COMPANY_TICKERS,
    GRAPH_SCHEMA,
)

EXPECTED_TICKERS = {
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "AMD",
    "INTC",
    "AVGO",
    "QCOM",
    "CSCO",
    "IBM",
    "ORCL",
    "CRM",
    "ADBE",
    "NOW",
    "PANW",
    "NFLX",
    "TSLA",
    "TXN",
    "MU",
    "AMAT",
    "LRCX",
    "KLAC",
    "PLTR",
}


def test_serving_registry_matches_25_company_corpus() -> None:
    assert set(COMPANIES_BY_TICKER) == EXPECTED_TICKERS
    assert set(CANONICAL_COMPANY_TICKERS) == EXPECTED_TICKERS


def test_scaled_company_names_and_tickers_are_recognized() -> None:
    assert extract_company_tickers("Describe AMD's business in 2025.") == ("AMD",)

    assert _extract_tickers("What were risks for TSLA in 2025?") == ("TSLA",)

    assert extract_company_tickers("Compare ServiceNow and Palantir.") == (
        "NOW",
        "PLTR",
    )


def test_company_matching_is_boundary_aware() -> None:
    assert _extract_tickers("Describe the metaphysical risks to Microsoft.") == ("MSFT",)


def test_supported_scaled_company_is_not_rejected() -> None:
    assert not _has_explicit_unsupported_company("What were risks for TSLA in 2025?")

    assert not _has_explicit_unsupported_company("Describe AMD's business in 2025.")


def test_explicit_unsupported_ticker_is_rejected() -> None:
    assert _has_explicit_unsupported_company("What were risks for JPM in 2025?")


def test_graph_schema_exposes_scaled_company_universe() -> None:
    assert "Advanced Micro Devices -> AMD" in GRAPH_SCHEMA

    assert "ServiceNow -> NOW" in GRAPH_SCHEMA

    assert "Palantir -> PLTR" in GRAPH_SCHEMA
