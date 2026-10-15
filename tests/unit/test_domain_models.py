"""Tests for canonical domain models."""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from fingraph.domain.models import Company, FinancialFact


def test_company_is_immutable() -> None:
    company = Company(
        cik="0000320193",
        ticker="AAPL",
        name="Apple Inc.",
        industry="Technology",
        sic="3571",
    )

    with pytest.raises(ValidationError):
        company.name = "Changed"  # type: ignore[misc]


def test_financial_fact_preserves_decimal_value() -> None:
    fact = FinancialFact(
        fact_id="fact-1",
        cik="0000320193",
        taxonomy="us-gaap",
        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
        value=Decimal("391035000000"),
        unit="USD",
        start_date=date(2023, 10, 1),
        end_date=date(2024, 9, 28),
        fiscal_year=2024,
        fiscal_period="FY",
        accession_number="0000320193-24-000123",
    )

    assert fact.value == Decimal("391035000000")


def test_company_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Company(
            cik="0000320193",
            ticker="AAPL",
            name="Apple Inc.",
            unexpected="not allowed",  # type: ignore[call-arg]
        )
