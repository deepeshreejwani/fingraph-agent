"""Regression tests for unsupported-company and unsupported-metric evidence guards."""

from fingraph.agent.tools import (
    _has_explicit_unsupported_company,
    _has_explicit_unsupported_financial_metric,
)


def test_supported_companies_are_not_rejected() -> None:
    assert not _has_explicit_unsupported_company("Describe Apple's business in its 2025 10-K.")

    assert not _has_explicit_unsupported_company(
        "What market risks did Alphabet describe in its 2025 10-K?"
    )

    assert not _has_explicit_unsupported_company("Describe AMD's business in its 2025 10-K.")

    assert not _has_explicit_unsupported_company(
        "What risks did Salesforce describe in its 2025 10-K?"
    )

    assert not _has_explicit_unsupported_company(
        "What market risks did Oracle describe in its 2025 10-K?"
    )

    assert not _has_explicit_unsupported_company(
        "Compare total assets for Alphabet and Meta in 2025."
    )

    assert not _has_explicit_unsupported_company(
        "Compare operating income for Alphabet and Meta in 2025 "
        "and discuss what management said about performance."
    )


def test_unsupported_named_companies_are_rejected() -> None:
    assert _has_explicit_unsupported_company("Describe JPMorgan's business in its 2025 10-K.")

    assert _has_explicit_unsupported_company("What risks did Uber describe in its 2025 10-K?")

    assert _has_explicit_unsupported_company(
        "What market risks did Boeing describe in its 2025 10-K?"
    )


def test_unsupported_financial_metrics_are_rejected() -> None:
    assert _has_explicit_unsupported_financial_metric("What was Apple EBIT in 2025?")

    assert _has_explicit_unsupported_financial_metric("What was Microsoft EBITDA in 2026?")

    assert _has_explicit_unsupported_financial_metric("What was Alphabet gross profit in 2025?")

    assert _has_explicit_unsupported_financial_metric("What was Meta EPS in 2025?")

    assert _has_explicit_unsupported_financial_metric(
        "What was Amazon operating cash flow in 2025?"
    )


def test_supported_financial_metrics_are_not_rejected() -> None:
    assert not _has_explicit_unsupported_financial_metric("What was Apple revenue in 2025?")

    assert not _has_explicit_unsupported_financial_metric("What was Meta operating income in 2025?")
