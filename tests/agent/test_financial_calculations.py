"""Regression tests for deterministic financial arithmetic."""

from decimal import Decimal

import pytest

from fingraph.agent.calculations import (
    calculate_change,
    calculate_margin,
)


def test_zero_to_zero_change_is_defined_without_percentage() -> None:
    result = calculate_change(
        metric="revenue",
        start_year=2024,
        end_year=2025,
        start_value=0,
        end_value=0,
    )

    assert result.absolute_change == Decimal("0")
    assert result.percentage_change is None


def test_margin_with_zero_denominator_is_undefined_not_exception() -> None:
    result = calculate_margin(
        metric="operating_margin",
        year=2025,
        numerator_metric="operating_income",
        numerator_value=0,
        denominator_metric="revenue",
        denominator_value=0,
    )

    assert result.margin_percentage is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_value", float("nan")),
        ("start_value", float("inf")),
        ("end_value", float("-inf")),
    ],
)
def test_change_rejects_non_finite_operands(
    field: str,
    value: float,
) -> None:
    kwargs: dict[str, object] = {
        "metric": "revenue",
        "start_year": 2024,
        "end_year": 2025,
        "start_value": 100,
        "end_value": 110,
    }
    kwargs[field] = value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        calculate_change(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("numerator_value", float("nan")),
        ("denominator_value", float("inf")),
    ],
)
def test_margin_rejects_non_finite_operands(
    field: str,
    value: float,
) -> None:
    kwargs: dict[str, object] = {
        "metric": "operating_margin",
        "year": 2025,
        "numerator_metric": "operating_income",
        "numerator_value": 40,
        "denominator_metric": "revenue",
        "denominator_value": 200,
    }
    kwargs[field] = value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        calculate_margin(**kwargs)  # type: ignore[arg-type]
