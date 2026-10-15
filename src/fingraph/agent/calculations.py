"""Deterministic financial calculations."""

from decimal import Decimal, InvalidOperation

from pydantic import (
    BaseModel,
    ConfigDict,
)


class FinancialChange(BaseModel):
    """Deterministically calculated change between two periods."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    metric: str

    start_year: int
    end_year: int

    start_value: Decimal
    end_value: Decimal

    absolute_change: Decimal

    percentage_change: Decimal | None


class FinancialCAGR(BaseModel):
    """Deterministically calculated compound annual growth rate."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    metric: str

    start_year: int
    end_year: int
    periods: int

    start_value: Decimal
    end_value: Decimal

    cagr_percentage: Decimal


class FinancialMargin(BaseModel):
    """Deterministically calculated financial margin."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    metric: str
    year: int

    numerator_metric: str
    numerator_value: Decimal

    denominator_metric: str
    denominator_value: Decimal

    margin_percentage: Decimal | None


def _finite_decimal(
    value: int | float | Decimal,
    *,
    field_name: str,
) -> Decimal:
    """Convert one numeric operand to a finite Decimal."""

    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid finite number.") from exc

    if not result.is_finite():
        raise ValueError(f"{field_name} must be a finite number.")

    return result


def calculate_change(
    *,
    metric: str,
    start_year: int,
    end_year: int,
    start_value: int | float | Decimal,
    end_value: int | float | Decimal,
) -> FinancialChange:
    """Calculate absolute and percentage change deterministically."""

    start = _finite_decimal(
        start_value,
        field_name="start_value",
    )

    end = _finite_decimal(
        end_value,
        field_name="end_value",
    )

    absolute = end - start

    percentage: Decimal | None
    percentage = None if start == 0 else (absolute / start) * Decimal("100")

    return FinancialChange(
        metric=metric,
        start_year=start_year,
        end_year=end_year,
        start_value=start,
        end_value=end,
        absolute_change=absolute,
        percentage_change=percentage,
    )


def calculate_cagr(
    *,
    metric: str,
    start_year: int,
    end_year: int,
    start_value: int | float | Decimal,
    end_value: int | float | Decimal,
) -> FinancialCAGR:
    """Calculate CAGR deterministically from two positive-period observations."""

    if end_year <= start_year:
        raise ValueError("end_year must be greater than start_year for CAGR.")

    start = _finite_decimal(
        start_value,
        field_name="start_value",
    )

    end = _finite_decimal(
        end_value,
        field_name="end_value",
    )

    if start <= 0:
        raise ValueError("CAGR requires a positive start value.")

    if end < 0:
        raise ValueError("CAGR does not support a negative end value.")

    periods = end_year - start_year

    if end == 0:
        cagr_percentage = Decimal("-100")

    else:
        ratio = end / start

        exponent = Decimal(1) / Decimal(periods)

        cagr_percentage = ((ratio.ln() * exponent).exp() - Decimal(1)) * Decimal("100")

    if not cagr_percentage.is_finite():
        raise ValueError("CAGR calculation produced a non-finite result.")

    return FinancialCAGR(
        metric=metric,
        start_year=start_year,
        end_year=end_year,
        periods=periods,
        start_value=start,
        end_value=end,
        cagr_percentage=cagr_percentage,
    )


def calculate_margin(
    *,
    metric: str,
    year: int,
    numerator_metric: str,
    numerator_value: int | float | Decimal,
    denominator_metric: str,
    denominator_value: int | float | Decimal,
) -> FinancialMargin:
    """Calculate a percentage margin deterministically."""

    numerator = _finite_decimal(
        numerator_value,
        field_name="numerator_value",
    )

    denominator = _finite_decimal(
        denominator_value,
        field_name="denominator_value",
    )

    margin_percentage: Decimal | None
    margin_percentage = None if denominator == 0 else (numerator / denominator) * Decimal("100")

    return FinancialMargin(
        metric=metric,
        year=year,
        numerator_metric=numerator_metric,
        numerator_value=numerator,
        denominator_metric=denominator_metric,
        denominator_value=denominator,
        margin_percentage=margin_percentage,
    )


def calculate_series_change(
    *,
    metric: str,
    rows: list[dict[str, object]],
    start_year: int,
    end_year: int,
    year_key: str,
    value_key: str,
) -> FinancialChange:
    """Calculate change from an unambiguous structured result series."""

    by_year: dict[
        int,
        Decimal,
    ] = {}

    for row in rows:
        raw_year = row.get(year_key)

        raw_value = row.get(value_key)

        if raw_year is None or raw_value is None:
            continue

        try:
            year = int(str(raw_year))

        except ValueError as exc:
            raise ValueError(f"Invalid year value: {raw_year}") from exc

        if isinstance(
            raw_value,
            bool,
        ) or not isinstance(
            raw_value,
            (
                int,
                float,
                Decimal,
            ),
        ):
            raise ValueError(f"{value_key}[{year}] must be numeric.")

        value = _finite_decimal(
            raw_value,
            field_name=(f"{value_key}[{year}]"),
        )

        existing = by_year.get(year)

        if existing is not None and existing != value:
            raise ValueError(f"Ambiguous values for year {year}: {existing} and {value}.")

        by_year[year] = value

    if start_year not in by_year:
        raise ValueError(f"Missing start-year value: {start_year}")

    if end_year not in by_year:
        raise ValueError(f"Missing end-year value: {end_year}")

    return calculate_change(
        metric=metric,
        start_year=start_year,
        end_year=end_year,
        start_value=(by_year[start_year]),
        end_value=(by_year[end_year]),
    )
