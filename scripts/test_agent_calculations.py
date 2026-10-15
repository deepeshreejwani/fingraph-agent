"""Smoke test deterministic FinGraph calculations."""

from fingraph.agent import (
    calculate_cagr,
    calculate_margin,
    calculate_series_change,
)


def main() -> None:
    rows: list[dict[str, object]] = [
        {
            "year": 2024,
            "revenue": 245_122_000_000,
        },
        {
            "year": 2025,
            "revenue": 281_724_000_000,
        },
        {
            "year": 2026,
            "revenue": 331_839_000_000,
        },
    ]

    change = calculate_series_change(
        metric="revenue",
        rows=rows,
        start_year=2024,
        end_year=2026,
        year_key="year",
        value_key="revenue",
    )

    cagr = calculate_cagr(
        metric="revenue",
        start_year=2024,
        end_year=2026,
        start_value=245_122_000_000,
        end_value=331_839_000_000,
    )

    margin = calculate_margin(
        metric="operating_margin",
        year=2025,
        numerator_metric="operating_income",
        numerator_value=109_433_000_000,
        denominator_metric="revenue",
        denominator_value=281_724_000_000,
    )

    print(
        "Start:",
        f"{change.start_value:.3f}",
    )

    print(
        "End:",
        f"{change.end_value:.3f}",
    )

    print(
        "Absolute change:",
        f"{change.absolute_change:.3f}",
    )

    if change.percentage_change is not None:
        print(
            "Percentage change:",
            f"{change.percentage_change:.3f}%",
        )

    print(
        "CAGR:",
        f"{cagr.cagr_percentage:.3f}%",
    )

    if margin.margin_percentage is not None:
        print(
            "Operating margin:",
            f"{margin.margin_percentage:.3f}%",
        )


if __name__ == "__main__":
    main()
