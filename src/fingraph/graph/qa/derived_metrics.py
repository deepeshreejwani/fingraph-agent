"""Validation for supported derived financial graph metrics."""

import re
from typing import Any

from fingraph.graph.qa.schema import (
    CANONICAL_FINANCIAL_METRICS,
    DERIVED_FINANCIAL_METRICS,
)
from fingraph.graph.qa.validator import (
    CypherValidationError,
    requested_company_tickers,
    requested_economic_years,
    validate_financial_projection_contract,
    validate_financial_row_identity,
)

_DERIVED_PATTERNS: tuple[
    tuple[
        str,
        re.Pattern[str],
    ],
    ...,
] = (
    (
        "operating_margin",
        re.compile(
            r"\boperating\s+margin\b",
            re.IGNORECASE,
        ),
    ),
    (
        "net_margin",
        re.compile(
            r"\bnet\s+margin\b",
            re.IGNORECASE,
        ),
    ),
)


def requested_derived_financial_metric(
    question: str,
) -> str | None:
    """Return a supported derived financial metric requested by the user."""

    for (
        metric,
        pattern,
    ) in _DERIVED_PATTERNS:
        if pattern.search(question):
            return metric

    return None


def _required_concepts(
    metric: str,
) -> dict[
    str,
    set[str],
]:
    return {
        raw_metric: set(CANONICAL_FINANCIAL_METRICS[raw_metric])
        for raw_metric in DERIVED_FINANCIAL_METRICS[metric]
    }


def _concepts_used(
    *,
    cypher: str,
    parameters: dict[
        str,
        Any,
    ],
) -> set[str]:
    all_concepts = {
        concept for concepts in CANONICAL_FINANCIAL_METRICS.values() for concept in concepts
    }

    used = {
        concept
        for concept in all_concepts
        if re.search(
            rf"(?<![A-Za-z0-9_])"
            rf"{re.escape(concept)}"
            rf"(?![A-Za-z0-9_])",
            cypher,
        )
    }

    for value in parameters.values():
        candidates = (
            value
            if isinstance(
                value,
                list,
            )
            else [value]
        )

        for candidate in candidates:
            if (
                isinstance(
                    candidate,
                    str,
                )
                and candidate in all_concepts
            ):
                used.add(candidate)

    return used


def validate_derived_metric_cypher(
    *,
    question: str,
    cypher: str,
    parameters: dict[
        str,
        Any,
    ],
) -> None:
    """Validate raw-metric coverage before a derived query reaches Neo4j."""

    metric = requested_derived_financial_metric(question)

    if metric is None:
        return

    required = _required_concepts(metric)

    allowed = set().union(*required.values())

    used = _concepts_used(
        cypher=cypher,
        parameters=parameters,
    )

    missing = [
        raw_metric
        for (
            raw_metric,
            concepts,
        ) in required.items()
        if not used.intersection(concepts)
    ]

    if missing:
        raise CypherValidationError(
            f"Generated Cypher for derived metric '{metric}' "
            "is missing required raw metric(s): "
            + ", ".join(missing)
            + ". Retrieve all required raw metrics without "
            "calculating the derived percentage in Cypher."
        )

    unexpected = used - allowed

    if unexpected:
        raise CypherValidationError(
            f"Generated Cypher for derived metric '{metric}' "
            "uses unexpected canonical concept(s): " + ", ".join(sorted(unexpected)) + "."
        )

    if not re.search(
        r"\b(?:latest\.fact|fact)\.concept"
        r"\s+AS\s+metric_concept\b",
        cypher,
        flags=re.IGNORECASE,
    ):
        raise CypherValidationError(
            "Derived financial metric queries must preserve "
            "raw metric identity with fact.concept AS metric_concept "
            "or latest.fact.concept AS metric_concept."
        )

    validate_financial_projection_contract(cypher)


def validate_derived_metric_rows(
    *,
    question: str,
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
) -> None:
    """Reject incomplete, cross-company, mixed-unit or non-annual margin evidence."""

    metric = requested_derived_financial_metric(question)

    if metric is None or not rows:
        return

    required = _required_concepts(metric)

    allowed = set().union(*required.values())

    requested_tickers = set(requested_company_tickers(question))

    requested_years = set(requested_economic_years(question))

    for (
        index,
        row,
    ) in enumerate(rows):
        validate_financial_row_identity(
            row=row,
            row_index=index,
            allowed_concepts=allowed,
            require_annual_duration=True,
        )

    actual_tickers = {str(row["ticker"]).upper() for row in rows}

    if requested_tickers:
        unexpected = actual_tickers - requested_tickers

        if unexpected:
            raise CypherValidationError(
                "Derived financial results contain unrequested "
                "ticker(s): " + ", ".join(sorted(unexpected)) + "."
            )

        missing_tickers = requested_tickers - actual_tickers

        if missing_tickers:
            raise CypherValidationError(
                "Derived financial results are missing requested "
                "ticker(s): " + ", ".join(sorted(missing_tickers)) + "."
            )

    actual_years: set[int] = set()

    for row in rows:
        raw_year = row.get("year")

        if raw_year is None:
            raw_year = row.get("period_year")

        if raw_year is None:
            raw_year = row.get("economic_year")

        if raw_year is None:
            raise CypherValidationError(
                "Derived financial metric results must RETURN "
                "the economic year for every raw metric row."
            )

        actual_years.add(int(str(raw_year)))

    missing_years = requested_years - actual_years

    if missing_years:
        raise CypherValidationError(
            "Derived financial results are missing requested "
            "economic year(s): " + ", ".join(str(year) for year in sorted(missing_years)) + "."
        )

    grouped: dict[
        tuple[
            str,
            int,
            str,
        ],
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = {}

    for row in rows:
        raw_year = row.get("year")

        if raw_year is None:
            raw_year = row.get("period_year")

        if raw_year is None:
            raw_year = row.get("economic_year")

        if raw_year is None:
            raise CypherValidationError(
                "Derived financial metric results must RETURN "
                "the economic year for every raw metric row."
            )

        key = (
            str(row["ticker"]).upper(),
            int(str(raw_year)),
            str(row["unit"]),
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    for (
        key,
        group_rows,
    ) in grouped.items():
        concepts = {str(row["metric_concept"]) for row in group_rows}

        unexpected = concepts - allowed

        if unexpected:
            raise CypherValidationError(
                "Derived financial metric results contain "
                "unexpected canonical concept(s): " + ", ".join(sorted(unexpected)) + "."
            )

        missing_metrics = [
            raw_metric
            for (
                raw_metric,
                concepts_for_metric,
            ) in required.items()
            if not concepts.intersection(concepts_for_metric)
        ]

        if missing_metrics:
            (
                ticker,
                year,
                unit,
            ) = key

            raise CypherValidationError(
                "Derived financial metric results are missing "
                "required raw metric(s) for "
                f"{ticker} {year} {unit}: "
                + ", ".join(missing_metrics)
                + ". Select the latest filing observation "
                "separately per fact.concept and return one row "
                "per required raw metric."
            )

        metric_periods: dict[
            str,
            tuple[
                str,
                str,
            ],
        ] = {}

        for (
            raw_metric,
            concepts_for_metric,
        ) in required.items():
            metric_rows = [
                row for row in group_rows if str(row["metric_concept"]) in concepts_for_metric
            ]

            if len(metric_rows) != 1:
                (
                    ticker,
                    year,
                    unit,
                ) = key

                raise CypherValidationError(
                    "Derived financial metric results contain "
                    "ambiguous raw-fact identity for "
                    f"{raw_metric} in {ticker} {year} {unit}. "
                    "Return exactly one canonical raw fact per required metric."
                )

            metric_row = metric_rows[0]

            metric_periods[raw_metric] = (
                str(metric_row["start_date"]),
                str(metric_row["end_date"]),
            )

        if len(set(metric_periods.values())) != 1:
            (
                ticker,
                year,
                unit,
            ) = key

            raise CypherValidationError(
                "Derived financial metric results mix economic periods for "
                f"{ticker} {year} {unit}. Required raw metrics must share "
                "the same start_date and end_date."
            )
