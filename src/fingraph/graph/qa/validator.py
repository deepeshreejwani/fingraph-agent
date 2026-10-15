"""Validation and normalization of generated Cypher."""

import re
from datetime import date
from decimal import (
    Decimal,
    InvalidOperation,
)
from typing import Any

from fingraph.graph.qa.models import (
    CypherPlan,
    ValidatedCypher,
)
from fingraph.graph.qa.schema import (
    ALLOWED_NODE_LABELS,
    ALLOWED_NODE_PROPERTIES,
    ALLOWED_RELATIONSHIP_TYPES,
    CANONICAL_COMPANY_TICKERS,
    CANONICAL_FINANCIAL_METRICS,
)


class CypherValidationError(ValueError):
    """Generated Cypher violated the read-only contract."""


FORBIDDEN_PATTERNS = (
    r"\bCREATE\b",
    r"\bINSERT\b",
    r"\bMERGE\b",
    r"\bDELETE\b",
    r"\bDETACH\b",
    r"\bSET\b",
    r"\bREMOVE\b",
    r"\bDROP\b",
    r"\bALTER\b",
    r"\bLOAD\s+CSV\b",
    r"\bFOREACH\b",
    r"\bCALL\b",
    r"\bSHOW\b",
    r"\bTERMINATE\b",
    r"\bUSE\b",
    r"\bGRANT\b",
    r"\bDENY\b",
    r"\bREVOKE\b",
    r"\bSTART\s+DATABASE\b",
    r"\bSTOP\s+DATABASE\b",
    r"\bUNION\b",
)

COMMENT_PATTERN = re.compile(
    r"//.*?$|/\*.*?\*/",
    flags=(re.MULTILINE | re.DOTALL),
)

NODE_LABEL_PATTERN = re.compile(r"\([^)]*?:`?([A-Za-z_][A-Za-z0-9_]*)`?")

RELATIONSHIP_PATTERN = re.compile(r"\[[^\]]*?:`?([A-Za-z_][A-Za-z0-9_]*)`?")

NODE_PATTERN_CONTENT = re.compile(r"\(([^()]*)\)")

RELATIONSHIP_PATTERN_CONTENT = re.compile(r"\[([^\[\]]*)\]")

SCHEMA_TOKEN_PATTERN = re.compile(r":\s*`?[A-Za-z_][A-Za-z0-9_]*`?")

SCHEMA_EXPRESSION_OPERATOR_PATTERN = re.compile(r"[|&!%]")

MATCH_CLAUSE_PATTERN = re.compile(
    r"\b(?:OPTIONAL\s+)?MATCH\b",
    flags=re.IGNORECASE,
)

MATCH_CLAUSE_TERMINATOR_PATTERN = re.compile(
    r"\b(?:"
    r"OPTIONAL\s+MATCH|"
    r"MATCH|"
    r"WHERE|"
    r"WITH|"
    r"RETURN|"
    r"UNWIND|"
    r"ORDER\s+BY|"
    r"SKIP|"
    r"LIMIT"
    r")\b",
    flags=re.IGNORECASE,
)

DYNAMIC_SCHEMA_TOKEN_PATTERN = re.compile(
    r":\s*\$",
    flags=re.IGNORECASE,
)

IDENTIFIER_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"

NODE_BINDING_PATTERN = re.compile(
    rf"^\s*(?:(?P<variable>{IDENTIFIER_PATTERN})\s*)?:\s*"
    rf"(?:`(?P<label_quoted>{IDENTIFIER_PATTERN})`|(?P<label_plain>{IDENTIFIER_PATTERN}))"
)

RELATIONSHIP_BINDING_PATTERN = re.compile(
    rf"^\s*(?:(?P<variable>{IDENTIFIER_PATTERN})\s*)?:\s*"
    rf"(?:`(?P<type_quoted>{IDENTIFIER_PATTERN})`|(?P<type_plain>{IDENTIFIER_PATTERN}))"
)

PATTERN_MAP_PROPERTY_PATTERN = re.compile(
    rf"(?:`(?P<quoted>{IDENTIFIER_PATTERN})`|(?P<plain>{IDENTIFIER_PATTERN}))\s*:"
)

FINAL_RETURN_COLLECTION_AGGREGATE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])collect\s*\(",
    flags=re.IGNORECASE,
)


_RESULT_ECONOMIC_YEAR_KEYS = {
    "year",
    "period_year",
    "economic_year",
}


_FLOW_FINANCIAL_METRICS = {
    "revenue",
    "net_income",
    "operating_income",
}


_ECONOMIC_YEAR_TOKEN = r"(?:19|20)\d{2}"

_ECONOMIC_YEAR_RANGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"\b(?:from\s+)?(?P<start>{_ECONOMIC_YEAR_TOKEN})\s+"
        rf"(?:to|through)\s+(?P<end>{_ECONOMIC_YEAR_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bbetween\s+(?P<start>{_ECONOMIC_YEAR_TOKEN})\s+"
        rf"and\s+(?P<end>{_ECONOMIC_YEAR_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<start>{_ECONOMIC_YEAR_TOKEN})\s*[\-–—]\s*"
        rf"(?P<end>{_ECONOMIC_YEAR_TOKEN})\b",
        re.IGNORECASE,
    ),
)


_METRIC_PATTERNS: tuple[
    tuple[
        str,
        re.Pattern[str],
    ],
    ...,
] = (
    (
        "total_liabilities",
        re.compile(
            r"\b(?:total\s+)?liabilit(?:y|ies)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "total_assets",
        re.compile(
            r"\b(?:total\s+)?assets?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "operating_income",
        re.compile(
            r"\boperating\s+income\b",
            re.IGNORECASE,
        ),
    ),
    (
        "net_income",
        re.compile(
            r"\bnet\s+income\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cash",
        re.compile(
            r"\bcash(?:\s+and\s+cash\s+equivalents)?"
            r"(?:\s+balance)?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "revenue",
        re.compile(
            r"\brevenues?\b|\bnet\s+sales\b",
            re.IGNORECASE,
        ),
    ),
)


_FINANCIAL_PROJECTION_PATTERNS: tuple[
    tuple[
        str,
        re.Pattern[str],
    ],
    ...,
] = (
    (
        "ticker",
        re.compile(
            r"\.[`]?ticker[`]?\s+AS\s+[`]?ticker[`]?",
            re.IGNORECASE,
        ),
    ),
    (
        "economic_year",
        re.compile(
            r"\.[`]?period_year[`]?\s+AS\s+"
            r"[`]?(?:year|period_year|economic_year)[`]?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "metric_concept",
        re.compile(
            r"\.[`]?concept[`]?\s+AS\s+[`]?metric_concept[`]?",
            re.IGNORECASE,
        ),
    ),
    (
        "value",
        re.compile(
            r"\.[`]?value[`]?\s+AS\s+[`]?value[`]?",
            re.IGNORECASE,
        ),
    ),
    (
        "unit",
        re.compile(
            r"\.[`]?unit[`]?\s+AS\s+[`]?unit[`]?",
            re.IGNORECASE,
        ),
    ),
    (
        "fact_id",
        re.compile(
            r"\.[`]?fact_id[`]?\s+AS\s+[`]?fact_id[`]?",
            re.IGNORECASE,
        ),
    ),
    (
        "start_date",
        re.compile(
            r"\.[`]?start_date[`]?\s+AS\s+[`]?start_date[`]?",
            re.IGNORECASE,
        ),
    ),
    (
        "end_date",
        re.compile(
            r"\.[`]?end_date[`]?\s+AS\s+[`]?end_date[`]?",
            re.IGNORECASE,
        ),
    ),
    (
        "accession_number",
        re.compile(
            r"\.[`]?accession_number[`]?\s+AS\s+"
            r"[`]?accession_number[`]?",
            re.IGNORECASE,
        ),
    ),
)


def requested_financial_metric(
    question: str,
) -> str | None:
    """Return the canonical financial metric explicitly requested by a question."""

    for (
        metric,
        pattern,
    ) in _METRIC_PATTERNS:
        if pattern.search(question):
            return metric

    return None


def requested_company_tickers(
    question: str,
) -> tuple[str, ...]:
    """Return canonical tickers explicitly named by the question."""

    normalized = question.lower()

    found: set[str] = set()

    for (
        ticker,
        aliases,
    ) in CANONICAL_COMPANY_TICKERS.items():
        for alias in aliases:
            if re.search(
                rf"(?<![a-z0-9])"
                rf"{re.escape(alias)}"
                rf"(?![a-z0-9])",
                normalized,
            ):
                found.add(ticker)

                break

    return tuple(sorted(found))


def requested_economic_years(
    question: str,
) -> tuple[int, ...]:
    """Return explicitly requested economic years, expanding inclusive ranges."""

    years = {
        int(match)
        for match in re.findall(
            rf"\b{_ECONOMIC_YEAR_TOKEN}\b",
            question,
        )
    }

    for pattern in _ECONOMIC_YEAR_RANGE_PATTERNS:
        for match in pattern.finditer(question):
            start = int(match.group("start"))
            end = int(match.group("end"))
            step = 1 if end >= start else -1

            years.update(range(start, end + step, step))

    return tuple(sorted(years))


def _concepts_used(
    *,
    cypher: str,
    parameters: dict[
        str,
        Any,
    ],
) -> set[str]:
    """Collect canonical XBRL concepts referenced by query text or parameters."""

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


def _final_return_clause(
    cypher: str,
) -> str:
    """Return the final RETURN projection from one validated Cypher statement."""

    code_view = _mask_string_literals(cypher)

    matches = list(
        re.finditer(
            r"\bRETURN\b",
            code_view,
            flags=re.IGNORECASE,
        )
    )

    if not matches:
        raise CypherValidationError("Generated Cypher must contain RETURN.")

    return cypher[matches[-1].start() :]


def _validate_final_projection_payload(
    cypher: str,
) -> None:
    """Reject unbounded collection aggregation in the final result projection."""

    final_return = _final_return_clause(cypher)

    code_view = _mask_string_literals(final_return)

    if FINAL_RETURN_COLLECTION_AGGREGATE_PATTERN.search(code_view) is not None:
        raise CypherValidationError(
            "Generated Cypher must not RETURN collect(...) in the final projection. "
            "Return bounded scalar fields instead."
        )


def validate_financial_projection_contract(
    cypher: str,
) -> None:
    """Require final financial projections to expose direct raw-fact identity fields."""

    final_return = _final_return_clause(cypher)

    missing = [
        field
        for (
            field,
            pattern,
        ) in _FINANCIAL_PROJECTION_PATTERNS
        if pattern.search(final_return) is None
    ]

    if missing:
        raise CypherValidationError(
            "Financial queries must RETURN direct raw-fact "
            "identity fields in the final RETURN clause. Missing projection(s): "
            + ", ".join(missing)
            + "."
        )


def _validate_financial_metric_alignment(
    *,
    question: str,
    cypher: str,
    parameters: dict[
        str,
        Any,
    ],
) -> None:
    """Reject Cypher whose canonical concepts do not match the requested metric."""

    metric = requested_financial_metric(question)

    if metric is None:
        return

    allowed = set(CANONICAL_FINANCIAL_METRICS[metric])

    used = _concepts_used(
        cypher=cypher,
        parameters=parameters,
    )

    if not used:
        raise CypherValidationError(
            "Generated Cypher does not reference a canonical "
            "XBRL concept for requested metric "
            f"'{metric}'."
        )

    unexpected = used - allowed

    if unexpected:
        raise CypherValidationError(
            "Generated Cypher mixes canonical concepts outside "
            f"requested metric '{metric}': " + ", ".join(sorted(unexpected)) + "."
        )

    validate_financial_projection_contract(cypher)


def _row_economic_year(
    row: dict[
        str,
        Any,
    ],
) -> int | None:
    """Extract a normalized economic year from one result row."""

    for (
        key,
        value,
    ) in row.items():
        if key.lower() not in _RESULT_ECONOMIC_YEAR_KEYS:
            continue

        if value is None:
            return None

        try:
            return int(str(value))

        except (
            TypeError,
            ValueError,
        ):
            return None

    return None


def _parse_iso_date(
    value: object,
    *,
    field_name: str,
    row_index: int,
) -> date:
    """Parse one required ISO date from a financial result row."""

    if value is None:
        raise CypherValidationError(f"Financial result row {row_index} is missing {field_name}.")

    try:
        return date.fromisoformat(str(value))

    except ValueError as exc:
        raise CypherValidationError(
            f"Financial result row {row_index} has invalid {field_name}: {value}."
        ) from exc


def validate_financial_row_identity(
    *,
    row: dict[
        str,
        Any,
    ],
    row_index: int,
    allowed_concepts: set[str],
    require_annual_duration: bool,
) -> None:
    """Validate one raw financial row and its period/fact identity."""

    required_non_empty = (
        "ticker",
        "metric_concept",
        "unit",
        "fact_id",
        "accession_number",
        "end_date",
    )

    missing = [
        field
        for field in required_non_empty
        if (row.get(field) is None or str(row.get(field)).strip() == "")
    ]

    if row.get("value") is None:
        missing.append("value")

    year = _row_economic_year(row)

    if year is None:
        missing.append("economic_year")

    if missing:
        raise CypherValidationError(
            f"Financial result row {row_index} "
            "is missing required field(s): " + ", ".join(sorted(set(missing))) + "."
        )

    concept = str(row["metric_concept"])

    if concept not in allowed_concepts:
        raise CypherValidationError(
            f"Financial result row {row_index} contains unexpected concept: {concept}."
        )

    try:
        numeric_value = Decimal(str(row["value"]))

    except InvalidOperation as exc:
        raise CypherValidationError(
            f"Financial result row {row_index} contains a non-numeric value."
        ) from exc

    if not numeric_value.is_finite():
        raise CypherValidationError(
            f"Financial result row {row_index} contains a non-finite value."
        )

    end_date = _parse_iso_date(
        row.get("end_date"),
        field_name="end_date",
        row_index=row_index,
    )

    if year is not None and end_date.year != year:
        raise CypherValidationError(
            f"Financial result row {row_index} economic year {year} "
            f"does not match end_date year {end_date.year}."
        )

    if not require_annual_duration:
        return

    start_date = _parse_iso_date(
        row.get("start_date"),
        field_name="start_date",
        row_index=row_index,
    )

    duration_days = (end_date - start_date).days

    if duration_days < 300 or duration_days > 400:
        raise CypherValidationError(
            f"Financial result row {row_index} "
            "is not an annual-duration observation "
            f"({duration_days} days)."
        )


def validate_result_rows(
    *,
    question: str,
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
) -> None:
    """Validate raw financial result identity, company, period, metric and unit."""

    metric = requested_financial_metric(question)

    if metric is None or not rows:
        return

    requested_tickers = set(requested_company_tickers(question))

    requested_years = set(requested_economic_years(question))

    allowed_concepts = set(CANONICAL_FINANCIAL_METRICS[metric])

    require_annual_duration = metric in _FLOW_FINANCIAL_METRICS

    for (
        index,
        row,
    ) in enumerate(rows):
        validate_financial_row_identity(
            row=row,
            row_index=index,
            allowed_concepts=(allowed_concepts),
            require_annual_duration=(require_annual_duration),
        )

    actual_tickers = {str(row["ticker"]).upper() for row in rows}

    if requested_tickers:
        unexpected = actual_tickers - requested_tickers

        if unexpected:
            raise CypherValidationError(
                "Financial results contain unrequested ticker(s): "
                + ", ".join(sorted(unexpected))
                + "."
            )

        missing_tickers = requested_tickers - actual_tickers

        if missing_tickers:
            raise CypherValidationError(
                "Financial results are missing requested ticker(s): "
                + ", ".join(sorted(missing_tickers))
                + "."
            )

    actual_years = {year for row in rows if (year := _row_economic_year(row)) is not None}

    if requested_years:
        unexpected_years = actual_years - requested_years

        if unexpected_years:
            raise CypherValidationError(
                "Financial results contain unrequested economic year(s): "
                + ", ".join(str(year) for year in sorted(unexpected_years))
                + "."
            )

        missing_years = requested_years - actual_years

        if missing_years:
            raise CypherValidationError(
                "Financial results are missing requested economic year(s): "
                + ", ".join(str(year) for year in sorted(missing_years))
                + "."
            )

        if requested_tickers:
            actual_company_years = {
                (
                    str(row["ticker"]).upper(),
                    year,
                )
                for row in rows
                if (year := _row_economic_year(row)) is not None
            }

            missing_company_years = {
                (
                    ticker,
                    year,
                )
                for ticker in requested_tickers
                for year in requested_years
                if (
                    ticker,
                    year,
                )
                not in actual_company_years
            }

            if missing_company_years:
                formatted = ", ".join(
                    f"{ticker} {year}"
                    for (
                        ticker,
                        year,
                    ) in sorted(missing_company_years)
                )

                raise CypherValidationError(
                    "Financial results are missing requested "
                    "company/economic-year coverage: "
                    f"{formatted}."
                )

    units = {str(row["unit"]) for row in rows}

    if len(units) != 1:
        raise CypherValidationError(
            "Financial results mix incompatible units: " + ", ".join(sorted(units)) + "."
        )


def _remove_comments(
    cypher: str,
) -> str:
    return COMMENT_PATTERN.sub(
        "",
        cypher,
    ).strip()


def _validate_single_statement(
    cypher: str,
) -> None:
    stripped = cypher.rstrip()

    if ";" not in stripped:
        return

    if stripped.count(";") == 1 and stripped.endswith(";"):
        return

    raise CypherValidationError("Only one Cypher statement is allowed.")


def _mask_string_literals(
    cypher: str,
) -> str:
    """Mask quoted string content while preserving character positions."""

    characters = list(cypher)

    quote: str | None = None
    escaped = False
    index = 0

    while index < len(characters):
        character = characters[index]

        if quote is None:
            if character in {
                "'",
                '"',
            }:
                quote = character

                characters[index] = " "

            index += 1

            continue

        characters[index] = " "

        if escaped:
            escaped = False
            index += 1

            continue

        if character == "\\":
            escaped = True
            index += 1

            continue

        if character == quote:
            if index + 1 < len(characters) and characters[index + 1] == quote:
                characters[index + 1] = " "

                index += 2

                continue

            quote = None

        index += 1

    return "".join(characters)


def _validate_forbidden_clauses(
    cypher: str,
) -> None:
    code_view = _mask_string_literals(cypher)

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(
            pattern,
            code_view,
            flags=re.IGNORECASE,
        ):
            raise CypherValidationError(
                f"Generated Cypher contains a forbidden clause matching: {pattern}"
            )


def _validate_query_shape(
    cypher: str,
) -> None:
    if not re.search(
        r"\bMATCH\b",
        cypher,
        flags=re.IGNORECASE,
    ):
        raise CypherValidationError("Generated Cypher must contain MATCH.")

    if not re.search(
        r"\bRETURN\b",
        cypher,
        flags=re.IGNORECASE,
    ):
        raise CypherValidationError("Generated Cypher must contain RETURN.")


def _match_clause_bodies(
    cypher: str,
) -> tuple[str, ...]:
    """Return MATCH pattern bodies with strings masked and non-pattern clauses removed."""

    code_view = _mask_string_literals(cypher)
    bodies: list[str] = []

    matches = list(MATCH_CLAUSE_PATTERN.finditer(code_view))

    for index, match in enumerate(matches):
        body_start = match.end()
        search_end = matches[index + 1].start() if index + 1 < len(matches) else len(code_view)
        candidate = code_view[body_start:search_end]

        terminator = MATCH_CLAUSE_TERMINATOR_PATTERN.search(candidate)

        if terminator is not None:
            candidate = candidate[: terminator.start()]

        bodies.append(candidate)

    return tuple(bodies)


def _schema_identifier(
    match: re.Match[str],
    *,
    quoted_group: str,
    plain_group: str,
) -> str:
    """Return one identifier captured either quoted or unquoted."""

    value = match.group(quoted_group) or match.group(plain_group)

    if value is None:
        raise CypherValidationError("Generated Cypher contains an invalid schema identifier.")

    return value


def _node_binding(
    content: str,
) -> tuple[str | None, str | None]:
    """Return the node variable and optional explicit label for one MATCH pattern."""

    match = NODE_BINDING_PATTERN.match(content)

    if match is not None:
        label = _schema_identifier(
            match,
            quoted_group="label_quoted",
            plain_group="label_plain",
        )

        return (
            match.group("variable"),
            label,
        )

    header = content.split(
        "{",
        maxsplit=1,
    )[0].strip()

    if re.fullmatch(IDENTIFIER_PATTERN, header) is not None:
        return (
            header,
            None,
        )

    raise CypherValidationError(
        "Generated Cypher MATCH node patterns must use one explicit approved node label."
    )


def _relationship_binding(
    content: str,
) -> tuple[str | None, str]:
    """Return the optional variable and required type for one MATCH relationship pattern."""

    match = RELATIONSHIP_BINDING_PATTERN.match(content)

    if match is None:
        raise CypherValidationError(
            "Generated Cypher MATCH relationships must use one explicit approved relationship type."
        )

    relationship_type = _schema_identifier(
        match,
        quoted_group="type_quoted",
        plain_group="type_plain",
    )

    return (
        match.group("variable"),
        relationship_type,
    )


def _pattern_map_properties(
    content: str,
) -> set[str]:
    """Return literal property-map keys used inside one pattern."""

    open_brace = content.find("{")

    if open_brace < 0:
        return set()

    close_brace = content.rfind("}")

    if close_brace < open_brace:
        raise CypherValidationError(
            "Generated Cypher contains an unterminated pattern property map."
        )

    body = content[open_brace + 1 : close_brace]

    properties: set[str] = set()

    for match in PATTERN_MAP_PROPERTY_PATTERN.finditer(body):
        property_name = match.group("quoted") or match.group("plain")

        if property_name is not None:
            properties.add(property_name)

    return properties


def _validate_pattern_connection_types(
    body: str,
) -> None:
    """Reject anonymous/untyped relationships connecting MATCH node patterns."""

    nodes = list(NODE_PATTERN_CONTENT.finditer(body))

    for left, right in zip(nodes, nodes[1:], strict=False):
        between = body[left.end() : right.start()]

        if "-" not in between:
            continue

        if "[" not in between or "]" not in between:
            raise CypherValidationError(
                "Generated Cypher MATCH relationships must "
                "use one explicit approved relationship type."
            )


def _validate_node_property_accesses(
    *,
    cypher: str,
    variable_labels: dict[str, str],
) -> None:
    """Validate node properties referenced through known MATCH variables."""

    code_view = _mask_string_literals(cypher)

    for variable, label in variable_labels.items():
        allowed_properties = ALLOWED_NODE_PROPERTIES[label]

        dynamic_property_pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(variable)}\s*\[")

        if dynamic_property_pattern.search(code_view) is not None:
            raise CypherValidationError(
                f"Generated Cypher uses dynamic property access on node variable '{variable}'."
            )

        properties_function_pattern = re.compile(
            rf"\bproperties\s*\(\s*{re.escape(variable)}\s*\)",
            flags=re.IGNORECASE,
        )

        if properties_function_pattern.search(code_view) is not None:
            raise CypherValidationError(
                f"Generated Cypher must not use properties({variable}) because it bypasses "
                "the explicit property allowlist."
            )

        map_projection_pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(variable)}\s*\{{")

        if map_projection_pattern.search(code_view) is not None:
            raise CypherValidationError(
                f"Generated Cypher must not use a map projection on node variable '{variable}' "
                "because it can bypass the explicit property allowlist."
            )

        property_access_pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(variable)}\s*\.\s*"
            rf"(?:`(?P<quoted>{IDENTIFIER_PATTERN})`|(?P<plain>{IDENTIFIER_PATTERN}))"
        )

        used_properties = {
            property_name
            for match in property_access_pattern.finditer(code_view)
            if (property_name := (match.group("quoted") or match.group("plain"))) is not None
        }

        invalid_properties = used_properties - allowed_properties

        if invalid_properties:
            raise CypherValidationError(
                f"Unknown property/properties for node label '{label}': "
                + ", ".join(sorted(invalid_properties))
            )


def _validate_relationship_property_accesses(
    *,
    cypher: str,
    relationship_variables: set[str],
) -> None:
    """Reject relationship-property access because the controlled schema defines none."""

    code_view = _mask_string_literals(cypher)

    for variable in relationship_variables:
        direct_property_pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(variable)}\s*\.")
        dynamic_property_pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(variable)}\s*\[")
        properties_function_pattern = re.compile(
            rf"\bproperties\s*\(\s*{re.escape(variable)}\s*\)",
            flags=re.IGNORECASE,
        )
        map_projection_pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(variable)}\s*\{{")

        if (
            direct_property_pattern.search(code_view) is not None
            or dynamic_property_pattern.search(code_view) is not None
            or properties_function_pattern.search(code_view) is not None
            or map_projection_pattern.search(code_view) is not None
        ):
            raise CypherValidationError(
                f"Generated Cypher references properties on relationship variable '{variable}', "
                "but the controlled graph schema defines no relationship properties."
            )


def _validate_schema_boundary(
    cypher: str,
) -> None:
    """Enforce explicit approved labels/types and node-property allowlists."""

    variable_labels: dict[str, str] = {}
    relationship_variables: set[str] = set()

    bodies = _match_clause_bodies(cypher)

    if not bodies:
        raise CypherValidationError("Generated Cypher must contain MATCH.")

    for body in bodies:
        _validate_pattern_connection_types(body)

        node_matches = list(NODE_PATTERN_CONTENT.finditer(body))

        if not node_matches:
            raise CypherValidationError(
                "Generated Cypher MATCH clauses must contain labelled node patterns."
            )

        for node_match in node_matches:
            content = node_match.group(1)

            variable, explicit_label = _node_binding(content)

            if explicit_label is None:
                if variable is None or variable not in variable_labels:
                    raise CypherValidationError(
                        "Generated Cypher MATCH node patterns must use one explicit approved "
                        "node label when a variable is first introduced."
                    )

                label = variable_labels[variable]

            else:
                label = explicit_label

                if label not in ALLOWED_NODE_LABELS:
                    raise CypherValidationError(f"Unknown node label(s): {label}")

                if variable is not None:
                    previous_label = variable_labels.get(variable)

                    if previous_label is not None and previous_label != label:
                        raise CypherValidationError(
                            f"Node variable '{variable}' is bound to conflicting labels: "
                            f"{previous_label}, {label}."
                        )

                    variable_labels[variable] = label

            invalid_map_properties = (
                _pattern_map_properties(content) - ALLOWED_NODE_PROPERTIES[label]
            )

            if invalid_map_properties:
                raise CypherValidationError(
                    f"Unknown property/properties for node label '{label}': "
                    + ", ".join(sorted(invalid_map_properties))
                )

        for relationship_match in RELATIONSHIP_PATTERN_CONTENT.finditer(body):
            content = relationship_match.group(1)

            relationship_variable, relationship_type = _relationship_binding(content)

            if relationship_type not in ALLOWED_RELATIONSHIP_TYPES:
                raise CypherValidationError(f"Unknown relationship type(s): {relationship_type}")

            if _pattern_map_properties(content):
                raise CypherValidationError(
                    "Generated Cypher relationship patterns must not use properties because "
                    "the controlled graph schema defines no relationship properties."
                )

            if relationship_variable is not None:
                relationship_variables.add(relationship_variable)

    _validate_node_property_accesses(
        cypher=cypher,
        variable_labels=variable_labels,
    )

    _validate_relationship_property_accesses(
        cypher=cypher,
        relationship_variables=relationship_variables,
    )


def _validate_schema_pattern_expressions(
    cypher: str,
) -> None:
    """
    Reject unsupported node-label and relationship-type expressions.

    FinGraph's generated Text-to-Cypher contract permits at most one simple
    node label per node pattern and at most one simple relationship type per
    relationship pattern. Cypher alternatives and compound schema expressions
    are intentionally rejected rather than partially parsed.
    """

    code_view = _mask_string_literals(cypher)

    pattern_groups = (
        (
            "node label",
            NODE_PATTERN_CONTENT,
        ),
        (
            "relationship type",
            RELATIONSHIP_PATTERN_CONTENT,
        ),
    )

    for (
        schema_kind,
        pattern,
    ) in pattern_groups:
        for match in pattern.finditer(code_view):
            content = match.group(1)

            header = content.split(
                "{",
                maxsplit=1,
            )[0]

            if DYNAMIC_SCHEMA_TOKEN_PATTERN.search(header) is not None:
                raise CypherValidationError(
                    "Generated Cypher must not use dynamic node labels or relationship types."
                )

            if ":" not in header:
                continue

            if SCHEMA_EXPRESSION_OPERATOR_PATTERN.search(header) is not None:
                raise CypherValidationError(
                    f"Generated Cypher contains an unsupported {schema_kind} expression."
                )

            schema_tokens = SCHEMA_TOKEN_PATTERN.findall(header)

            if len(schema_tokens) > 1:
                raise CypherValidationError(
                    f"Generated Cypher must use at most one {schema_kind} per pattern."
                )


def _validate_schema_elements(
    cypher: str,
) -> None:
    _validate_schema_pattern_expressions(cypher)

    code_view = _mask_string_literals(cypher)

    labels = set(NODE_LABEL_PATTERN.findall(code_view))

    invalid_labels = labels - ALLOWED_NODE_LABELS

    if invalid_labels:
        raise CypherValidationError("Unknown node label(s): " + ", ".join(sorted(invalid_labels)))

    relationships = set(RELATIONSHIP_PATTERN.findall(code_view))

    invalid_relationships = relationships - ALLOWED_RELATIONSHIP_TYPES

    if invalid_relationships:
        raise CypherValidationError(
            "Unknown relationship type(s): " + ", ".join(sorted(invalid_relationships))
        )

    _validate_schema_boundary(cypher)


def _normalize_parameters(
    plan: CypherPlan,
) -> dict[
    str,
    Any,
]:
    parameters: dict[
        str,
        Any,
    ] = {}

    for parameter in plan.parameters:
        if parameter.name in parameters:
            raise CypherValidationError(f"Duplicate Cypher parameter: {parameter.name}")

        if parameter.kind == "scalar":
            if parameter.values:
                raise CypherValidationError(
                    f"Scalar parameter must have an empty values list: {parameter.name}"
                )

            parameters[parameter.name] = parameter.scalar

            continue

        if parameter.kind == "list":
            if parameter.scalar:
                raise CypherValidationError(
                    f"List parameter must have an empty scalar value: {parameter.name}"
                )

            parameters[parameter.name] = parameter.values

            continue

        raise CypherValidationError(f"Unsupported parameter kind: {parameter.kind}")

    return parameters


def _validate_parameter_references(
    cypher: str,
    parameters: dict[
        str,
        Any,
    ],
) -> None:
    code_view = _mask_string_literals(cypher)

    referenced = set(
        re.findall(
            r"\$([A-Za-z_][A-Za-z0-9_]*)",
            code_view,
        )
    )

    supplied = set(parameters)

    missing = referenced - supplied

    if missing:
        raise CypherValidationError("Missing parameter value(s): " + ", ".join(sorted(missing)))


def _apply_row_limit(
    cypher: str,
    *,
    max_rows: int,
) -> str:
    if max_rows <= 0:
        raise CypherValidationError("max_rows must be greater than zero.")

    normalized = cypher.strip().rstrip(";").strip()

    code_view = _mask_string_literals(normalized)

    limit_matches = list(
        re.finditer(
            r"(?<![$.A-Za-z0-9_])LIMIT\b",
            code_view,
            flags=re.IGNORECASE,
        )
    )

    if not limit_matches:
        return f"{normalized}\nLIMIT {max_rows}"

    if len(limit_matches) != 1:
        raise CypherValidationError(
            "Generated Cypher must contain at most one terminal LIMIT clause."
        )

    match = limit_matches[0]

    limit_tail = code_view[match.start() :]

    terminal_match = re.fullmatch(
        r"LIMIT\s+([1-9]\d*)\s*",
        limit_tail,
        flags=re.IGNORECASE,
    )

    if terminal_match is None:
        raise CypherValidationError(
            "Generated LIMIT must be one positive integer literal in the terminal clause."
        )

    value = int(terminal_match.group(1))

    if value > max_rows:
        raise CypherValidationError(f"Generated LIMIT exceeds maximum of {max_rows}.")

    return normalized


def validate_cypher(
    plan: CypherPlan,
    *,
    max_rows: int = 25,
    question: str | None = None,
) -> ValidatedCypher:
    """Validate and normalize generated read-only Cypher."""

    cypher = _remove_comments(plan.cypher)

    _validate_single_statement(cypher)

    _validate_forbidden_clauses(cypher)

    _validate_query_shape(cypher)

    _validate_final_projection_payload(cypher)

    _validate_schema_elements(cypher)

    parameters = _normalize_parameters(plan)

    _validate_parameter_references(
        cypher,
        parameters,
    )

    if question is not None:
        _validate_financial_metric_alignment(
            question=question,
            cypher=cypher,
            parameters=parameters,
        )

    limited_cypher = _apply_row_limit(
        cypher,
        max_rows=max_rows,
    )

    return ValidatedCypher(
        cypher=limited_cypher,
        parameters=parameters,
    )
