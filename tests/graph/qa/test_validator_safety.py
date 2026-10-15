"""Regression tests for generated-Cypher read-only and row-limit safety."""

import pytest

from fingraph.graph.qa.models import (
    CypherParameter,
    CypherPlan,
)
from fingraph.graph.qa.validator import (
    CypherValidationError,
    validate_cypher,
)


def _plan(
    cypher: str,
) -> CypherPlan:
    return CypherPlan(
        cypher=cypher,
        parameters=[],
    )


@pytest.mark.parametrize(
    "clause",
    [
        'CREATE (:Company {cik: "x"})',
        'INSERT (:Company {cik: "x"})',
        'MERGE (:Company {cik: "x"})',
        "DELETE c",
        "DETACH DELETE c",
        'SET c.name = "x"',
        "REMOVE c.name",
    ],
)
def test_validator_rejects_write_clauses(
    clause: str,
) -> None:
    with pytest.raises(CypherValidationError):
        validate_cypher(
            _plan(
                f"""
MATCH (c:Company)
{clause}
RETURN c.cik AS cik
""".strip()
            ),
            max_rows=25,
        )


def test_write_keyword_inside_string_literal_does_not_trigger_clause_rejection() -> None:
    validated = validate_cypher(
        _plan(
            """
MATCH (c:Company)
RETURN "INSERT CREATE DELETE" AS note
""".strip()
        ),
        max_rows=25,
    )

    assert validated.cypher.endswith("LIMIT 25")


def test_missing_limit_gets_terminal_maximum() -> None:
    validated = validate_cypher(
        _plan("MATCH (c:Company) RETURN c.ticker AS ticker"),
        max_rows=25,
    )

    assert validated.cypher.endswith("LIMIT 25")


def test_terminal_constant_limit_within_bound_is_preserved() -> None:
    validated = validate_cypher(
        _plan("MATCH (c:Company) RETURN c.ticker AS ticker LIMIT 10"),
        max_rows=25,
    )

    assert validated.cypher.endswith("LIMIT 10")


def test_oversized_terminal_limit_is_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="exceeds maximum",
    ):
        validate_cypher(
            _plan("MATCH (c:Company) RETURN c.ticker AS ticker LIMIT 26"),
            max_rows=25,
        )


@pytest.mark.parametrize(
    "limit_clause",
    [
        "LIMIT 0",
        "LIMIT -1",
        "LIMIT 1 + 100000",
        "LIMIT toInteger(25)",
    ],
)
def test_non_literal_or_non_positive_terminal_limits_are_rejected(
    limit_clause: str,
) -> None:
    with pytest.raises(
        CypherValidationError,
        match="positive integer literal",
    ):
        validate_cypher(
            _plan(f"MATCH (c:Company) RETURN c.ticker AS ticker {limit_clause}"),
            max_rows=25,
        )


def test_parameterized_limit_is_rejected() -> None:
    plan = CypherPlan(
        cypher=("MATCH (c:Company) RETURN c.ticker AS ticker LIMIT $limit"),
        parameters=[
            CypherParameter(
                name="limit",
                kind="scalar",
                scalar="25",
                values=[],
            )
        ],
    )

    with pytest.raises(
        CypherValidationError,
        match="positive integer literal",
    ):
        validate_cypher(
            plan,
            max_rows=25,
        )


def test_intermediate_limit_is_rejected_even_when_small() -> None:
    with pytest.raises(
        CypherValidationError,
        match="positive integer literal",
    ):
        validate_cypher(
            _plan(
                """
MATCH (c:Company)
WITH c LIMIT 1
MATCH (f:FinancialFact)
RETURN f.value AS value
""".strip()
            ),
            max_rows=25,
        )


def test_multiple_limit_clauses_are_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="at most one terminal LIMIT",
    ):
        validate_cypher(
            _plan(
                """
MATCH (c:Company)
WITH c LIMIT 1
RETURN c.ticker AS ticker
LIMIT 1
""".strip()
            ),
            max_rows=25,
        )


def test_limit_keyword_inside_string_literal_does_not_suppress_outer_limit() -> None:
    validated = validate_cypher(
        _plan(
            """
MATCH (c:Company)
RETURN "LIMIT 999" AS note
""".strip()
        ),
        max_rows=25,
    )

    assert validated.cypher.endswith("LIMIT 25")


def test_non_positive_configured_max_rows_is_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match=("max_rows must be greater than zero"),
    ):
        validate_cypher(
            _plan("MATCH (c:Company) RETURN c.ticker AS ticker"),
            max_rows=0,
        )


def test_final_collect_projection_is_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="must not RETURN collect",
    ):
        validate_cypher(
            _plan(
                """
MATCH (f:FinancialFact)
RETURN collect(f) AS facts
LIMIT 25
""".strip()
            ),
            max_rows=25,
        )


def test_final_collect_projection_without_limit_is_rejected_before_limit_insertion() -> None:
    with pytest.raises(
        CypherValidationError,
        match="must not RETURN collect",
    ):
        validate_cypher(
            _plan(
                """
MATCH (f:FinancialFact)
RETURN collect(f.value) AS values
""".strip()
            ),
            max_rows=25,
        )


def test_collect_text_inside_string_literal_does_not_trigger_payload_rejection() -> None:
    validated = validate_cypher(
        _plan(
            """
MATCH (c:Company)
RETURN "collect(f) is documentation text" AS note
""".strip()
        ),
        max_rows=25,
    )

    assert validated.cypher.endswith("LIMIT 25")


def test_scalar_final_projection_remains_allowed() -> None:
    validated = validate_cypher(
        _plan(
            """
MATCH (f:FinancialFact)
RETURN f.value AS value
""".strip()
        ),
        max_rows=25,
    )

    assert validated.cypher.endswith("LIMIT 25")


def test_simple_allowed_node_label_is_accepted() -> None:
    validated = validate_cypher(
        _plan(
            """
MATCH (c:Company)
RETURN c.ticker AS ticker
""".strip()
        ),
        max_rows=25,
    )

    assert validated.cypher.endswith("LIMIT 25")


def test_simple_allowed_relationship_type_is_accepted() -> None:
    validated = validate_cypher(
        _plan(
            """
MATCH (c:Company)-[:FILED]->(f:Filing)
RETURN f.accession_number AS accession_number
""".strip()
        ),
        max_rows=25,
    )

    assert validated.cypher.endswith("LIMIT 25")


def test_unknown_simple_node_label_is_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="Unknown node label",
    ):
        validate_cypher(
            _plan(
                """
MATCH (n:UnexpectedLabel)
RETURN n
""".strip()
            ),
            max_rows=25,
        )


def test_unknown_simple_relationship_type_is_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="Unknown relationship type",
    ):
        validate_cypher(
            _plan(
                """
MATCH (c:Company)-[:UNEXPECTED_REL]->(f:Filing)
RETURN f
""".strip()
            ),
            max_rows=25,
        )


def test_alternate_node_label_with_unknown_label_is_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="unsupported node label expression",
    ):
        validate_cypher(
            _plan(
                """
MATCH (n:Company|UnexpectedLabel)
RETURN n
""".strip()
            ),
            max_rows=25,
        )


def test_alternate_relationship_type_with_unknown_type_is_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="unsupported relationship type expression",
    ):
        validate_cypher(
            _plan(
                """
MATCH (c:Company)-[:FILED|UNEXPECTED_REL]->(f:Filing)
RETURN f
""".strip()
            ),
            max_rows=25,
        )


def test_alternate_allowed_node_labels_are_still_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="unsupported node label expression",
    ):
        validate_cypher(
            _plan(
                """
MATCH (n:Company|Filing)
RETURN n
""".strip()
            ),
            max_rows=25,
        )


def test_alternate_allowed_relationship_types_are_still_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="unsupported relationship type expression",
    ):
        validate_cypher(
            _plan(
                """
MATCH (c:Company)-[:FILED|HAS_SECTION]->(f:Filing)
RETURN f
""".strip()
            ),
            max_rows=25,
        )


def test_multiple_node_labels_on_one_pattern_are_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="at most one node label",
    ):
        validate_cypher(
            _plan(
                """
MATCH (n:Company:Filing)
RETURN n
""".strip()
            ),
            max_rows=25,
        )


def test_schema_expression_text_inside_string_literal_does_not_trigger_rejection() -> None:
    validated = validate_cypher(
        _plan(
            """
MATCH (c:Company)
RETURN ":Company|UnexpectedLabel [:FILED|UNEXPECTED_REL]" AS note
""".strip()
        ),
        max_rows=25,
    )

    assert validated.cypher.endswith("LIMIT 25")


def test_property_map_colon_does_not_count_as_second_node_label() -> None:
    validated = validate_cypher(
        _plan(
            """
MATCH (c:Company {ticker: "AAPL"})
RETURN c.ticker AS ticker
""".strip()
        ),
        max_rows=25,
    )

    assert validated.cypher.endswith("LIMIT 25")


def test_unlabelled_node_pattern_is_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="must use one explicit approved node label",
    ):
        validate_cypher(
            _plan(
                """
MATCH (n)
RETURN n
""".strip()
            ),
            max_rows=25,
        )


def test_untyped_named_relationship_is_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="must use one explicit approved relationship type",
    ):
        validate_cypher(
            _plan(
                """
MATCH (c:Company)-[r]->(f:Filing)
RETURN f.accession_number AS accession_number
""".strip()
            ),
            max_rows=25,
        )


@pytest.mark.parametrize(
    "connector",
    [
        "--",
        "-->",
        "<--",
    ],
)
def test_untyped_anonymous_relationship_is_rejected(
    connector: str,
) -> None:
    with pytest.raises(
        CypherValidationError,
        match="must use one explicit approved relationship type",
    ):
        validate_cypher(
            _plan(
                f"""
MATCH (c:Company){connector}(f:Filing)
RETURN f.accession_number AS accession_number
""".strip()
            ),
            max_rows=25,
        )


def test_dynamic_node_label_is_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="must not use dynamic node labels or relationship types",
    ):
        validate_cypher(
            _plan(
                """
MATCH (n:$label)
RETURN n
""".strip()
            ),
            max_rows=25,
        )


def test_dynamic_relationship_type_is_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="must not use dynamic node labels or relationship types",
    ):
        validate_cypher(
            _plan(
                """
MATCH (c:Company)-[r:$type]->(f:Filing)
RETURN f.accession_number AS accession_number
""".strip()
            ),
            max_rows=25,
        )


def test_unknown_direct_node_property_is_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="Unknown property/properties for node label 'Company': password",
    ):
        validate_cypher(
            _plan(
                """
MATCH (c:Company)
RETURN c.password AS password
""".strip()
            ),
            max_rows=25,
        )


def test_unknown_pattern_map_property_is_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="Unknown property/properties for node label 'Company': password",
    ):
        validate_cypher(
            _plan(
                """
MATCH (c:Company {password: "secret"})
RETURN c.ticker AS ticker
""".strip()
            ),
            max_rows=25,
        )


def test_dynamic_node_property_access_is_rejected() -> None:
    plan = CypherPlan(
        cypher="""
MATCH (c:Company)
RETURN c[$property] AS value
""".strip(),
        parameters=[
            CypherParameter(
                name="property",
                kind="scalar",
                scalar="ticker",
                values=[],
            )
        ],
    )

    with pytest.raises(
        CypherValidationError,
        match="dynamic property access on node variable 'c'",
    ):
        validate_cypher(
            plan,
            max_rows=25,
        )


def test_properties_function_on_node_is_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match=r"must not use properties\(c\)",
    ):
        validate_cypher(
            _plan(
                """
MATCH (c:Company)
RETURN properties(c) AS company_properties
""".strip()
            ),
            max_rows=25,
        )


def test_node_map_projection_is_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="must not use a map projection on node variable 'c'",
    ):
        validate_cypher(
            _plan(
                """
MATCH (c:Company)
RETURN c{.*} AS company
""".strip()
            ),
            max_rows=25,
        )


def test_relationship_pattern_properties_are_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="relationship patterns must not use properties",
    ):
        validate_cypher(
            _plan(
                """
MATCH (c:Company)-[r:FILED {since: 2020}]->(f:Filing)
RETURN f.accession_number AS accession_number
""".strip()
            ),
            max_rows=25,
        )


def test_relationship_direct_property_access_is_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="references properties on relationship variable 'r'",
    ):
        validate_cypher(
            _plan(
                """
MATCH (c:Company)-[r:FILED]->(f:Filing)
RETURN r.since AS since
""".strip()
            ),
            max_rows=25,
        )


def test_known_pattern_map_property_remains_allowed() -> None:
    validated = validate_cypher(
        _plan(
            """
MATCH (c:Company {ticker: "AAPL"})
RETURN c.ticker AS ticker
""".strip()
        ),
        max_rows=25,
    )

    assert validated.cypher.endswith("LIMIT 25")


def test_known_properties_across_labels_remain_allowed() -> None:
    validated = validate_cypher(
        _plan(
            """
MATCH (c:Company)-[:FILED]->(f:Filing)
RETURN c.ticker AS ticker,
       f.accession_number AS accession_number,
       f.filing_date AS filing_date
""".strip()
        ),
        max_rows=25,
    )

    assert validated.cypher.endswith("LIMIT 25")


def test_repeated_node_variable_with_same_label_remains_allowed() -> None:
    validated = validate_cypher(
        _plan(
            """
MATCH (c:Company)-[:REPORTS]->(fact:FinancialFact)
MATCH (c:Company)-[:FILED]->(filing:Filing)
RETURN c.ticker AS ticker
""".strip()
        ),
        max_rows=25,
    )

    assert validated.cypher.endswith("LIMIT 25")


def test_node_variable_conflicting_labels_are_rejected() -> None:
    with pytest.raises(
        CypherValidationError,
        match="bound to conflicting labels",
    ):
        validate_cypher(
            _plan(
                """
MATCH (n:Company)
MATCH (n:Filing)
RETURN n
""".strip()
            ),
            max_rows=25,
        )


def test_property_like_text_inside_string_literal_does_not_trigger_property_rejection() -> None:
    validated = validate_cypher(
        _plan(
            """
MATCH (c:Company)
RETURN "c.password c[$property] properties(c)" AS note
""".strip()
        ),
        max_rows=25,
    )

    assert validated.cypher.endswith("LIMIT 25")
