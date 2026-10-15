"""Tests for generated-Cypher preflight execution."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from fingraph.graph.qa.models import (
    CypherParameter,
    CypherPlan,
)
from fingraph.graph.qa.service import (
    GraphQAService,
)
from fingraph.storage.neo4j import (
    Neo4jQueryValidationError,
)
from fingraph.storage.neo4j.store import (
    Neo4jStore,
)


class FakeGenerator:
    """Return one deterministic Cypher plan."""

    def generate_cypher(
        self,
        **_: Any,
    ) -> CypherPlan:
        return CypherPlan(
            cypher=("MATCH (c:Company) WHERE c.ticker = $ticker RETURN c.ticker AS ticker"),
            parameters=[
                CypherParameter(
                    name="ticker",
                    kind="scalar",
                    scalar="MSFT",
                    values=[],
                )
            ],
        )


class RepairGenerator:
    """Return an invalid first plan and a repaired second plan."""

    def __init__(
        self,
    ) -> None:
        self.calls: list[
            dict[
                str,
                Any,
            ]
        ] = []

    def generate_cypher(
        self,
        *,
        question: str,
        schema: str,
        previous_cypher: str | None = None,
        error: str | None = None,
    ) -> CypherPlan:
        self.calls.append(
            {
                "question": question,
                "schema": schema,
                "previous_cypher": previous_cypher,
                "error": error,
            }
        )

        if len(self.calls) == 1:
            return CypherPlan(
                cypher=("MATCH (c:Company) WHERE c.ticker = $ticker RETURN c.ticker AS ticker"),
                parameters=[
                    CypherParameter(
                        name="ticker",
                        kind="scalar",
                        scalar="MSFT",
                        values=[],
                    )
                ],
            )

        return CypherPlan(
            cypher=("MATCH (c:Company) WHERE c.ticker = $ticker RETURN c.ticker AS ticker"),
            parameters=[
                CypherParameter(
                    name="ticker",
                    kind="scalar",
                    scalar="MSFT",
                    values=[],
                )
            ],
        )


class FakeStore:
    """Record graph preflight and execution order."""

    def __init__(
        self,
    ) -> None:
        self.calls: list[str] = []

    def explain_read(
        self,
        query: str,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.calls.append("explain")

        assert "LIMIT 25" in query
        assert parameters == {"ticker": "MSFT"}

    def execute_read(
        self,
        query: str,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> list[
        dict[
            str,
            Any,
        ]
    ]:
        self.calls.append("execute")

        assert "LIMIT 25" in query
        assert parameters == {"ticker": "MSFT"}

        return [
            {
                "ticker": "MSFT",
            }
        ]


class RepairStore:
    """Reject the first EXPLAIN attempt and accept the repaired attempt."""

    def __init__(
        self,
    ) -> None:
        self.explain_calls = 0
        self.execute_calls = 0

    def explain_read(
        self,
        query: str,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.explain_calls += 1

        assert "LIMIT 25" in query
        assert parameters == {"ticker": "MSFT"}

        if self.explain_calls == 1:
            raise Neo4jQueryValidationError(
                "Neo4j EXPLAIN classified generated Cypher as non-read (query_type=rw)."
            )

    def execute_read(
        self,
        query: str,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> list[
        dict[
            str,
            Any,
        ]
    ]:
        self.execute_calls += 1

        assert "LIMIT 25" in query
        assert parameters == {"ticker": "MSFT"}

        return [
            {
                "ticker": "MSFT",
            }
        ]


def test_graph_qa_explains_before_execution() -> None:
    store = FakeStore()

    service = GraphQAService(
        store=store,  # type: ignore[arg-type]
        generator=FakeGenerator(),
        max_attempts=1,
        max_rows=25,
    )

    result = service.ask("What is Microsoft's ticker?")

    assert store.calls == [
        "explain",
        "execute",
    ]

    assert result.rows == [
        {
            "ticker": "MSFT",
        }
    ]


def test_graph_qa_repairs_after_explain_query_type_rejection() -> None:
    generator = RepairGenerator()

    store = RepairStore()

    service = GraphQAService(
        store=store,  # type: ignore[arg-type]
        generator=generator,
        max_attempts=2,
        max_rows=25,
    )

    result = service.ask("What is Microsoft's ticker?")

    assert result.attempts == 2

    assert len(generator.calls) == 2

    assert generator.calls[0]["error"] is None

    assert generator.calls[1]["previous_cypher"] is not None

    second_error = str(generator.calls[1]["error"])

    assert "classified generated Cypher as non-read" in second_error

    assert store.explain_calls == 2
    assert store.execute_calls == 1

    assert result.rows == [
        {
            "ticker": "MSFT",
        }
    ]


def _store_with_explain_query_type(
    query_type: str | None,
) -> Neo4jStore:
    """Build a store test double without opening a real connection."""

    store = object.__new__(Neo4jStore)

    driver = Mock()

    driver.execute_query.return_value = (
        [],
        SimpleNamespace(query_type=query_type),
        [],
    )

    cast(
        Any,
        store,
    ).driver = driver

    cast(
        Any,
        store,
    ).database = "neo4j"

    cast(
        Any,
        store,
    ).read_timeout_seconds = None

    return store


def test_neo4j_explain_accepts_read_only_query_type() -> None:
    store = _store_with_explain_query_type("r")

    store.explain_read(
        "MATCH (c:Company) RETURN c.ticker AS ticker LIMIT 25",
    )


@pytest.mark.parametrize(
    "query_type",
    [
        "rw",
        "w",
        "s",
        None,
    ],
)
def test_neo4j_explain_rejects_non_read_query_types(
    query_type: str | None,
) -> None:
    store = _store_with_explain_query_type(query_type)

    with pytest.raises(
        Neo4jQueryValidationError,
        match=("classified generated Cypher as non-read"),
    ):
        store.explain_read(
            "MATCH (c:Company) RETURN c.ticker AS ticker LIMIT 25",
        )
