"""Controlled unsupported Graph QA behavior."""

from typing import Any

from fingraph.graph.qa.models import (
    CypherPlan,
)
from fingraph.graph.qa.service import (
    GraphQAService,
)


class UnsupportedGenerator:
    """Generator that reports an unsupported graph request."""

    def generate_cypher(
        self,
        *,
        question: str,
        schema: str,
        previous_cypher: str | None = None,
        error: str | None = None,
    ) -> CypherPlan:
        del question
        del schema
        del previous_cypher
        del error

        return CypherPlan(
            cypher="",
            parameters=[],
        )


class NeverCalledStore:
    """Neo4j store that must not execute unsupported requests."""

    def execute_read(
        self,
        cypher: str,
        *,
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del cypher
        del parameters

        raise AssertionError("Unsupported request must not reach Neo4j.")


def test_empty_cypher_is_controlled_unsupported_result() -> None:
    service = GraphQAService(
        store=NeverCalledStore(),  # type: ignore[arg-type]
        generator=UnsupportedGenerator(),
        max_attempts=2,
        max_rows=25,
    )

    result = service.ask("What was Microsoft free cash flow in 2026?")

    assert result.cypher == ""
    assert result.parameters == {}
    assert result.rows == []
    assert result.attempts == 1
