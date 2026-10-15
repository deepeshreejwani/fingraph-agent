"""LLM interfaces used by FinGraph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fingraph.graph.qa.models import CypherPlan


class CypherGenerator(Protocol):
    """Interface for Text-to-Cypher generation."""

    def generate_cypher(
        self,
        *,
        question: str,
        schema: str,
        previous_cypher: str | None = None,
        error: str | None = None,
    ) -> CypherPlan:
        """Generate or repair a structured Cypher plan."""
        ...
