"""Models for safe graph question answering."""

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class CypherParameter(BaseModel):
    """One parameter supplied to a generated Cypher query."""

    model_config = ConfigDict(
        extra="forbid",
    )

    name: str = Field(
        min_length=1,
    )

    kind: Literal[
        "scalar",
        "list",
    ]

    scalar: str

    values: list[str]


class CypherPlan(BaseModel):
    """
    Structured Text-to-Cypher generation result.

    An empty Cypher string is a controlled unsupported result. It means
    the supplied graph schema cannot answer the question without
    inventing entities, metrics, concepts, or relationships.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    cypher: str

    parameters: list[CypherParameter]


class ValidatedCypher(BaseModel):
    """Cypher query after application guardrails."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    cypher: str

    parameters: dict[
        str,
        Any,
    ]


class GraphQAResult(BaseModel):
    """Final deterministic graph execution result."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    question: str

    cypher: str

    parameters: dict[
        str,
        Any,
    ]

    rows: list[dict[str, Any]]

    attempts: int
