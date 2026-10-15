"""Canonical state and routing models for FinGraph."""

from typing import (
    Literal,
    NotRequired,
    TypedDict,
)

from fingraph.agent.evidence import (
    Evidence,
)

Route = Literal[
    "vector",
    "graph",
    "hybrid",
]


class AgentState(TypedDict):
    """Explicit FinGraph LangGraph workflow state."""

    question: str

    route: Route | None
    route_reason: str | None

    dependent_hybrid: NotRequired[bool]
    selected_tickers: NotRequired[list[str]]

    vector_evidence: list[Evidence]

    graph_evidence: list[Evidence]

    evidence: list[Evidence]

    errors: list[str]
