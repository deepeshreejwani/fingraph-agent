"""FinGraph controlled agent workflow."""

from fingraph.agent.calculations import (
    FinancialCAGR,
    FinancialChange,
    FinancialMargin,
    calculate_cagr,
    calculate_change,
    calculate_margin,
    calculate_series_change,
)
from fingraph.agent.evidence import (
    Evidence,
    EvidenceSource,
)
from fingraph.agent.models import (
    AgentState,
    Route,
)
from fingraph.agent.router import (
    RouteDecision,
    route_question,
)
from fingraph.agent.tools import (
    GraphEvidenceTool,
    VectorEvidenceTool,
)
from fingraph.agent.workflow import (
    build_agent_graph,
)

__all__ = [
    "AgentState",
    "Evidence",
    "EvidenceSource",
    "FinancialCAGR",
    "FinancialChange",
    "FinancialMargin",
    "GraphEvidenceTool",
    "Route",
    "RouteDecision",
    "VectorEvidenceTool",
    "build_agent_graph",
    "calculate_cagr",
    "calculate_change",
    "calculate_margin",
    "calculate_series_change",
    "route_question",
]
