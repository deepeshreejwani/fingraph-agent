"""Controlled LangGraph workflow for FinGraph."""

from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from fingraph.agent.evidence import (
    Evidence,
)
from fingraph.agent.models import (
    AgentState,
)
from fingraph.agent.router import (
    graph_question_for_hybrid,
    is_dependent_hybrid,
    narrative_question_for_dependent_hybrid,
    route_question,
)

EvidenceTool = Callable[
    [str],
    list[Evidence],
]

ExecutionPath = Literal[
    "vector",
    "graph",
    "hybrid",
    "dependent_hybrid",
]

_NON_METRIC_KEYS = frozenset(
    {
        "year",
        "fiscal_year",
        "economic_year",
        "period_year",
        "rank",
        "attempts",
    }
)


def _graph_rows(
    evidence: list[Evidence],
) -> list[dict[str, object]]:
    """Collect structured graph rows from normalized evidence."""

    rows: list[dict[str, object]] = []

    for item in evidence:
        if item.source_type != "graph":
            continue

        payload = item.structured_payload or {}
        raw_rows = payload.get("rows", [])

        if not isinstance(raw_rows, list):
            continue

        rows.extend(row for row in raw_rows if isinstance(row, dict))

    return rows


def _metric_value(
    row: dict[str, object],
) -> Decimal | None:
    """Extract exactly one comparable numeric metric from one graph row."""

    values: list[Decimal] = []

    for key, raw in row.items():
        if key.lower() in _NON_METRIC_KEYS:
            continue

        if isinstance(raw, bool) or not isinstance(
            raw,
            (
                int,
                float,
                Decimal,
            ),
        ):
            continue

        try:
            values.append(Decimal(str(raw)))

        except InvalidOperation:
            continue

    return values[0] if len(values) == 1 else None


def _row_year(
    row: dict[str, object],
) -> int | None:
    """Extract an economic year from one graph row."""

    for key in (
        "year",
        "fiscal_year",
        "economic_year",
        "period_year",
    ):
        raw = row.get(key)

        if raw is None:
            continue

        try:
            return int(str(raw))

        except ValueError:
            return None

    return None


def _select_tickers_from_graph_evidence(
    *,
    question: str,
    evidence: list[Evidence],
) -> tuple[str, ...]:
    """
    Select graph-result entities deterministically for a dependent narrative hop.

    Supported semantics:
    - higher/highest/larger/greater -> maximum comparable metric;
    - lower/lowest/smaller/less -> minimum comparable metric;
    - improving/increased/grew -> companies whose latest value exceeds earliest.
    """

    normalized = question.lower()
    rows = _graph_rows(evidence)

    observations: list[
        tuple[
            str,
            int | None,
            Decimal,
        ]
    ] = []

    for row in rows:
        raw_ticker = row.get("ticker")
        value = _metric_value(row)

        if raw_ticker is None or value is None:
            continue

        observations.append(
            (
                str(raw_ticker),
                _row_year(row),
                value,
            )
        )

    if not observations:
        return ()

    if any(
        term in normalized
        for term in (
            "improving",
            "increased",
            "increase over",
            "grew",
            "growth over",
        )
    ):
        grouped: dict[str, list[tuple[int, Decimal]]] = {}

        for ticker, year, value in observations:
            if year is None:
                continue

            grouped.setdefault(ticker, []).append((year, value))

        selected: list[str] = []

        for ticker, values in sorted(grouped.items()):
            ordered = sorted(values)

            if len(ordered) >= 2 and ordered[-1][1] > ordered[0][1]:
                selected.append(ticker)

        return tuple(selected)

    latest_by_ticker: dict[str, tuple[int | None, Decimal]] = {}

    for ticker, year, value in observations:
        current = latest_by_ticker.get(ticker)

        if current is None:
            latest_by_ticker[ticker] = (year, value)
            continue

        current_year = current[0]

        if year is not None and (current_year is None or year > current_year):
            latest_by_ticker[ticker] = (year, value)

    comparable = {ticker: value for ticker, (_, value) in latest_by_ticker.items()}

    if not comparable:
        return ()

    if any(
        term in normalized
        for term in (
            "lower",
            "lowest",
            "smaller",
            "less than",
        )
    ):
        target = min(comparable.values())

    elif any(
        term in normalized
        for term in (
            "higher",
            "highest",
            "larger",
            "greater",
            "more than",
        )
    ):
        target = max(comparable.values())

    else:
        return tuple(sorted(comparable))

    return tuple(sorted(ticker for ticker, value in comparable.items() if value == target))


def build_agent_graph(
    *,
    vector_tool: EvidenceTool,
    graph_tool: EvidenceTool,
) -> Any:
    """Build the controlled FinGraph execution graph."""

    def route_node(
        state: AgentState,
    ) -> dict[str, object]:
        decision = route_question(state["question"])

        dependent = decision.route == "hybrid" and is_dependent_hybrid(state["question"])

        return {
            "route": decision.route,
            "route_reason": decision.reason,
            "dependent_hybrid": dependent,
            "selected_tickers": [],
        }

    def vector_node(
        state: AgentState,
    ) -> dict[str, object]:
        evidence = vector_tool(state["question"])

        return {
            "vector_evidence": evidence,
            "evidence": evidence,
        }

    def graph_node(
        state: AgentState,
    ) -> dict[str, object]:
        evidence = graph_tool(state["question"])

        return {
            "graph_evidence": evidence,
            "evidence": evidence,
        }

    def hybrid_node(
        state: AgentState,
    ) -> dict[str, object]:
        question = state["question"]

        # Standard HYBRID requests are independent: narrative retrieval can use
        # the complete question while Graph QA receives the structured prefix.
        vector_evidence = vector_tool(question)
        graph_question = graph_question_for_hybrid(question)
        graph_evidence = graph_tool(graph_question)

        return {
            "vector_evidence": vector_evidence,
            "graph_evidence": graph_evidence,
            "evidence": graph_evidence + vector_evidence,
        }

    def dependent_graph_node(
        state: AgentState,
    ) -> dict[str, object]:
        question = state["question"]
        graph_question = graph_question_for_hybrid(question)
        graph_evidence = graph_tool(graph_question)

        selected_tickers = _select_tickers_from_graph_evidence(
            question=question,
            evidence=graph_evidence,
        )

        return {
            "graph_evidence": graph_evidence,
            "evidence": graph_evidence,
            "selected_tickers": list(selected_tickers),
        }

    def dependent_vector_node(
        state: AgentState,
    ) -> dict[str, object]:
        graph_evidence = list(state.get("graph_evidence", []))
        selected_tickers = tuple(state.get("selected_tickers", []))

        if not selected_tickers:
            return {
                "vector_evidence": [],
                "evidence": graph_evidence,
            }

        narrative_question = narrative_question_for_dependent_hybrid(
            state["question"],
            selected_tickers,
        )

        if not narrative_question:
            return {
                "vector_evidence": [],
                "evidence": graph_evidence,
            }

        vector_evidence = vector_tool(narrative_question)

        return {
            "vector_evidence": vector_evidence,
            "evidence": graph_evidence + vector_evidence,
        }

    def select_route(
        state: AgentState,
    ) -> ExecutionPath:
        route = state.get("route")

        if route is None:
            raise ValueError("Routing node did not produce a route.")

        if route == "hybrid" and bool(state.get("dependent_hybrid", False)):
            return "dependent_hybrid"

        return route

    builder = StateGraph(AgentState)

    builder.add_node(
        "route",
        route_node,
    )

    builder.add_node(
        "vector",
        vector_node,
    )

    builder.add_node(
        "graph",
        graph_node,
    )

    builder.add_node(
        "hybrid",
        hybrid_node,
    )

    builder.add_node(
        "dependent_graph",
        dependent_graph_node,
    )

    builder.add_node(
        "dependent_vector",
        dependent_vector_node,
    )

    builder.add_edge(
        START,
        "route",
    )

    builder.add_conditional_edges(
        "route",
        select_route,
        {
            "vector": "vector",
            "graph": "graph",
            "hybrid": "hybrid",
            "dependent_hybrid": "dependent_graph",
        },
    )

    builder.add_edge(
        "vector",
        END,
    )

    builder.add_edge(
        "graph",
        END,
    )

    builder.add_edge(
        "hybrid",
        END,
    )

    builder.add_edge(
        "dependent_graph",
        "dependent_vector",
    )

    builder.add_edge(
        "dependent_vector",
        END,
    )

    return builder.compile()
