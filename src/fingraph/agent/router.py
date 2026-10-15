"""Deterministic FinGraph query router."""

import re
from dataclasses import dataclass

from fingraph.agent.models import Route


@dataclass(frozen=True)
class RouteDecision:
    """Routing decision for one user question."""

    route: Route
    reason: str


STRUCTURED_FINANCIAL_TERMS = frozenset(
    {
        "revenue",
        "net income",
        "operating income",
        "operating margin",
        "net margin",
        "assets",
        "total assets",
        "liabilities",
        "total liabilities",
        "cash",
    }
)


# Financial measures that should still enter the controlled graph path even
# when FinGraph's canonical metric catalog may not currently support them.
#
# This is intentional: unsupported structured financial questions should fail
# safely through Graph QA rather than being treated as narrative retrieval.
CONTROLLED_FINANCIAL_TERMS = frozenset(
    {
        "ebitda",
        "ebit",
        "free cash flow",
        "operating cash flow",
        "cash flow",
        "gross profit",
        "earnings per share",
        "eps",
    }
)


STRUCTURED_GRAPH_TERMS = frozenset(
    {
        "ticker",
        "company",
        "companies",
        "filing date",
        "filing dates",
        "available filings",
        "list filings",
        "show filings",
        "accession",
        "accession number",
    }
)


NARRATIVE_TERMS = frozenset(
    {
        "risk",
        "risks",
        "risk factors",
        "strategy",
        "strategic",
        "management",
        "discussion",
        "explain",
        "explanation",
        "describe",
        "description",
        "business",
        "competition",
        "competitive",
        "market risk",
        "md&a",
        "outlook",
        "narrative",
        "drivers",
    }
)


HYBRID_TERMS = frozenset(
    {
        "why",
        "explain why",
        "what drove",
        "what caused",
        "drivers of",
        "reason for",
        "reasons for",
        "context for",
        "and explain",
        "along with",
    }
)


# Compound HYBRID questions frequently contain one graph-answerable clause
# followed by a qualitative clause. The graph branch must receive only the
# structured clause; otherwise Graph QA can correctly reject the entire
# compound question because the narrative portion is outside its schema.
#
# Keep these separators deterministic and deliberately conservative.
HYBRID_GRAPH_CLAUSE_MARKERS = (
    " and how did that company",
    " and how does that company",
    " and what risks did that company",
    " and what risk factors did that company",
    " and explain",
    " and what does management",
    " and what did management",
    " and what management",
    " and what did the company",
    " and what does the company",
    " and what the company",
    " and what drove",
    " and what caused",
    " and why",
    " and describe",
    " and discuss",
    " along with",
    " while explaining",
)


DEPENDENT_HYBRID_REFERENCES = (
    "that company",
    "that firm",
    "the higher-revenue company",
    "the lower-revenue company",
    "the selected company",
    "the winning company",
    "those companies",
)


def _hybrid_split_position(
    question: str,
) -> int | None:
    """Return the earliest deterministic graph/narrative split position."""

    normalized = question.strip().lower()

    positions = [
        position
        for marker in HYBRID_GRAPH_CLAUSE_MARKERS
        if (position := normalized.find(marker)) > 0
    ]

    return min(positions) if positions else None


def is_dependent_hybrid(
    question: str,
) -> bool:
    """Detect a HYBRID request whose narrative hop depends on graph output."""

    normalized = question.strip().lower()

    if not any(reference in normalized for reference in DEPENDENT_HYBRID_REFERENCES):
        return False

    return route_question(question).route == "hybrid"


def narrative_question_for_dependent_hybrid(
    question: str,
    selected_tickers: tuple[str, ...],
) -> str:
    """Build a bounded narrative sub-question from graph-selected companies."""

    if not selected_tickers:
        return ""

    split_at = _hybrid_split_position(question)

    if split_at is None:
        return ""

    narrative = question.strip()[split_at:].lstrip(" \t\r\n,;:-?.!")

    if narrative.lower().startswith("and "):
        narrative = narrative[4:].lstrip()

    selected_label = " and ".join(selected_tickers)

    for reference in sorted(
        DEPENDENT_HYBRID_REFERENCES,
        key=len,
        reverse=True,
    ):
        narrative = re.sub(
            re.escape(reference),
            selected_label,
            narrative,
            flags=re.IGNORECASE,
        )

    return narrative


def _contains_any(
    text: str,
    terms: frozenset[str],
) -> bool:
    return any(term in text for term in terms)


def graph_question_for_hybrid(
    question: str,
) -> str:
    """
    Extract the structured clause of a deterministic HYBRID question.

    Vector retrieval receives the complete original question because the
    narrative language is useful for section selection and retrieval.

    Graph QA receives only the graph-answerable prefix when the question
    clearly contains a structured clause followed by a qualitative clause.

    If no high-confidence separator exists, the original question is returned
    unchanged rather than attempting speculative decomposition.
    """

    stripped = question.strip()

    if not stripped:
        return stripped

    split_at = _hybrid_split_position(stripped)

    if split_at is None:
        return stripped

    graph_question = stripped[:split_at].rstrip(
        " \t\r\n,;:-?.!",
    )

    if not graph_question:
        return stripped

    return f"{graph_question}?"


def route_question(
    question: str,
) -> RouteDecision:
    """Route a question to vector, graph, or hybrid execution."""

    normalized = question.strip().lower()

    has_supported_financial = _contains_any(
        normalized,
        STRUCTURED_FINANCIAL_TERMS,
    )

    has_controlled_financial = _contains_any(
        normalized,
        CONTROLLED_FINANCIAL_TERMS,
    )

    has_financial = has_supported_financial or has_controlled_financial

    has_graph_metadata = _contains_any(
        normalized,
        STRUCTURED_GRAPH_TERMS,
    )

    has_narrative = _contains_any(
        normalized,
        NARRATIVE_TERMS,
    )

    has_hybrid_signal = _contains_any(
        normalized,
        HYBRID_TERMS,
    )

    has_graph = has_financial or has_graph_metadata

    # Explicit request combining structured values with
    # qualitative explanation.
    if has_hybrid_signal and has_graph:
        return RouteDecision(
            route="hybrid",
            reason=(
                "Question combines structured financial information with narrative explanation."
            ),
        )

    # A query containing both genuine graph content and
    # narrative content requires both retrieval systems.
    if has_graph and has_narrative:
        return RouteDecision(
            route="hybrid",
            reason=("Question requires both graph-resident facts and narrative evidence."),
        )

    if has_graph:
        return RouteDecision(
            route="graph",
            reason=(
                "Question primarily requests structured financial or graph-resident information."
            ),
        )

    return RouteDecision(
        route="vector",
        reason=("Question primarily requests qualitative or narrative evidence."),
    )
