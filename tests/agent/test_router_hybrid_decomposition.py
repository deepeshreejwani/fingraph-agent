"""Regression tests for deterministic query routing and HYBRID decomposition."""

from fingraph.agent.router import (
    graph_question_for_hybrid,
    route_question,
)


def test_decomposes_management_driver_clause() -> None:
    question = (
        "How did Microsoft revenue change from 2024 to 2026, "
        "and what did management say drove the change?"
    )

    assert graph_question_for_hybrid(question) == (
        "How did Microsoft revenue change from 2024 to 2026?"
    )


def test_decomposes_company_driver_clause() -> None:
    question = (
        "How did NVIDIA net income change from 2024 to 2026, "
        "and what did the company say drove the change?"
    )

    assert graph_question_for_hybrid(question) == (
        "How did NVIDIA net income change from 2024 to 2026?"
    )


def test_existing_what_drove_pattern_remains_supported() -> None:
    question = "How did Microsoft revenue change from 2024 to 2026, and what drove the change?"

    assert graph_question_for_hybrid(question) == (
        "How did Microsoft revenue change from 2024 to 2026?"
    )


def test_unknown_compound_pattern_is_not_speculatively_split() -> None:
    question = "Compare Microsoft revenue from 2024 to 2026 with management commentary."

    assert graph_question_for_hybrid(question) == question


def test_management_driver_question_routes_hybrid() -> None:
    question = (
        "How did Microsoft revenue change from 2024 to 2026, "
        "and what did management say drove the change?"
    )

    decision = route_question(question)

    assert decision.route == "hybrid"


def test_operating_margin_routes_graph() -> None:
    decision = route_question("What was Microsoft's operating margin in 2026?")

    assert decision.route == "graph"


def test_net_margin_routes_graph() -> None:
    decision = route_question("What was Microsoft's net margin in 2026?")

    assert decision.route == "graph"


def test_percentage_change_routes_graph() -> None:
    decision = route_question(
        "What was the percentage change in Microsoft revenue from 2025 to 2026?"
    )

    assert decision.route == "graph"


def test_cagr_routes_graph() -> None:
    decision = route_question("What was Microsoft's revenue CAGR from 2024 to 2026?")

    assert decision.route == "graph"


def test_narrative_risk_question_routes_vector() -> None:
    decision = route_question("What risks does NVIDIA describe in its 2026 10-K?")

    assert decision.route == "vector"


def test_financial_and_risk_question_routes_hybrid() -> None:
    decision = route_question(
        "What was Microsoft revenue in 2026 and explain the business risks "
        "Microsoft described in its 2026 10-K?"
    )

    assert decision.route == "hybrid"
