"""Focused dependent-HYBRID decomposition tests."""

from fingraph.agent.router import (
    graph_question_for_hybrid,
    is_dependent_hybrid,
    narrative_question_for_dependent_hybrid,
)

QUESTION = (
    "Between Apple and Amazon, which company had higher revenue in 2024, "
    "and how did that company describe its risks in its 2024 10-K?"
)


def test_dependent_hybrid_is_detected() -> None:
    assert is_dependent_hybrid(QUESTION)


def test_dependent_graph_clause_is_isolated() -> None:
    assert graph_question_for_hybrid(QUESTION) == (
        "Between Apple and Amazon, which company had higher revenue in 2024?"
    )


def test_dependent_narrative_clause_uses_selected_ticker_only() -> None:
    narrative = narrative_question_for_dependent_hybrid(
        QUESTION,
        ("AMZN",),
    )

    assert narrative == ("how did AMZN describe its risks in its 2024 10-K?")
    assert "Apple" not in narrative
    assert "Amazon" not in narrative
