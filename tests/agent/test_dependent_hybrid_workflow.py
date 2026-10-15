"""Focused dependent-HYBRID workflow tests."""

from fingraph.agent import Evidence
from fingraph.agent.workflow import build_agent_graph


def _initial_state(question: str) -> dict[str, object]:
    return {
        "question": question,
        "route": None,
        "route_reason": None,
        "vector_evidence": [],
        "graph_evidence": [],
        "evidence": [],
        "errors": [],
    }


def test_dependent_hybrid_runs_graph_before_winner_scoped_vector() -> None:
    graph_questions: list[str] = []
    vector_questions: list[str] = []

    def graph_tool(question: str) -> list[Evidence]:
        graph_questions.append(question)

        return [
            Evidence(
                evidence_id="graph-comparison",
                source_type="graph",
                structured_payload={
                    "rows": [
                        {
                            "ticker": "AAPL",
                            "year": 2024,
                            "revenue": 391035000000.0,
                            "unit": "USD",
                        },
                        {
                            "ticker": "AMZN",
                            "year": 2024,
                            "revenue": 637959000000.0,
                            "unit": "USD",
                        },
                    ]
                },
            )
        ]

    def vector_tool(question: str) -> list[Evidence]:
        vector_questions.append(question)

        return [
            Evidence(
                evidence_id="amazon-risk",
                source_type="vector",
                ticker="AMZN",
                filing_id="amazon-2024",
                section_name="Item 1A. Risk Factors",
                text="Amazon risk evidence.",
                source_url="https://sec.example/amzn-2024",
                rank=1,
            )
        ]

    agent = build_agent_graph(
        vector_tool=vector_tool,
        graph_tool=graph_tool,
    )

    question = (
        "Between Apple and Amazon, which company had higher revenue in 2024, "
        "and how did that company describe its risks in its 2024 10-K?"
    )

    result = agent.invoke(_initial_state(question))

    assert result["route"] == "hybrid"
    assert result["dependent_hybrid"] is True
    assert result["selected_tickers"] == ["AMZN"]

    assert graph_questions == [
        "Between Apple and Amazon, which company had higher revenue in 2024?"
    ]

    assert len(vector_questions) == 1
    assert "AMZN" in vector_questions[0]
    assert "Apple" not in vector_questions[0]
    assert "Amazon" not in vector_questions[0]
    assert "2024" in vector_questions[0]
    assert "risks" in vector_questions[0].lower()

    assert [item.source_type for item in result["evidence"]] == [
        "graph",
        "vector",
    ]


def test_dependent_hybrid_stops_if_graph_has_no_comparable_rows() -> None:
    vector_questions: list[str] = []

    def graph_tool(question: str) -> list[Evidence]:
        assert question

        return [
            Evidence(
                evidence_id="empty-graph",
                source_type="graph",
                structured_payload={"rows": []},
            )
        ]

    def vector_tool(question: str) -> list[Evidence]:
        vector_questions.append(question)
        return []

    agent = build_agent_graph(
        vector_tool=vector_tool,
        graph_tool=graph_tool,
    )

    result = agent.invoke(
        _initial_state(
            "Between Apple and Amazon, which company had higher revenue "
            "in 2024, and how did that company describe its risks?"
        )
    )

    assert result["selected_tickers"] == []
    assert vector_questions == []
    assert result["vector_evidence"] == []
