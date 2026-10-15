"""Smoke test the FinGraph LangGraph routing workflow."""

import json
from typing import Any, cast

from fingraph.agent import (
    Evidence,
    build_agent_graph,
)


def fake_vector_tool(
    question: str,
) -> list[Evidence]:
    return cast(
        list[Evidence],
        [
            {
                "source": "vector",
                "question": question,
            }
        ],
    )


def fake_graph_tool(
    question: str,
) -> list[Evidence]:
    return cast(
        list[Evidence],
        [
            {
                "source": "graph",
                "question": question,
            }
        ],
    )


def run_question(
    graph: Any,
    question: str,
) -> None:
    initial_state: dict[str, Any] = {
        "question": question,
        "route": None,
        "route_reason": None,
        "vector_evidence": [],
        "graph_evidence": [],
        "evidence": [],
        "errors": [],
    }

    result = graph.invoke(initial_state)

    print(
        json.dumps(
            {
                "question": question,
                "route": result["route"],
                "reason": result["route_reason"],
                "evidence": result["evidence"],
            },
            indent=2,
        )
    )

    print()


def main() -> None:
    graph = build_agent_graph(
        vector_tool=fake_vector_tool,
        graph_tool=fake_graph_tool,
    )

    questions = [
        "What was Microsoft's revenue in 2026?",
        "What risks does NVIDIA describe in its 10-K?",
        (
            "How did Microsoft's revenue "
            "change from 2024 to 2026 "
            "and explain what management "
            "said drove the trend?"
        ),
        "Show Meta's available filings.",
        "Describe Apple's competitive risks.",
    ]

    for question in questions:
        run_question(
            graph,
            question,
        )


if __name__ == "__main__":
    main()
