"""Regression coverage for synthesis handling of unusable evidence."""

from typing import cast

from fingraph.agent import Evidence
from fingraph.answer.synthesis import (
    GroqAnswerSynthesizer,
)
from fingraph.llm.state import (
    LLMStateStore,
)


def _synthesizer_without_state_use() -> GroqAnswerSynthesizer:
    return GroqAnswerSynthesizer(
        api_key="test-key",
        model="openai/gpt-oss-120b",
        state_store=cast(
            LLMStateStore,
            object(),
        ),
    )


def test_empty_evidence_returns_controlled_insufficient_answer() -> None:
    (
        plan,
        calculations,
        citations,
    ) = _synthesizer_without_state_use().synthesize(
        question=("What was Microsoft's revenue in 2025?"),
        evidence=[],
    )

    assert plan.insufficient_evidence
    assert plan.cited_evidence_ids == []

    assert "Insufficient evidence" in plan.answer

    assert calculations == []
    assert citations == []


def test_empty_graph_wrapper_returns_controlled_insufficient_answer() -> None:
    evidence = [
        Evidence(
            evidence_id="graph-empty",
            source_type="graph",
            ticker="MSFT",
            structured_payload={
                "rows": [],
            },
        )
    ]

    (
        plan,
        calculations,
        citations,
    ) = _synthesizer_without_state_use().synthesize(
        question=("What was Microsoft's revenue in 2025?"),
        evidence=evidence,
    )

    assert plan.insufficient_evidence
    assert plan.cited_evidence_ids == []
    assert calculations == []
    assert citations == []
