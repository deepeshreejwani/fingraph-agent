"""Insufficient-answer normalization tests."""

from fingraph.answer.models import (
    SynthesisPlan,
)


def test_insufficient_answer_drops_citations() -> None:
    plan = SynthesisPlan(
        answer=("The available evidence does not support this request."),
        cited_evidence_ids=["empty-graph-evidence"],
        insufficient_evidence=True,
    )

    assert plan.cited_evidence_ids == []


def test_supported_answer_preserves_citations() -> None:
    plan = SynthesisPlan(
        answer="Supported answer.",
        cited_evidence_ids=["real-evidence"],
        insufficient_evidence=False,
    )

    assert plan.cited_evidence_ids == ["real-evidence"]
