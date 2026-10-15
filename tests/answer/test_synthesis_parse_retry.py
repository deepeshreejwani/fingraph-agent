"""Regression tests for bounded synthesis validation failures."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import fingraph.answer.synthesis as synthesis_module
from fingraph.agent import Evidence
from fingraph.answer.synthesis import (
    GroqAnswerSynthesizer,
    _is_retryable_synthesis_validation_error,
)
from fingraph.llm.groq_client import LLMQualityError
from fingraph.llm.state import LLMStateStore


class InvalidSynthesisProvider:
    """Return no structured content and count provider calls."""

    def __init__(self) -> None:
        self.calls = 0

    def create(
        self,
        **_: Any,
    ) -> SimpleNamespace:
        self.calls += 1

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                    )
                )
            ],
            usage=None,
        )


def _fake_client(
    provider: InvalidSynthesisProvider,
) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=provider,
        )
    )


def _evidence() -> list[Evidence]:
    return [
        Evidence(
            evidence_id="evidence-msft",
            source_type="vector",
            ticker="MSFT",
            filing_id="msft-filing",
            section_name="Item 1A. Risk Factors",
            text="Microsoft described infrastructure risks.",
            source_url="https://sec.example/msft",
        )
    ]


def test_output_parse_failed_is_retryable() -> None:
    error = "Groq HTTP 400: Parsing failed. code='output_parse_failed'; failed_generation='draft'"

    assert _is_retryable_synthesis_validation_error(error)


def test_unrelated_client_error_is_not_retryable() -> None:
    error = "Groq HTTP 400: invalid API request"

    assert not _is_retryable_synthesis_validation_error(error)


def test_exhausted_synthesis_validation_is_quality_failure_and_not_reexecuted(
    tmp_path: Path,
) -> None:
    store = LLMStateStore(tmp_path / "state.db")

    provider = InvalidSynthesisProvider()

    synthesizer = GroqAnswerSynthesizer(
        api_key="test-key",
        model="openai/gpt-oss-120b",
        state_store=store,
        infrastructure_max_attempts=2,
        max_retry_wait_seconds=0.0,
    )

    synthesizer.client = _fake_client(provider)  # type: ignore[assignment]

    with pytest.raises(
        LLMQualityError,
        match="deterministic validation",
    ):
        synthesizer.synthesize(
            question="What risks did Microsoft describe?",
            evidence=_evidence(),
        )

    assert provider.calls == 2

    with pytest.raises(
        LLMQualityError,
        match="previously exhausted bounded quality validation",
    ):
        synthesizer.synthesize(
            question="What risks did Microsoft describe?",
            evidence=_evidence(),
        )

    assert provider.calls == 2


def test_validation_version_change_reexecutes_previous_quality_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LLMStateStore(tmp_path / "state.db")

    provider = InvalidSynthesisProvider()

    synthesizer = GroqAnswerSynthesizer(
        api_key="test-key",
        model="openai/gpt-oss-120b",
        state_store=store,
        infrastructure_max_attempts=2,
        max_retry_wait_seconds=0.0,
    )

    synthesizer.client = _fake_client(provider)  # type: ignore[assignment]

    monkeypatch.setattr(
        synthesis_module,
        "SYNTHESIS_VALIDATION_VERSION",
        "grounded-answer-validation-test-v1",
    )

    with pytest.raises(
        LLMQualityError,
        match="deterministic validation",
    ):
        synthesizer.synthesize(
            question="What risks did Microsoft describe?",
            evidence=_evidence(),
        )

    assert provider.calls == 2

    with pytest.raises(
        LLMQualityError,
        match="previously exhausted bounded quality validation",
    ):
        synthesizer.synthesize(
            question="What risks did Microsoft describe?",
            evidence=_evidence(),
        )

    assert provider.calls == 2

    monkeypatch.setattr(
        synthesis_module,
        "SYNTHESIS_VALIDATION_VERSION",
        "grounded-answer-validation-test-v2",
    )

    with pytest.raises(
        LLMQualityError,
        match="deterministic validation",
    ):
        synthesizer.synthesize(
            question="What risks did Microsoft describe?",
            evidence=_evidence(),
        )

    assert provider.calls == 4
