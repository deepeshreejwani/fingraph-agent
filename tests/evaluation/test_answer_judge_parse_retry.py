"""Regression tests for bounded answer-judge validation failures."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fingraph.evaluation.answers import (
    AnswerGoldenCase,
    GroqAnswerJudge,
    _is_retryable_judge_validation_error,
)
from fingraph.llm.groq_client import LLMDeferredError, LLMQualityError
from fingraph.llm.state import LLMStateStore


class InvalidJudgeProvider:
    """Return no structured judge content and count provider calls."""

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
    provider: InvalidJudgeProvider,
) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=provider,
        )
    )


def test_output_parse_failed_is_retryable_for_judge() -> None:
    error = "Groq HTTP 400: Parsing failed. code='output_parse_failed'; failed_generation='draft'"

    assert _is_retryable_judge_validation_error(error)


def test_unknown_client_error_is_not_retryable_for_judge() -> None:
    error = "Groq HTTP 400: invalid API request"

    assert not _is_retryable_judge_validation_error(error)


def test_exhausted_judge_validation_is_quality_failure_and_not_reexecuted(
    tmp_path: Path,
) -> None:
    store = LLMStateStore(tmp_path / "state.db")

    provider = InvalidJudgeProvider()

    judge = GroqAnswerJudge(
        api_key="test-key",
        model="openai/gpt-oss-120b",
        state_store=store,
        infrastructure_max_attempts=2,
        max_retry_wait_seconds=0.0,
    )

    judge.client = _fake_client(provider)  # type: ignore[assignment]

    case = AnswerGoldenCase(
        id="quality-failure",
        question="What risks did Microsoft describe?",
        expected_route="vector",
    )

    with pytest.raises(
        LLMQualityError,
        match="structured validation",
    ):
        judge.evaluate(
            case=case,
            answer="Microsoft described infrastructure risks.",
            insufficient_evidence=False,
            citations=[],
            calculations=[],
            evidence=[],
        )

    assert provider.calls == 2

    with pytest.raises(
        LLMQualityError,
        match="previously exhausted bounded structured-output validation",
    ):
        judge.evaluate(
            case=case,
            answer="Microsoft described infrastructure risks.",
            insufficient_evidence=False,
            citations=[],
            calculations=[],
            evidence=[],
        )

    assert provider.calls == 2


def test_running_judge_request_owned_by_another_worker_is_not_reexecuted(
    tmp_path: Path,
) -> None:
    store = LLMStateStore(tmp_path / "state.db")

    provider = InvalidJudgeProvider()

    judge = GroqAnswerJudge(
        api_key="test-key",
        model="openai/gpt-oss-120b",
        state_store=store,
        infrastructure_max_attempts=2,
        max_retry_wait_seconds=0.0,
    )

    judge.client = _fake_client(provider)  # type: ignore[assignment]

    case = AnswerGoldenCase(
        id="owned-elsewhere",
        question="What risks did Microsoft describe?",
        expected_route="vector",
    )

    request_id, input_hash, payload = judge._request_identity(
        case=case,
        answer="Microsoft described infrastructure risks.",
        insufficient_evidence=False,
        citations=[],
        calculations=[],
        evidence=[],
    )

    store.register(
        request_id=request_id,
        task_type="answer_evaluation_judge",
        provider="groq",
        model="openai/gpt-oss-120b",
        prompt_version="answer-judge-v1",
        input_hash=input_hash,
        request_payload=payload,
    )

    assert store.claim_execution(
        request_id=request_id,
        owner_id="other-worker",
        lease_seconds=120.0,
    )

    with pytest.raises(
        LLMDeferredError,
        match="already being executed by another worker",
    ):
        judge.evaluate(
            case=case,
            answer="Microsoft described infrastructure risks.",
            insufficient_evidence=False,
            citations=[],
            calculations=[],
            evidence=[],
        )

    assert provider.calls == 0
