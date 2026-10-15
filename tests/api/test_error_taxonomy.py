"""Regression coverage for stable public API failure taxonomy."""

import asyncio
import json
from typing import Any

import pytest
from fastapi import status

from fingraph.api.routes import (
    _agent_error_response,
    _sse_event,
)
from fingraph.api.service import (
    AgentBusyError,
    AgentDeferredError,
    AgentExecutionError,
    AgentPermanentError,
    AgentQualityError,
    AgentService,
    AgentTimeoutError,
)
from fingraph.llm.groq_client import LLMPermanentError


class PermanentFailureAgent:
    """Raise the production permanent-provider LLM signal."""

    def invoke(
        self,
        _: dict[str, Any],
    ) -> dict[str, Any]:
        raise LLMPermanentError("provider-secret-detail")


class UnusedSynthesizer:
    """Synthesizer that must never execute."""

    def synthesize(
        self,
        **_: Any,
    ) -> None:
        raise AssertionError("synthesizer should not run")


def test_llm_permanent_error_is_preserved_by_agent_service() -> None:
    async def run() -> None:
        service = AgentService(
            agent=PermanentFailureAgent(),
            synthesizer=UnusedSynthesizer(),  # type: ignore[arg-type]
            request_timeout_seconds=5.0,
            queue_timeout_seconds=1.0,
            max_concurrency=1,
        )

        with pytest.raises(
            AgentPermanentError,
            match="failed permanently",
        ) as captured:
            await service.ask(
                question="What was Microsoft revenue?",
                request_id="permanent-failure",
            )

        assert "provider-secret-detail" not in str(captured.value)

    asyncio.run(run())


@pytest.mark.parametrize(
    (
        "error",
        "expected_status",
        "expected_code",
        "expected_message",
    ),
    [
        (
            AgentDeferredError("provider-secret-detail"),
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "llm_temporarily_unavailable",
            "LLM provider temporarily unavailable. Retry later.",
        ),
        (
            AgentBusyError("private-capacity-detail"),
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "agent_busy",
            "Agent capacity is currently exhausted. Retry later.",
        ),
        (
            AgentTimeoutError("private-timeout-detail"),
            status.HTTP_504_GATEWAY_TIMEOUT,
            "agent_timeout",
            "Agent execution timed out.",
        ),
        (
            AgentQualityError("private-quality-detail"),
            status.HTTP_502_BAD_GATEWAY,
            "llm_quality_validation_failed",
            "LLM output failed deterministic quality validation.",
        ),
        (
            AgentPermanentError("private-provider-detail"),
            status.HTTP_502_BAD_GATEWAY,
            "llm_permanent_failure",
            "LLM provider request failed permanently.",
        ),
        (
            AgentExecutionError("private-execution-detail"),
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "agent_execution_failed",
            "Agent execution failed.",
        ),
    ],
)
def test_known_agent_failures_have_stable_safe_public_mapping(
    error: Exception,
    expected_status: int,
    expected_code: str,
    expected_message: str,
) -> None:
    status_code, response = _agent_error_response(
        request_id="request-123",
        error=error,
    )

    assert status_code == expected_status
    assert response.request_id == "request-123"
    assert response.error_code == expected_code
    assert response.message == expected_message

    serialized = response.model_dump_json()

    assert str(error) not in serialized


@pytest.mark.parametrize(
    (
        "error",
        "expected_code",
    ),
    [
        (
            AgentDeferredError("provider-secret-detail"),
            "llm_temporarily_unavailable",
        ),
        (
            AgentBusyError("private-capacity-detail"),
            "agent_busy",
        ),
        (
            AgentTimeoutError("private-timeout-detail"),
            "agent_timeout",
        ),
        (
            AgentQualityError("private-quality-detail"),
            "llm_quality_validation_failed",
        ),
        (
            AgentPermanentError("private-provider-detail"),
            "llm_permanent_failure",
        ),
        (
            AgentExecutionError("private-execution-detail"),
            "agent_execution_failed",
        ),
    ],
)
def test_sse_error_payload_uses_same_public_error_code(
    error: Exception,
    expected_code: str,
) -> None:
    _, response = _agent_error_response(
        request_id="request-456",
        error=error,
    )

    event = _sse_event(
        event="error",
        data=response.model_dump(mode="json"),
    )

    lines = event.strip().splitlines()

    assert lines[0] == "event: error"
    assert lines[1].startswith("data: ")

    payload = json.loads(lines[1].removeprefix("data: "))

    assert payload["request_id"] == "request-456"
    assert payload["error_code"] == expected_code
    assert str(error) not in lines[1]


def test_unknown_error_is_not_silently_assigned_known_taxonomy() -> None:
    with pytest.raises(
        TypeError,
        match="Unsupported agent error type",
    ):
        _agent_error_response(
            request_id="request-789",
            error=RuntimeError("unexpected"),
        )
