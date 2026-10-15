"""Focused retry-classification regression tests."""

from fingraph.answer.synthesis import (
    _is_retryable_synthesis_validation_error,
)
from fingraph.evaluation.answers import (
    _is_retryable_judge_validation_error,
)
from fingraph.llm.groq_client import (
    LLMDeferredError,
    LLMQualityError,
    _is_json_validation_failure,
)


def test_json_schema_generation_failure_is_retryable_within_bounded_attempts() -> None:
    error = "Groq HTTP 400: {'error': {'code': 'json_validate_failed'}}"

    assert _is_json_validation_failure(error)


def test_company_citation_alignment_failure_is_retryable_within_bounded_attempts() -> None:
    error = (
        "ValueError: Grounded answer failed deterministic "
        "company/citation alignment: "
        "segment_1:company_claim_without_inline_evidence"
    )

    assert _is_retryable_synthesis_validation_error(error)


def test_answer_judge_parse_failure_is_retryable_within_bounded_attempts() -> None:
    error = "Groq HTTP 400: Parsing failed. code='output_parse_failed'; failed_generation='draft'"

    assert _is_retryable_judge_validation_error(error)


def test_unrelated_permanent_error_is_not_retryable() -> None:
    assert not _is_json_validation_failure("Groq HTTP 401: invalid API key")

    assert not _is_retryable_synthesis_validation_error(
        "Permission denied while opening local state database."
    )

    assert not _is_retryable_judge_validation_error("Groq HTTP 401: invalid API key")


def test_quality_and_deferred_failures_are_distinct_types() -> None:
    quality_error = LLMQualityError("grounding validation exhausted")
    deferred_error = LLMDeferredError("provider temporarily unavailable")

    assert not isinstance(
        quality_error,
        LLMDeferredError,
    )

    assert not isinstance(
        deferred_error,
        LLMQualityError,
    )
