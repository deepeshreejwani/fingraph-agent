"""Focused tests for Groq quota classification."""

from fingraph.evaluation.quota import (
    classify_groq_rate_limit_text,
)


def test_daily_token_quota_is_detected() -> None:
    signal = classify_groq_rate_limit_text(
        "Groq rate limit: tokens per day (TPD): "
        "Limit 200000, Used 197840, Requested 4675. "
        "Please try again in 18m6.48s."
    )

    assert signal.scope == "daily_tokens"

    assert signal.retry_after_seconds == 1086.48


def test_minute_token_quota_is_detected() -> None:
    signal = classify_groq_rate_limit_text(
        "Groq rate limit: tokens per minute (TPM): "
        "Limit 8000, Used 7550, Requested 6056. "
        "Please try again in 42.5s."
    )

    assert signal.scope == "minute_tokens"

    assert signal.retry_after_seconds == 42.5


def test_daily_limit_fallback_matches_observed_provider_shape() -> None:
    signal = classify_groq_rate_limit_text(
        "Groq rate limit: Limit 200000, Used 199200, Requested 4895."
    )

    assert signal.scope == "daily_tokens"


def test_unknown_rate_limit_does_not_claim_daily_quota() -> None:
    signal = classify_groq_rate_limit_text("Groq rate limit: capacity temporarily unavailable.")

    assert signal.scope == "unknown"

    assert signal.retry_after_seconds is None


def test_explicit_retry_after_takes_precedence() -> None:
    signal = classify_groq_rate_limit_text(
        ("tokens per day; Please try again in 10m0s."),
        retry_after_seconds=12.25,
    )

    assert signal.scope == "daily_tokens"

    assert signal.retry_after_seconds == 12.25
