"""Groq quota classification helpers for resumable evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import groq

QuotaScope = Literal[
    "daily_tokens",
    "minute_tokens",
    "requests",
    "unknown",
]


@dataclass(frozen=True)
class GroqQuotaSignal:
    """Classified Groq rate-limit signal."""

    scope: QuotaScope
    retry_after_seconds: float | None
    raw_error: str


_RETRY_MESSAGE_PATTERN = re.compile(
    r"please\s+try\s+again\s+in\s+"
    r"(?:(?P<hours>\d+)h)?"
    r"(?:(?P<minutes>\d+)m)?"
    r"(?P<seconds>[\d.]+)s",
    re.IGNORECASE,
)


def _retry_after_from_text(
    text: str,
) -> float | None:
    """Parse Groq's human-readable retry interval."""

    match = _RETRY_MESSAGE_PATTERN.search(text)

    if match is None:
        return None

    hours = int(match.group("hours") or 0)

    minutes = int(match.group("minutes") or 0)

    seconds = float(match.group("seconds"))

    return (hours * 3600) + (minutes * 60) + seconds


def classify_groq_rate_limit_text(
    text: str,
    *,
    retry_after_seconds: float | None = None,
) -> GroqQuotaSignal:
    """Classify a Groq rate-limit message without depending on SDK internals."""

    normalized = text.casefold()

    if any(
        marker in normalized
        for marker in (
            "tokens per day",
            "token per day",
            "tokens/day",
            " tpd",
            "(tpd",
        )
    ):
        scope: QuotaScope = "daily_tokens"

    elif any(
        marker in normalized
        for marker in (
            "tokens per minute",
            "token per minute",
            "tokens/minute",
            " tpm",
            "(tpm",
        )
    ):
        scope = "minute_tokens"

    elif any(
        marker in normalized
        for marker in (
            "requests per minute",
            "request per minute",
            "requests/minute",
            " rpm",
            "(rpm",
            "requests per day",
            "request per day",
            " rpd",
            "(rpd",
        )
    ):
        scope = "requests"

    # Defensive fallback for the currently observed Groq free/on-demand
    # openai/gpt-oss-120b limits. Descriptive markers above are preferred.
    elif (
        "rate limit" in normalized
        and re.search(r"\blimit\s+200000\b", normalized)
        and "requested" in normalized
    ):
        scope = "daily_tokens"

    elif (
        "rate limit" in normalized
        and re.search(r"\blimit\s+8000\b", normalized)
        and "requested" in normalized
    ):
        scope = "minute_tokens"

    else:
        scope = "unknown"

    parsed_retry = (
        retry_after_seconds if retry_after_seconds is not None else _retry_after_from_text(text)
    )

    return GroqQuotaSignal(
        scope=scope,
        retry_after_seconds=parsed_retry,
        raw_error=text,
    )


def classify_groq_deferred_error(
    error: BaseException,
) -> GroqQuotaSignal | None:
    """
    Find and classify the Groq RateLimitError beneath a resumable exception.

    FinGraph raises LLMDeferredError with the provider exception as its cause.
    Walking the exception chain preserves provider detail without changing the
    shared deferred-error public API.
    """

    current: BaseException | None = error

    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))

        if isinstance(current, groq.RateLimitError):
            response = getattr(
                current,
                "response",
                None,
            )

            header_retry: float | None = None

            if response is not None:
                raw_retry = response.headers.get("retry-after")

                if raw_retry is not None:
                    try:
                        header_retry = float(raw_retry)

                    except ValueError:
                        header_retry = None

            return classify_groq_rate_limit_text(
                str(current),
                retry_after_seconds=header_retry,
            )

        cause = current.__cause__

        if cause is not None:
            current = cause
            continue

        current = current.__context__

    return None
