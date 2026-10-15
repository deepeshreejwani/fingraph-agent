"""HTTP client for public SEC EDGAR resources."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from threading import Lock
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SecClientError(RuntimeError):
    """Raised when an SEC request cannot be completed."""


class SecClient:
    """Thread-safe, globally rate-limited HTTP client for SEC public data."""

    def __init__(
        self,
        *,
        user_agent: str,
        requests_per_second: float = 5.0,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        max_retry_wait_seconds: float = 60.0,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("SEC User-Agent must not be empty.")

        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive.")

        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        if max_retry_wait_seconds < 0:
            raise ValueError("max_retry_wait_seconds cannot be negative.")

        self._minimum_interval = 1.0 / requests_per_second

        self._max_retries = max_retries

        self._max_retry_wait_seconds = max_retry_wait_seconds

        self._last_request_at = 0.0

        self._rate_limit_lock = Lock()

        self._client = httpx.Client(
            headers={
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "*/*",
            },
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=True,
        )

    def __enter__(
        self,
    ) -> SecClient:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()

    def close(
        self,
    ) -> None:
        """Close network resources."""

        self._client.close()

    def get_json(
        self,
        url: str,
    ) -> Mapping[
        str,
        Any,
    ]:
        """Fetch and decode a JSON object."""

        response = self._request(url)

        payload = response.json()

        if not isinstance(
            payload,
            dict,
        ):
            raise SecClientError(f"Expected JSON object from {url}")

        return payload

    def get_text(
        self,
        url: str,
    ) -> str:
        """Fetch text content."""

        return self._request(url).text

    def _request(
        self,
        url: str,
    ) -> httpx.Response:
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            self._respect_rate_limit()

            try:
                response = self._client.get(url)

                if response.status_code == 429:
                    raise httpx.HTTPStatusError(
                        "SEC rate limit response",
                        request=response.request,
                        response=response,
                    )

                response.raise_for_status()

                return response

            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.HTTPStatusError,
            ) as exc:
                last_error = exc

                status = (
                    exc.response.status_code
                    if isinstance(
                        exc,
                        httpx.HTTPStatusError,
                    )
                    else None
                )

                retryable = (
                    status is None or status == 429 or (status is not None and 500 <= status < 600)
                )

                if not retryable or attempt >= self._max_retries:
                    break

                delay = self._retry_delay_seconds(
                    error=exc,
                    attempt=attempt,
                )

                logger.warning(
                    "SEC request failed; retrying",
                    extra={
                        "url": url,
                        "attempt": attempt + 1,
                        "status_code": status,
                        "retry_delay_seconds": round(
                            delay,
                            3,
                        ),
                    },
                )

                time.sleep(delay)

        raise SecClientError(f"SEC request failed after retries: {url}") from last_error

    def _respect_rate_limit(
        self,
    ) -> None:
        """
        Serialize request starts so the configured SEC rate is global.

        Multiple company workers may share this client, but the aggregate
        process-level request rate remains bounded by requests_per_second.
        """

        with self._rate_limit_lock:
            now = time.monotonic()

            elapsed = now - self._last_request_at

            remaining = self._minimum_interval - elapsed

            if remaining > 0:
                time.sleep(remaining)

            self._last_request_at = time.monotonic()

    def _retry_delay_seconds(
        self,
        *,
        error: Exception,
        attempt: int,
    ) -> float:
        """Prefer provider Retry-After when available."""

        retry_after = self._retry_after_seconds(error)

        if retry_after is None:
            retry_after = self._backoff_seconds(attempt)

        return min(
            retry_after,
            self._max_retry_wait_seconds,
        )

    @staticmethod
    def _retry_after_seconds(
        error: Exception,
    ) -> float | None:
        """Parse numeric or HTTP-date Retry-After."""

        if not isinstance(
            error,
            httpx.HTTPStatusError,
        ):
            return None

        raw = error.response.headers.get("Retry-After")

        if raw is None:
            return None

        try:
            seconds = float(raw)

        except ValueError:
            try:
                retry_at = parsedate_to_datetime(raw)

                date_header = error.response.headers.get("Date")

                if date_header is None:
                    return None

                response_at = parsedate_to_datetime(date_header)

                seconds = (retry_at - response_at).total_seconds()

            except (
                TypeError,
                ValueError,
                OverflowError,
            ):
                return None

        return max(
            seconds,
            0.0,
        )

    @staticmethod
    def _backoff_seconds(
        attempt: int,
    ) -> float:
        base = min(
            2**attempt,
            8,
        )

        return float(
            base
            + random.uniform(
                0.0,
                0.25,
            )
        )
