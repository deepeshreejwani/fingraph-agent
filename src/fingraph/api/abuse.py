"""Lightweight API access and request-budget controls."""

from __future__ import annotations

import hmac
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class RateLimitDecision:
    """Result of one request-budget admission check."""

    allowed: bool

    remaining: int

    retry_after_seconds: int | None = None


def access_key_matches(
    *,
    supplied_key: str | None,
    configured_key: str | None,
) -> bool:
    """Verify an API access key without leaking comparison timing."""

    if configured_key is None:
        return True

    if supplied_key is None:
        return False

    return hmac.compare_digest(
        supplied_key,
        configured_key,
    )


class SlidingWindowRateLimiter:
    """Process-local per-client sliding-window request limiter."""

    def __init__(
        self,
        *,
        requests_per_window: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        max_tracked_clients: int = 10_000,
    ) -> None:
        if requests_per_window <= 0:
            raise ValueError("requests_per_window must be positive.")

        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")

        if max_tracked_clients <= 0:
            raise ValueError("max_tracked_clients must be positive.")

        self._requests_per_window = requests_per_window

        self._window_seconds = window_seconds

        self._clock = clock

        self._max_tracked_clients = max_tracked_clients

        self._requests: dict[
            str,
            deque[float],
        ] = {}

        self._last_seen: dict[
            str,
            float,
        ] = {}

        self._lock = Lock()

    def check(
        self,
        client_id: str,
    ) -> RateLimitDecision:
        """Consume one request slot when capacity is available."""

        normalized_client_id = client_id.strip()

        if not normalized_client_id:
            normalized_client_id = "unknown"

        now = self._clock()

        cutoff = now - self._window_seconds

        with self._lock:
            self._prune_stale_clients(cutoff=cutoff)

            requests = self._requests.setdefault(
                normalized_client_id,
                deque(),
            )

            while requests and requests[0] <= cutoff:
                requests.popleft()

            self._last_seen[normalized_client_id] = now

            if len(requests) >= self._requests_per_window:
                oldest = requests[0]

                retry_after = max(
                    1,
                    math.ceil((oldest + self._window_seconds) - now),
                )

                return RateLimitDecision(
                    allowed=False,
                    remaining=0,
                    retry_after_seconds=(retry_after),
                )

            requests.append(now)

            remaining = max(
                0,
                self._requests_per_window - len(requests),
            )

            return RateLimitDecision(
                allowed=True,
                remaining=remaining,
            )

    def _prune_stale_clients(
        self,
        *,
        cutoff: float,
    ) -> None:
        """Discard expired clients and bound limiter bookkeeping."""

        stale_clients = [
            client_id for client_id, last_seen in self._last_seen.items() if last_seen <= cutoff
        ]

        for client_id in stale_clients:
            self._last_seen.pop(
                client_id,
                None,
            )

            self._requests.pop(
                client_id,
                None,
            )

        overflow = len(self._requests) - self._max_tracked_clients

        if overflow <= 0:
            return

        oldest_clients = sorted(
            self._last_seen,
            key=self._last_seen.__getitem__,
        )[:overflow]

        for client_id in oldest_clients:
            self._last_seen.pop(
                client_id,
                None,
            )

            self._requests.pop(
                client_id,
                None,
            )
