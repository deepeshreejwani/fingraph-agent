"""Tests for SEC client concurrency and rate limiting."""

from __future__ import annotations

import threading
import time

from pytest import MonkeyPatch

from fingraph.ingestion.sec.client import SecClient


class FakeClock:
    """Thread-safe deterministic monotonic clock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.now = 100.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        with self._lock:
            return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self.now += seconds


def test_rate_limit_is_shared_across_threads(
    monkeypatch: MonkeyPatch,
) -> None:
    """Concurrent callers must share one process-level SEC throttle."""

    clock = FakeClock()

    monkeypatch.setattr(
        time,
        "monotonic",
        clock.monotonic,
    )

    monkeypatch.setattr(
        time,
        "sleep",
        clock.sleep,
    )

    client = SecClient(
        user_agent="fingraph-test test@example.com",
        requests_per_second=5.0,
    )

    barrier = threading.Barrier(3)

    def worker() -> None:
        barrier.wait()
        client._respect_rate_limit()

    first = threading.Thread(target=worker)
    second = threading.Thread(target=worker)

    first.start()
    second.start()

    barrier.wait()

    first.join()
    second.join()

    client.close()

    assert len(clock.sleeps) == 1
    assert clock.sleeps[0] == 0.2
