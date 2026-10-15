"""Formal route-specific Locust load benchmark for FinGraph."""

from __future__ import annotations

import json
import os
import time
from itertools import cycle
from typing import Any, ClassVar
from uuid import uuid4

from locust import HttpUser, constant, task

ROUTE_QUESTIONS: dict[str, tuple[str, ...]] = {
    "vector": (
        "What risks does NVIDIA describe in its 2026 10-K?",
        "Describe Meta's competition risks in its 2025 10-K.",
        "What business risks does Amazon describe in its 2025 10-K?",
    ),
    "graph": (
        "What was Microsoft revenue in 2026?",
        "What was NVIDIA net income in 2026?",
        "What was Amazon revenue in 2025?",
    ),
    "hybrid": (
        (
            "What was Microsoft revenue in 2026 and explain "
            "the business risks Microsoft described in its 2026 10-K?"
        ),
        (
            "What was NVIDIA net income in 2026 and explain "
            "the risks NVIDIA described in its 2026 10-K?"
        ),
        (
            "What was Amazon revenue in 2025 and explain "
            "the business risks Amazon described in its 2025 10-K?"
        ),
    ),
}

VALID_ROUTES = frozenset(ROUTE_QUESTIONS)

VALID_MODES = frozenset(
    {
        "sync",
        "stream",
    }
)


def _configuration() -> tuple[str, str]:
    """Return validated benchmark route and execution mode."""

    route = (
        os.getenv(
            "FINGRAPH_LOAD_ROUTE",
            "vector",
        )
        .strip()
        .lower()
    )

    mode = (
        os.getenv(
            "FINGRAPH_LOAD_MODE",
            "sync",
        )
        .strip()
        .lower()
    )

    if route not in VALID_ROUTES:
        raise RuntimeError("FINGRAPH_LOAD_ROUTE must be one of: " + ", ".join(sorted(VALID_ROUTES)))

    if mode not in VALID_MODES:
        raise RuntimeError("FINGRAPH_LOAD_MODE must be one of: " + ", ".join(sorted(VALID_MODES)))

    return route, mode


LOAD_ROUTE, LOAD_MODE = _configuration()


def _parse_sse_data(
    raw: str,
) -> dict[str, Any]:
    """Parse one SSE JSON data payload."""

    parsed = json.loads(raw)

    if not isinstance(
        parsed,
        dict,
    ):
        raise ValueError("SSE data payload must be a JSON object.")

    return parsed


class FinGraphLoadUser(HttpUser):
    """Exercise one configured FinGraph route."""

    wait_time = constant(0)

    host = os.getenv(
        "FINGRAPH_LOAD_HOST",
        "http://localhost:8000",
    )

    route: ClassVar[str] = LOAD_ROUTE

    mode: ClassVar[str] = LOAD_MODE

    questions: ClassVar[tuple[str, ...]] = ROUTE_QUESTIONS[LOAD_ROUTE]

    def on_start(
        self,
    ) -> None:
        """Initialize deterministic per-user question rotation."""

        self._question_cycle = cycle(self.questions)

    def _request_headers(
        self,
    ) -> dict[str, str]:
        """Build one unique request ID."""

        return {"X-Request-ID": (f"locust-{self.route}-{self.mode}-{uuid4()}")}

    def _validate_route(
        self,
        payload: dict[
            str,
            Any,
        ],
    ) -> str | None:
        """Return an error when the API selected an unexpected route."""

        actual_route = payload.get("route")

        if actual_route != self.route:
            return f"Route mismatch: expected={self.route}, actual={actual_route!r}"

        return None

    def _fire_ttft_metric(
        self,
        *,
        response_time_ms: float,
        exception: Exception | None,
    ) -> None:
        """Record client-visible time to first validated answer text."""

        self.environment.events.request.fire(
            request_type="TTFT",
            name=(f"{self.route.upper()} first-answer"),
            response_time=response_time_ms,
            response_length=0,
            exception=exception,
            context={
                "route": self.route,
                "semantics": ("time_to_first_client_visible_validated_answer_text"),
                "buffered_structured_output": True,
            },
            response=None,
        )

    def _fire_stream_completed_metric(
        self,
        *,
        response_time_ms: float,
        exception: Exception | None,
    ) -> None:
        """Record end-to-end SSE completion latency."""

        self.environment.events.request.fire(
            request_type="STREAM",
            name=(f"{self.route.upper()} completed"),
            response_time=response_time_ms,
            response_length=0,
            exception=exception,
            context={
                "route": self.route,
                "semantics": ("request_start_to_validated_sse_completed_event"),
            },
            response=None,
        )

    def _fire_stream_failure_metrics(
        self,
        *,
        started: float,
        message: str,
        ttft_ms: float | None = None,
    ) -> None:
        """Record consistent failed stream and TTFT metrics."""

        elapsed_ms = (time.perf_counter() - started) * 1000

        effective_ttft_ms = ttft_ms if ttft_ms is not None else elapsed_ms

        error = RuntimeError(message)

        self._fire_ttft_metric(
            response_time_ms=(effective_ttft_ms),
            exception=error,
        )

        self._fire_stream_completed_metric(
            response_time_ms=(elapsed_ms),
            exception=error,
        )

    def _run_sync(
        self,
        question: str,
    ) -> None:
        """Benchmark the standard completed-response endpoint."""

        with self.client.post(
            "/ask",
            json={
                "question": question,
            },
            headers=(self._request_headers()),
            name=(f"{self.route.upper()} /ask"),
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected status {response.status_code}: {response.text[:500]}")

                return

            try:
                payload = response.json()

            except ValueError as exc:
                response.failure(f"Invalid JSON response: {exc}")

                return

            if not isinstance(
                payload,
                dict,
            ):
                response.failure("Response payload is not a JSON object.")

                return

            route_error = self._validate_route(payload)

            if route_error is not None:
                response.failure(route_error)

                return

            response.success()

    def _run_stream(
        self,
        question: str,
    ) -> None:
        """Benchmark SSE through the validated completed event."""

        started = time.perf_counter()

        ttft_ms: float | None = None

        completed_payload: dict[str, Any] | None = None

        stream_error: str | None = None

        with self.client.post(
            "/ask/stream",
            json={
                "question": question,
            },
            headers=(self._request_headers()),
            name=(f"{self.route.upper()} /ask/stream-open"),
            stream=True,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                message = f"Unexpected status {response.status_code}"

                response.failure(message)

                self._fire_stream_failure_metrics(
                    started=started,
                    message=message,
                )

                return

            current_event: str | None = None

            try:
                for raw_line in response.iter_lines(decode_unicode=True):
                    if raw_line is None:
                        continue

                    line = raw_line.strip()

                    if line.startswith("event:"):
                        current_event = line[len("event:") :].strip()

                        continue

                    if not line.startswith("data:"):
                        continue

                    raw_data = line[len("data:") :].strip()

                    data = _parse_sse_data(raw_data)

                    if current_event == "error":
                        stream_error = str(
                            data.get(
                                "message",
                                "Unknown SSE error.",
                            )
                        )

                        break

                    if current_event != "completed":
                        continue

                    completed_payload = data

                    ttft_ms = (time.perf_counter() - started) * 1000

                    break

            except (
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                stream_error = f"{type(exc).__name__}: {exc}"

            if stream_error is not None:
                response.failure(stream_error)

                self._fire_stream_failure_metrics(
                    started=started,
                    message=stream_error,
                )

                return

            if completed_payload is None or ttft_ms is None:
                message = "SSE stream ended without a completed answer event."

                response.failure(message)

                self._fire_stream_failure_metrics(
                    started=started,
                    message=message,
                )

                return

            route_error = self._validate_route(completed_payload)

            if route_error is not None:
                response.failure(route_error)

                self._fire_stream_failure_metrics(
                    started=started,
                    message=route_error,
                    ttft_ms=ttft_ms,
                )

                return

            answer = completed_payload.get("answer")

            if (
                not isinstance(
                    answer,
                    str,
                )
                or not answer.strip()
            ):
                message = "Completed SSE response contains no answer text."

                response.failure(message)

                self._fire_stream_failure_metrics(
                    started=started,
                    message=message,
                    ttft_ms=ttft_ms,
                )

                return

            completed_ms = (time.perf_counter() - started) * 1000

            response.success()

            self._fire_ttft_metric(
                response_time_ms=ttft_ms,
                exception=None,
            )

            self._fire_stream_completed_metric(
                response_time_ms=(completed_ms),
                exception=None,
            )

    @task
    def ask(
        self,
    ) -> None:
        """Execute the configured benchmark workload."""

        question = next(self._question_cycle)

        if self.mode == "sync":
            self._run_sync(question)

            return

        self._run_stream(question)
