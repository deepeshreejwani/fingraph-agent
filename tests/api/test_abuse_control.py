"""Regression tests for API abuse-control admission."""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from fingraph.api.abuse import (
    SlidingWindowRateLimiter,
    access_key_matches,
)
from fingraph.api.app import request_context
from fingraph.config import Settings


class FakeClock:
    """Deterministic monotonic clock for limiter tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(
        self,
        seconds: float,
    ) -> None:
        self.now += seconds


def _request(
    *,
    path: str,
    api_key: str | None = None,
    client_host: str = "203.0.113.10",
    settings: Settings | None = None,
    limiter: SlidingWindowRateLimiter | None = None,
) -> Request:
    """Create a minimal HTTP request for middleware tests."""

    headers: list[
        tuple[
            bytes,
            bytes,
        ]
    ] = []

    if api_key is not None:
        headers.append(
            (
                b"x-api-key",
                api_key.encode("utf-8"),
            )
        )

    app = SimpleNamespace(state=SimpleNamespace())

    if settings is not None:
        app.state.resources = SimpleNamespace(settings=settings)

    if limiter is not None:
        app.state.rate_limiter = limiter

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {
            "version": "3.0",
            "spec_version": "2.3",
        },
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": (path.encode("utf-8")),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": (
            client_host,
            54321,
        ),
        "server": (
            "testserver",
            80,
        ),
        "app": app,
    }

    return Request(scope)


def test_access_key_matches_when_authentication_disabled() -> None:
    assert access_key_matches(
        supplied_key=None,
        configured_key=None,
    )


def test_access_key_rejects_missing_or_incorrect_key() -> None:
    assert not access_key_matches(
        supplied_key=None,
        configured_key="expected",
    )

    assert not access_key_matches(
        supplied_key="wrong",
        configured_key="expected",
    )


def test_access_key_accepts_matching_key() -> None:
    assert access_key_matches(
        supplied_key="expected",
        configured_key="expected",
    )


def test_rate_limiter_enforces_fixed_request_budget() -> None:
    clock = FakeClock()

    limiter = SlidingWindowRateLimiter(
        requests_per_window=2,
        window_seconds=60.0,
        clock=clock,
    )

    first = limiter.check("client-a")

    second = limiter.check("client-a")

    third = limiter.check("client-a")

    assert first.allowed
    assert first.remaining == 1

    assert second.allowed
    assert second.remaining == 0

    assert not third.allowed
    assert third.remaining == 0
    assert third.retry_after_seconds == 60


def test_rate_limiter_is_per_client() -> None:
    limiter = SlidingWindowRateLimiter(
        requests_per_window=1,
    )

    first_client = limiter.check("client-a")

    second_client = limiter.check("client-b")

    assert first_client.allowed
    assert second_client.allowed


def test_rate_limiter_recovers_after_window_expires() -> None:
    clock = FakeClock()

    limiter = SlidingWindowRateLimiter(
        requests_per_window=1,
        window_seconds=60.0,
        clock=clock,
    )

    assert limiter.check("client-a").allowed

    assert not limiter.check("client-a").allowed

    clock.advance(60.001)

    renewed = limiter.check("client-a")

    assert renewed.allowed
    assert renewed.remaining == 0


def test_production_requires_api_access_key() -> None:
    with pytest.raises(
        ValidationError,
        match="FINGRAPH_API_ACCESS_KEY",
    ):
        Settings(
            environment="production",
            api_access_key=None,
            langfuse_enabled=False,
            _env_file=None,
        )


def test_production_accepts_configured_api_access_key() -> None:
    settings = Settings(
        environment="production",
        api_access_key="secret",
        langfuse_enabled=False,
        _env_file=None,
    )

    assert settings.api_access_key == "secret"


def test_public_endpoint_bypasses_inference_admission() -> None:
    async def run() -> None:
        request = _request(
            path="/health",
        )

        called = False

        async def call_next(
            _: Request,
        ) -> Response:
            nonlocal called

            called = True

            return Response(status_code=200)

        response = await request_context(
            request,
            call_next,
        )

        assert called
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers

    asyncio.run(run())


@pytest.mark.parametrize(
    (
        "include_resources",
        "include_rate_limiter",
    ),
    [
        (
            False,
            True,
        ),
        (
            True,
            False,
        ),
        (
            False,
            False,
        ),
    ],
)
def test_protected_endpoint_fails_closed_when_admission_state_is_unavailable(
    include_resources: bool,
    include_rate_limiter: bool,
) -> None:
    async def run() -> None:
        settings = Settings(
            environment="test",
            api_access_key="secret",
            langfuse_enabled=False,
            _env_file=None,
        )

        limiter = SlidingWindowRateLimiter(
            requests_per_window=10,
        )

        request = _request(
            path="/ask",
            api_key="secret",
            settings=(settings if include_resources else None),
            limiter=(limiter if include_rate_limiter else None),
        )

        called = False

        async def call_next(
            _: Request,
        ) -> Response:
            nonlocal called

            called = True

            return Response(status_code=200)

        response = await request_context(
            request,
            call_next,
        )

        assert not called

        assert response.status_code == 503

        payload = json.loads(bytes(response.body))

        assert payload["error_code"] == "admission_control_unavailable"

        assert payload["request_id"] == response.headers["X-Request-ID"]

    asyncio.run(run())


def test_protected_endpoint_rejects_missing_api_key() -> None:
    async def run() -> None:
        settings = Settings(
            environment="test",
            api_access_key="secret",
            langfuse_enabled=False,
            _env_file=None,
        )

        limiter = SlidingWindowRateLimiter(
            requests_per_window=10,
        )

        request = _request(
            path="/ask",
            settings=settings,
            limiter=limiter,
        )

        called = False

        async def call_next(
            _: Request,
        ) -> Response:
            nonlocal called

            called = True

            return Response(status_code=200)

        response = await request_context(
            request,
            call_next,
        )

        assert not called

        assert response.status_code == 401

        payload = json.loads(bytes(response.body))

        assert payload["error_code"] == "unauthorized"

        assert payload["request_id"] == response.headers["X-Request-ID"]

    asyncio.run(run())


def test_invalid_api_key_does_not_consume_rate_budget() -> None:
    async def run() -> None:
        settings = Settings(
            environment="test",
            api_access_key="secret",
            langfuse_enabled=False,
            _env_file=None,
        )

        limiter = SlidingWindowRateLimiter(
            requests_per_window=1,
        )

        invalid_request = _request(
            path="/ask",
            api_key="wrong",
            settings=settings,
            limiter=limiter,
        )

        async def call_next(
            _: Request,
        ) -> Response:
            return Response(status_code=200)

        invalid_response = await request_context(
            invalid_request,
            call_next,
        )

        assert invalid_response.status_code == 401

        valid_request = _request(
            path="/ask",
            api_key="secret",
            settings=settings,
            limiter=limiter,
        )

        valid_response = await request_context(
            valid_request,
            call_next,
        )

        assert valid_response.status_code == 200

        assert valid_response.headers["X-RateLimit-Remaining"] == "0"

    asyncio.run(run())


def test_protected_endpoint_returns_429_when_budget_exhausted() -> None:
    async def run() -> None:
        settings = Settings(
            environment="test",
            api_access_key="secret",
            langfuse_enabled=False,
            _env_file=None,
        )

        limiter = SlidingWindowRateLimiter(
            requests_per_window=1,
        )

        async def call_next(
            _: Request,
        ) -> Response:
            return JSONResponse(
                {
                    "ok": True,
                },
                status_code=200,
            )

        first_request = _request(
            path="/ask/stream",
            api_key="secret",
            settings=settings,
            limiter=limiter,
        )

        first_response = await request_context(
            first_request,
            call_next,
        )

        assert first_response.status_code == 200

        assert first_response.headers["X-RateLimit-Remaining"] == "0"

        second_request = _request(
            path="/ask/stream",
            api_key="secret",
            settings=settings,
            limiter=limiter,
        )

        second_response = await request_context(
            second_request,
            call_next,
        )

        assert second_response.status_code == 429

        assert second_response.headers["X-RateLimit-Remaining"] == "0"

        assert int(second_response.headers["Retry-After"]) >= 1

        payload = json.loads(bytes(second_response.body))

        assert payload["error_code"] == "rate_limit_exceeded"

    asyncio.run(run())
