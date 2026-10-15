"""FastAPI application factory."""

import logging
import time
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
)
from contextlib import (
    asynccontextmanager,
)
from uuid import uuid4

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.encoders import (
    jsonable_encoder,
)
from fastapi.exceptions import (
    RequestValidationError,
)
from fastapi.responses import (
    JSONResponse,
    Response,
)

from fingraph.api.abuse import (
    SlidingWindowRateLimiter,
    access_key_matches,
)
from fingraph.api.dependencies import (
    build_application_resources,
)
from fingraph.api.models import (
    ApiErrorResponse,
)
from fingraph.api.routes import (
    router,
)
from fingraph.config import (
    get_settings,
)
from fingraph.logging import (
    configure_logging,
)

logger = logging.getLogger(__name__)

PROTECTED_INFERENCE_PATHS = frozenset(
    {
        "/ask",
        "/ask/stream",
    }
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """Initialize and release application resources."""

    settings = get_settings()

    configure_logging(settings.log_level)

    logger.info(
        "application_starting",
        extra={
            "environment": (settings.environment),
        },
    )

    resources = build_application_resources(settings)

    app.state.resources = resources

    app.state.rate_limiter = SlidingWindowRateLimiter(
        requests_per_window=(settings.api_rate_limit_requests_per_minute),
        window_seconds=60.0,
    )

    logger.info(
        "application_started",
        extra={
            "environment": (settings.environment),
            "api_max_concurrency": (settings.api_max_concurrency),
            "api_rate_limit_requests_per_minute": (settings.api_rate_limit_requests_per_minute),
            "api_access_control_enabled": bool(settings.api_access_key),
        },
    )

    try:
        yield

    finally:
        logger.info("application_stopping")

        resources.close()

        logger.info("application_stopped")


app = FastAPI(
    title=("FinGraph Agent API"),
    description=("Production-oriented agentic financial intelligence API over SEC filings."),
    version="0.1.0",
    lifespan=lifespan,
)


def _request_id(
    request: Request,
) -> str:
    """Return the current request ID."""

    value = getattr(
        request.state,
        "request_id",
        None,
    )

    if value is None:
        return "unknown"

    return str(value)


def _client_id(
    request: Request,
) -> str:
    """Return the direct network client identity for rate limiting."""

    if request.client is None:
        return "unknown"

    host = request.client.host.strip()

    if not host:
        return "unknown"

    return host


def _api_error_response(
    *,
    request: Request,
    status_code: int,
    error_code: str,
    message: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build the stable public API error envelope."""

    payload = ApiErrorResponse(
        request_id=(_request_id(request)),
        error_code=error_code,
        message=message,
    )

    return JSONResponse(
        status_code=status_code,
        content=(payload.model_dump(mode="json")),
        headers=headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return stable validation failures."""

    payload = ApiErrorResponse(
        request_id=(_request_id(request)),
        error_code=("validation_error"),
        message=("Request validation failed."),
        details=(jsonable_encoder(exc.errors())),
    )

    return JSONResponse(
        status_code=422,
        content=(payload.model_dump(mode="json")),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """Return stable HTTP error responses."""

    payload = ApiErrorResponse(
        request_id=(_request_id(request)),
        error_code=(f"http_{exc.status_code}"),
        message=str(exc.detail),
    )

    return JSONResponse(
        status_code=(exc.status_code),
        content=(payload.model_dump(mode="json")),
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Return a non-leaking generic internal error."""

    logger.exception(
        "unhandled_api_exception",
        extra={
            "request_id": (_request_id(request)),
            "path": (request.url.path),
        },
    )

    payload = ApiErrorResponse(
        request_id=(_request_id(request)),
        error_code=("internal_server_error"),
        message=("Internal server error."),
    )

    return JSONResponse(
        status_code=500,
        content=(payload.model_dump(mode="json")),
    )


@app.middleware("http")
async def request_context(
    request: Request,
    call_next: Callable[
        [Request],
        Awaitable[Response],
    ],
) -> Response:
    """Attach request context and enforce inference admission controls."""

    supplied_request_id = request.headers.get("X-Request-ID")

    if supplied_request_id and len(supplied_request_id) <= 128:
        request_id = supplied_request_id

    else:
        request_id = str(uuid4())

    request.state.request_id = request_id

    started = time.perf_counter()

    logger.info(
        "http_request_started",
        extra={
            "request_id": (request_id),
            "method": (request.method),
            "path": (request.url.path),
        },
    )

    response: Response

    if request.url.path in PROTECTED_INFERENCE_PATHS:
        resources = getattr(
            request.app.state,
            "resources",
            None,
        )

        rate_limiter = getattr(
            request.app.state,
            "rate_limiter",
            None,
        )

        if resources is None or rate_limiter is None:
            logger.error(
                "api_admission_control_unavailable",
                extra={
                    "request_id": (request_id),
                    "path": (request.url.path),
                    "resources_available": (resources is not None),
                    "rate_limiter_available": (rate_limiter is not None),
                },
            )

            response = _api_error_response(
                request=request,
                status_code=503,
                error_code=("admission_control_unavailable"),
                message=("Inference admission control is unavailable."),
            )

            latency_ms = round(
                (time.perf_counter() - started) * 1000,
                3,
            )

            response.headers["X-Request-ID"] = request_id

            logger.info(
                "http_request_completed",
                extra={
                    "request_id": (request_id),
                    "method": (request.method),
                    "path": (request.url.path),
                    "status_code": (response.status_code),
                    "latency_ms": (latency_ms),
                },
            )

            return response

        settings = resources.settings

        supplied_key = request.headers.get("X-API-Key")

        if not access_key_matches(
            supplied_key=supplied_key,
            configured_key=(settings.api_access_key),
        ):
            logger.warning(
                "api_access_denied",
                extra={
                    "request_id": (request_id),
                    "path": (request.url.path),
                    "client_id": (_client_id(request)),
                },
            )

            response = _api_error_response(
                request=request,
                status_code=401,
                error_code=("unauthorized"),
                message=("Valid API access key required."),
            )

            latency_ms = round(
                (time.perf_counter() - started) * 1000,
                3,
            )

            response.headers["X-Request-ID"] = request_id

            logger.info(
                "http_request_completed",
                extra={
                    "request_id": (request_id),
                    "method": (request.method),
                    "path": (request.url.path),
                    "status_code": (response.status_code),
                    "latency_ms": (latency_ms),
                },
            )

            return response

        client_id = _client_id(request)

        decision = rate_limiter.check(client_id)

        if not decision.allowed:
            retry_after = decision.retry_after_seconds or 1

            logger.warning(
                "api_rate_limit_exceeded",
                extra={
                    "request_id": (request_id),
                    "path": (request.url.path),
                    "client_id": (client_id),
                    "retry_after_seconds": (retry_after),
                },
            )

            response = _api_error_response(
                request=request,
                status_code=429,
                error_code=("rate_limit_exceeded"),
                message=("Request rate limit exceeded. Retry later."),
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Remaining": ("0"),
                },
            )

            latency_ms = round(
                (time.perf_counter() - started) * 1000,
                3,
            )

            response.headers["X-Request-ID"] = request_id

            logger.info(
                "http_request_completed",
                extra={
                    "request_id": (request_id),
                    "method": (request.method),
                    "path": (request.url.path),
                    "status_code": (response.status_code),
                    "latency_ms": (latency_ms),
                },
            )

            return response

        request.state.rate_limit_remaining = decision.remaining

    try:
        response = await call_next(request)

    except Exception:
        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            3,
        )

        logger.exception(
            "http_request_failed",
            extra={
                "request_id": (request_id),
                "method": (request.method),
                "path": (request.url.path),
                "latency_ms": (latency_ms),
            },
        )

        raise

    latency_ms = round(
        (time.perf_counter() - started) * 1000,
        3,
    )

    response.headers["X-Request-ID"] = request_id

    remaining = getattr(
        request.state,
        "rate_limit_remaining",
        None,
    )

    if remaining is not None:
        response.headers["X-RateLimit-Remaining"] = str(remaining)

    logger.info(
        "http_request_completed",
        extra={
            "request_id": (request_id),
            "method": (request.method),
            "path": (request.url.path),
            "status_code": (response.status_code),
            "latency_ms": (latency_ms),
        },
    )

    return response


app.include_router(router)
