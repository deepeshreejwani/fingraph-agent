"""HTTP routes for FinGraph."""

import asyncio
import json
import logging
from collections.abc import (
    AsyncIterator,
)
from contextlib import suppress

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
)

from fingraph import __version__
from fingraph.api.dependencies import (
    ApplicationResources,
)
from fingraph.api.models import (
    ApiErrorResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
    ReadinessResponse,
)
from fingraph.api.service import (
    AgentBusyError,
    AgentDeferredError,
    AgentExecutionError,
    AgentPermanentError,
    AgentQualityError,
    AgentTimeoutError,
)

logger = logging.getLogger(__name__)


router = APIRouter()


def _resources(
    request: Request,
) -> ApplicationResources:
    """Return initialized application resources."""

    resources = getattr(
        request.app.state,
        "resources",
        None,
    )

    if not isinstance(
        resources,
        ApplicationResources,
    ):
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=("Application resources are not initialized."),
        )

    return resources


def _request_id(
    request: Request,
) -> str:
    """Return middleware-assigned request ID."""

    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    if request_id is None:
        raise RuntimeError("Request ID middleware was not executed.")

    return str(request_id)


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
)
async def health() -> HealthResponse:
    """Liveness endpoint."""

    return HealthResponse(
        status="ok",
        service="fingraph-agent",
        version=__version__,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["system"],
)
async def ready(
    request: Request,
    response: Response,
) -> ReadinessResponse:
    """Dependency-readiness endpoint."""

    resources = _resources(request)

    checks = await resources.readiness()

    ready_state = all(checks.values())

    if not ready_state:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        ready=ready_state,
        checks=checks,
    )


def _agent_error_response(
    *,
    request_id: str,
    error: Exception,
) -> tuple[int, ApiErrorResponse]:
    """Map one known agent failure to a stable safe public API error."""

    if isinstance(error, AgentDeferredError):
        return (
            status.HTTP_503_SERVICE_UNAVAILABLE,
            ApiErrorResponse(
                request_id=request_id,
                error_code="llm_temporarily_unavailable",
                message="LLM provider temporarily unavailable. Retry later.",
            ),
        )

    if isinstance(error, AgentBusyError):
        return (
            status.HTTP_503_SERVICE_UNAVAILABLE,
            ApiErrorResponse(
                request_id=request_id,
                error_code="agent_busy",
                message="Agent capacity is currently exhausted. Retry later.",
            ),
        )

    if isinstance(error, AgentTimeoutError):
        return (
            status.HTTP_504_GATEWAY_TIMEOUT,
            ApiErrorResponse(
                request_id=request_id,
                error_code="agent_timeout",
                message="Agent execution timed out.",
            ),
        )

    if isinstance(error, AgentQualityError):
        return (
            status.HTTP_502_BAD_GATEWAY,
            ApiErrorResponse(
                request_id=request_id,
                error_code="llm_quality_validation_failed",
                message="LLM output failed deterministic quality validation.",
            ),
        )

    if isinstance(error, AgentPermanentError):
        return (
            status.HTTP_502_BAD_GATEWAY,
            ApiErrorResponse(
                request_id=request_id,
                error_code="llm_permanent_failure",
                message="LLM provider request failed permanently.",
            ),
        )

    if isinstance(error, AgentExecutionError):
        return (
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            ApiErrorResponse(
                request_id=request_id,
                error_code="agent_execution_failed",
                message="Agent execution failed.",
            ),
        )

    raise TypeError(f"Unsupported agent error type: {type(error).__name__}")


@router.post(
    "/ask",
    response_model=AskResponse,
    tags=["agent"],
)
async def ask(
    payload: AskRequest,
    request: Request,
) -> AskResponse | JSONResponse:
    """Execute one FinGraph question."""

    resources = _resources(request)

    request_id = _request_id(request)

    try:
        return await resources.agent_service.ask(
            question=payload.question,
            request_id=request_id,
        )

    except (
        AgentDeferredError,
        AgentBusyError,
        AgentTimeoutError,
        AgentQualityError,
        AgentPermanentError,
        AgentExecutionError,
    ) as exc:
        status_code, error = _agent_error_response(
            request_id=request_id,
            error=exc,
        )

        return JSONResponse(
            status_code=status_code,
            content=error.model_dump(mode="json"),
        )


def _sse_event(
    *,
    event: str,
    data: dict[
        str,
        object,
    ],
) -> str:
    """Serialize one Server-Sent Event."""

    payload = json.dumps(
        data,
        default=str,
        separators=(
            ",",
            ":",
        ),
    )

    return f"event: {event}\ndata: {payload}\n\n"


@router.post(
    "/ask/stream",
    tags=["agent"],
)
async def ask_stream(
    payload: AskRequest,
    request: Request,
) -> StreamingResponse:
    """Stream execution lifecycle events over SSE."""

    resources = _resources(request)

    request_id = _request_id(request)

    async def events() -> AsyncIterator[str]:
        yield _sse_event(
            event="accepted",
            data={
                "request_id": (request_id),
            },
        )

        agent_task = asyncio.create_task(
            resources.agent_service.ask(
                question=(payload.question),
                request_id=(request_id),
            )
        )

        heartbeat_interval = 15.0

        try:
            while not agent_task.done():
                if await request.is_disconnected():
                    logger.info(
                        "sse_client_disconnected",
                        extra={
                            "request_id": (request_id),
                        },
                    )

                    agent_task.cancel()

                    with suppress(asyncio.CancelledError):
                        await agent_task

                    return

                try:
                    await asyncio.wait_for(
                        asyncio.shield(agent_task),
                        timeout=(heartbeat_interval),
                    )

                except TimeoutError:
                    yield _sse_event(
                        event="heartbeat",
                        data={
                            "request_id": (request_id),
                        },
                    )

            response = agent_task.result()

            yield _sse_event(
                event="completed",
                data=(response.model_dump(mode="json")),
            )

        except (
            AgentDeferredError,
            AgentBusyError,
            AgentTimeoutError,
            AgentQualityError,
            AgentPermanentError,
            AgentExecutionError,
        ) as exc:
            if isinstance(exc, AgentExecutionError):
                logger.exception(
                    "sse_agent_execution_failed",
                    extra={
                        "request_id": request_id,
                    },
                )

            _, error = _agent_error_response(
                request_id=request_id,
                error=exc,
            )

            yield _sse_event(
                event="error",
                data=error.model_dump(mode="json"),
            )

        except asyncio.CancelledError:
            if not agent_task.done():
                agent_task.cancel()

            raise

        except Exception:
            logger.exception(
                "sse_unhandled_error",
                extra={
                    "request_id": (request_id),
                },
            )

            yield _sse_event(
                event="error",
                data={
                    "request_id": (request_id),
                    "type": ("internal_error"),
                    "message": ("Internal server error."),
                },
            )

    return StreamingResponse(
        events(),
        media_type=("text/event-stream"),
        headers={
            "Cache-Control": ("no-cache"),
            "X-Accel-Buffering": ("no"),
            "Connection": ("keep-alive"),
        },
    )
