"""Concurrency-safe asynchronous wrapper around the FinGraph agent."""

import asyncio
import logging
import time
from contextlib import nullcontext
from typing import Any

from langfuse import Langfuse, propagate_attributes

from fingraph.agent import Evidence
from fingraph.answer import GroqAnswerSynthesizer
from fingraph.api.models import AskResponse
from fingraph.llm.groq_client import (
    LLMDeferredError,
    LLMPermanentError,
    LLMQualityError,
)

logger = logging.getLogger(__name__)


class AgentTimeoutError(RuntimeError):
    """Agent execution exceeded the configured timeout."""


class AgentBusyError(RuntimeError):
    """Agent execution could not acquire capacity in time."""


class AgentDeferredError(RuntimeError):
    """Agent execution was deferred because an upstream LLM is unavailable."""


class AgentQualityError(RuntimeError):
    """Agent execution exhausted bounded LLM output-quality validation."""


class AgentPermanentError(RuntimeError):
    """Agent execution failed because an upstream LLM request failed permanently."""


class AgentExecutionError(RuntimeError):
    """Agent execution failed unexpectedly."""


class AgentService:
    """Execute the FinGraph evidence and answer pipeline safely."""

    def __init__(
        self,
        *,
        agent: Any,
        synthesizer: GroqAnswerSynthesizer,
        request_timeout_seconds: float,
        queue_timeout_seconds: float,
        max_concurrency: int,
        langfuse: Langfuse | None = None,
    ) -> None:
        self._agent = agent
        self._synthesizer = synthesizer
        self._request_timeout_seconds = request_timeout_seconds
        self._queue_timeout_seconds = queue_timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._langfuse = langfuse

    @staticmethod
    def _initial_state(
        question: str,
    ) -> dict[str, Any]:
        """Create a clean workflow state."""

        return {
            "question": question,
            "route": None,
            "route_reason": None,
            "vector_evidence": [],
            "graph_evidence": [],
            "evidence": [],
            "errors": [],
        }

    @staticmethod
    def _round_evidence(
        evidence: list[Evidence],
    ) -> list[Evidence]:
        """Round displayed scores only."""

        rounded: list[Evidence] = []

        for item in evidence:
            score = item.score

            rounded.append(
                item.model_copy(update={"score": (round(score, 3) if score is not None else None)})
            )

        return rounded

    def _execute_pipeline(
        self,
        question: str,
    ) -> dict[str, Any]:
        """Run evidence acquisition and grounded synthesis synchronously."""

        result = self._agent.invoke(self._initial_state(question))

        evidence = list(
            result.get(
                "evidence",
                [],
            )
        )

        (
            synthesis_plan,
            calculations,
            citations,
        ) = self._synthesizer.synthesize(
            question=question,
            evidence=evidence,
        )

        return {
            **result,
            "answer": synthesis_plan.answer,
            "insufficient_evidence": synthesis_plan.insufficient_evidence,
            "calculations": calculations,
            "citations": citations,
        }

    def _observation_context(
        self,
    ) -> Any:
        """Return tracing context or no-op context."""

        if self._langfuse is None:
            return nullcontext(None)

        return self._langfuse.start_as_current_observation(
            name="fingraph-request",
            as_type="agent",
        )

    def _attribute_context(
        self,
        *,
        request_id: str,
    ) -> Any:
        """Propagate request-scoped attributes to nested observations."""

        if self._langfuse is None:
            return nullcontext()

        return propagate_attributes(
            trace_name="fingraph-request",
            metadata={
                "request_id": request_id,
            },
            tags=[
                "fingraph",
                "api",
            ],
        )

    def _release_capacity_when_done(
        self,
        task: asyncio.Task[dict[str, Any]],
    ) -> None:
        """Release capacity after a still-running worker finishes."""

        def release(
            completed_task: asyncio.Task[dict[str, Any]],
        ) -> None:
            self._semaphore.release()

            try:
                completed_task.result()

            except Exception:
                logger.exception("background_agent_execution_failed")

        task.add_done_callback(release)

    async def ask(
        self,
        *,
        question: str,
        request_id: str,
    ) -> AskResponse:
        """Execute one question with queue and execution timeouts."""

        started = time.perf_counter()

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._queue_timeout_seconds,
            )

        except TimeoutError as exc:
            logger.warning(
                "agent_capacity_exhausted",
                extra={
                    "request_id": request_id,
                },
            )

            raise AgentBusyError("Agent capacity is currently exhausted.") from exc

        worker_task: asyncio.Task[dict[str, Any]] | None = None
        release_in_finally = True

        try:
            with (
                self._observation_context() as observation,
                self._attribute_context(
                    request_id=request_id,
                ),
            ):
                if observation is not None:
                    observation.update(
                        input={
                            "question": question,
                            "request_id": request_id,
                        },
                        metadata={
                            "request_id": request_id,
                        },
                    )

                worker_task = asyncio.create_task(
                    asyncio.to_thread(
                        self._execute_pipeline,
                        question,
                    )
                )

                try:
                    result = await asyncio.wait_for(
                        asyncio.shield(worker_task),
                        timeout=self._request_timeout_seconds,
                    )

                except TimeoutError as exc:
                    release_in_finally = False

                    self._release_capacity_when_done(worker_task)

                    if observation is not None:
                        observation.update(
                            output={
                                "status": "timeout",
                            },
                            level="WARNING",
                            status_message=("Agent request exceeded the configured timeout."),
                        )

                    logger.warning(
                        "agent_request_timed_out",
                        extra={
                            "request_id": request_id,
                            "timeout_seconds": (self._request_timeout_seconds),
                        },
                    )

                    raise AgentTimeoutError("Agent execution timed out.") from exc

                except asyncio.CancelledError:
                    release_in_finally = False

                    self._release_capacity_when_done(worker_task)

                    if observation is not None:
                        observation.update(
                            output={
                                "status": "cancelled",
                            },
                            level="WARNING",
                            status_message=("Agent request was cancelled."),
                        )

                    logger.info(
                        "agent_request_cancelled",
                        extra={
                            "request_id": request_id,
                        },
                    )

                    raise

                except LLMDeferredError as exc:
                    if observation is not None:
                        observation.update(
                            output={
                                "status": "deferred",
                            },
                            level="WARNING",
                            status_message=("Upstream LLM request was deferred."),
                        )

                    logger.warning(
                        "agent_request_deferred",
                        extra={
                            "request_id": request_id,
                        },
                    )

                    raise AgentDeferredError(
                        "LLM provider temporarily unavailable. Retry later."
                    ) from exc

                except LLMQualityError as exc:
                    if observation is not None:
                        observation.update(
                            output={
                                "status": "quality_failed",
                            },
                            level="ERROR",
                            status_message=("LLM output exhausted bounded quality validation."),
                        )

                    logger.error(
                        "agent_request_quality_failed",
                        extra={
                            "request_id": request_id,
                        },
                    )

                    raise AgentQualityError(
                        "LLM output failed deterministic quality validation "
                        "after bounded repair attempts."
                    ) from exc

                except LLMPermanentError as exc:
                    if observation is not None:
                        observation.update(
                            output={
                                "status": "permanent_failure",
                            },
                            level="ERROR",
                            status_message=("Upstream LLM request failed permanently."),
                        )

                    logger.error(
                        "agent_request_permanent_llm_failure",
                        extra={
                            "request_id": request_id,
                        },
                    )

                    raise AgentPermanentError("LLM provider request failed permanently.") from exc

                except Exception as exc:
                    logger.exception(
                        "agent_execution_failed",
                        extra={
                            "request_id": request_id,
                        },
                    )

                    if observation is not None:
                        observation.update(
                            output={
                                "status": "failed",
                            },
                            level="ERROR",
                            status_message=("Agent execution failed."),
                        )

                    raise AgentExecutionError("Agent execution failed.") from exc

                route = result.get("route")

                if route not in {
                    "vector",
                    "graph",
                    "hybrid",
                }:
                    raise AgentExecutionError("Agent returned an invalid route.")

                route_reason = result.get("route_reason") or ""

                evidence = self._round_evidence(
                    list(
                        result.get(
                            "evidence",
                            [],
                        )
                    )
                )

                errors = [
                    str(error)
                    for error in result.get(
                        "errors",
                        [],
                    )
                ]

                latency_ms = round(
                    (time.perf_counter() - started) * 1000,
                    3,
                )

                response = AskResponse(
                    request_id=request_id,
                    question=question,
                    route=route,
                    route_reason=route_reason,
                    answer=str(result["answer"]),
                    insufficient_evidence=bool(result["insufficient_evidence"]),
                    citations=list(
                        result.get(
                            "citations",
                            [],
                        )
                    ),
                    calculations=list(
                        result.get(
                            "calculations",
                            [],
                        )
                    ),
                    evidence=evidence,
                    errors=errors,
                    latency_ms=latency_ms,
                )

                if observation is not None:
                    observation.update(
                        output={
                            "status": "completed",
                            "route": response.route,
                            "evidence_count": len(response.evidence),
                            "citation_count": len(response.citations),
                            "calculation_count": len(response.calculations),
                            "insufficient_evidence": (response.insufficient_evidence),
                            "error_count": len(response.errors),
                            "latency_ms": (response.latency_ms),
                        },
                        metadata={
                            "route": response.route,
                        },
                    )

                logger.info(
                    "agent_request_completed",
                    extra={
                        "request_id": request_id,
                        "route": response.route,
                        "evidence_count": len(response.evidence),
                        "citation_count": len(response.citations),
                        "latency_ms": (response.latency_ms),
                    },
                )

                return response

        finally:
            if release_in_finally:
                self._semaphore.release()
