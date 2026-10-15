"""Focused runtime hardening regression tests."""

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest

from fingraph.api.dependencies import (
    ApplicationResources,
)
from fingraph.api.service import (
    AgentDeferredError,
    AgentQualityError,
    AgentService,
)
from fingraph.config import Settings
from fingraph.llm.groq_client import (
    LLMDeferredError,
    LLMQualityError,
)


class DeferredAgent:
    """Raise the production deferred LLM signal."""

    def invoke(
        self,
        _: dict[str, Any],
    ) -> dict[str, Any]:
        raise LLMDeferredError("quota exhausted")


class QualityFailureAgent:
    """Raise the production LLM output-quality failure signal."""

    def invoke(
        self,
        _: dict[str, Any],
    ) -> dict[str, Any]:
        raise LLMQualityError("grounding validation exhausted")


class UnusedSynthesizer:
    """Synthesizer that must never execute."""

    def synthesize(
        self,
        **_: Any,
    ) -> None:
        raise AssertionError("synthesizer should not run")


class SlowHealthStore:
    """Blocking dependency health check."""

    def health_check(
        self,
    ) -> None:
        time.sleep(0.25)


class HealthyNeo4jStore:
    """Successful Neo4j dependency health check."""

    def health_check(
        self,
    ) -> None:
        return None


class ReadyQdrantStore:
    """Healthy Qdrant store with the serving collection available."""

    collection_name = "test_chunks"

    def health_check(
        self,
    ) -> None:
        return None

    def collection_exists(
        self,
    ) -> bool:
        return True


class MissingCollectionQdrantStore:
    """Reachable Qdrant store without the configured serving collection."""

    collection_name = "test_chunks"

    def health_check(
        self,
    ) -> None:
        return None

    def collection_exists(
        self,
    ) -> bool:
        return False


class SlowQdrantStore:
    """Slow Qdrant readiness dependency."""

    collection_name = "test_chunks"

    def health_check(
        self,
    ) -> None:
        time.sleep(0.25)

    def collection_exists(
        self,
    ) -> bool:
        return True


def _settings(
    *,
    readiness_timeout_seconds: float = 0.05,
) -> Settings:
    """Return deterministic test settings isolated from the local environment."""

    return Settings(
        environment="test",
        groq_api_key="test",
        api_readiness_timeout_seconds=(readiness_timeout_seconds),
        langfuse_enabled=False,
        _env_file=None,
    )


def _resources(
    *,
    qdrant_store: Any,
    neo4j_store: Any,
) -> ApplicationResources:
    """Build lightweight application resources for readiness tests."""

    return ApplicationResources(
        settings=_settings(),
        qdrant_store=qdrant_store,
        neo4j_store=neo4j_store,
        state_store=SimpleNamespace(),  # type: ignore[arg-type]
        agent=SimpleNamespace(),
        agent_service=SimpleNamespace(),  # type: ignore[arg-type]
        langfuse=None,
    )


def test_llm_deferred_error_is_preserved() -> None:
    async def run() -> None:
        service = AgentService(
            agent=DeferredAgent(),
            synthesizer=UnusedSynthesizer(),  # type: ignore[arg-type]
            request_timeout_seconds=5.0,
            queue_timeout_seconds=1.0,
            max_concurrency=1,
        )

        with pytest.raises(
            AgentDeferredError,
            match="temporarily unavailable",
        ):
            await service.ask(
                question=("What was Microsoft revenue?"),
                request_id=("test-deferred"),
            )

    asyncio.run(run())


def test_llm_quality_error_is_not_reported_as_provider_deferral() -> None:
    async def run() -> None:
        service = AgentService(
            agent=QualityFailureAgent(),
            synthesizer=UnusedSynthesizer(),  # type: ignore[arg-type]
            request_timeout_seconds=5.0,
            queue_timeout_seconds=1.0,
            max_concurrency=1,
        )

        with pytest.raises(
            AgentQualityError,
            match=("deterministic quality validation"),
        ):
            await service.ask(
                question=("What was Microsoft revenue?"),
                request_id=("test-quality-failure"),
            )

    asyncio.run(run())


def test_readiness_requires_configured_qdrant_collection() -> None:
    async def run() -> None:
        resources = _resources(
            qdrant_store=(MissingCollectionQdrantStore()),
            neo4j_store=(HealthyNeo4jStore()),
        )

        checks = await resources.readiness()

        assert checks == {
            "qdrant": False,
            "neo4j": True,
            "groq_configured": True,
        }

    asyncio.run(run())


def test_readiness_accepts_available_qdrant_collection() -> None:
    async def run() -> None:
        resources = _resources(
            qdrant_store=(ReadyQdrantStore()),
            neo4j_store=(HealthyNeo4jStore()),
        )

        checks = await resources.readiness()

        assert checks == {
            "qdrant": True,
            "neo4j": True,
            "groq_configured": True,
        }

    asyncio.run(run())


def test_readiness_is_bounded() -> None:
    async def run() -> None:
        resources = _resources(
            qdrant_store=(ReadyQdrantStore()),
            neo4j_store=(SlowHealthStore()),
        )

        started = time.perf_counter()

        checks = await resources.readiness()

        elapsed = time.perf_counter() - started

        assert checks == {
            "qdrant": True,
            "neo4j": False,
            "groq_configured": True,
        }

        assert elapsed < 0.2

    asyncio.run(run())


def test_qdrant_readiness_is_bounded() -> None:
    async def run() -> None:
        resources = _resources(
            qdrant_store=(SlowQdrantStore()),
            neo4j_store=(HealthyNeo4jStore()),
        )

        started = time.perf_counter()

        checks = await resources.readiness()

        elapsed = time.perf_counter() - started

        assert checks == {
            "qdrant": False,
            "neo4j": True,
            "groq_configured": True,
        }

        assert elapsed < 0.2

    asyncio.run(run())
