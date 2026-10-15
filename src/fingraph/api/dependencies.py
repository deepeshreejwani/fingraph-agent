"""Application dependency wiring."""

import asyncio
from collections.abc import Callable
from dataclasses import (
    dataclass,
    field,
)
from typing import Any

from langfuse import Langfuse

from fingraph.agent import (
    GraphEvidenceTool,
    VectorEvidenceTool,
    build_agent_graph,
)
from fingraph.api.observability import (
    build_langfuse_client,
)
from fingraph.api.service import AgentService
from fingraph.api.tracing import (
    TracedEvidenceTool,
    TracedGroqAnswerSynthesizer,
    TracedGroqCypherGenerator,
)
from fingraph.config import Settings
from fingraph.graph.qa import GraphQAService
from fingraph.llm import LLMStateStore
from fingraph.storage.neo4j import Neo4jStore
from fingraph.storage.qdrant import (
    QdrantChunkStore,
)


@dataclass
class ApplicationResources:
    """Long-lived application resources."""

    settings: Settings

    qdrant_store: QdrantChunkStore

    neo4j_store: Neo4jStore

    state_store: LLMStateStore

    agent: Any

    agent_service: AgentService

    langfuse: Langfuse | None

    _closed: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    async def _bounded_health_check(
        self,
        check: Callable[[], None],
    ) -> bool:
        """Run one blocking health check within the readiness deadline."""

        try:
            await asyncio.wait_for(
                asyncio.to_thread(check),
                timeout=(self.settings.api_readiness_timeout_seconds),
            )

            return True

        except (
            TimeoutError,
            Exception,
        ):
            return False

    def _qdrant_readiness_check(
        self,
    ) -> None:
        """Verify Qdrant and the configured serving collection are available."""

        self.qdrant_store.health_check()

        if not self.qdrant_store.collection_exists():
            raise RuntimeError(
                "Configured Qdrant collection "
                f"{self.qdrant_store.collection_name!r} "
                "does not exist."
            )

    async def readiness(
        self,
    ) -> dict[str, bool]:
        """Check dependencies without executing an LLM request."""

        (
            qdrant_ready,
            neo4j_ready,
        ) = await asyncio.gather(
            self._bounded_health_check(self._qdrant_readiness_check),
            self._bounded_health_check(self.neo4j_store.health_check),
        )

        return {
            "qdrant": qdrant_ready,
            "neo4j": neo4j_ready,
            "groq_configured": bool(self.settings.groq_api_key),
        }

    def close(
        self,
    ) -> None:
        """Release external resources once."""

        if self._closed:
            return

        self._closed = True

        try:
            if self.langfuse is not None:
                self.langfuse.shutdown()

        finally:
            try:
                close_client = getattr(
                    self.qdrant_store.client,
                    "close",
                    None,
                )

                if callable(close_client):
                    close_client()

            finally:
                self.neo4j_store.close()


def build_application_resources(
    settings: Settings,
) -> ApplicationResources:
    """Build production application dependencies."""

    if not settings.groq_api_key:
        raise RuntimeError("FINGRAPH_GROQ_API_KEY is required.")

    if not settings.neo4j_password:
        raise RuntimeError("FINGRAPH_NEO4J_PASSWORD is required.")

    qdrant_store = QdrantChunkStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=(settings.qdrant_collection),
    )

    neo4j_store = Neo4jStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        read_timeout_seconds=(settings.neo4j_read_timeout_seconds),
        connection_timeout_seconds=(settings.neo4j_connection_timeout_seconds),
        max_transaction_retry_time_seconds=(settings.neo4j_max_transaction_retry_time_seconds),
    )

    langfuse: Langfuse | None = None

    try:
        state_store = LLMStateStore(settings.llm_state_db_path)

        langfuse = build_langfuse_client(settings)

        generator = TracedGroqCypherGenerator(
            api_key=(settings.groq_api_key),
            model=(settings.llm_model),
            state_store=state_store,
            reasoning_effort=(settings.llm_reasoning_effort),
            infrastructure_max_attempts=(settings.llm_infrastructure_max_attempts),
            max_retry_wait_seconds=(settings.llm_max_retry_wait_seconds),
            langfuse=langfuse,
        )

        synthesizer = TracedGroqAnswerSynthesizer(
            api_key=(settings.groq_api_key),
            model=(settings.llm_model),
            state_store=state_store,
            reasoning_effort=(settings.llm_reasoning_effort),
            infrastructure_max_attempts=(settings.llm_infrastructure_max_attempts),
            max_retry_wait_seconds=(settings.llm_max_retry_wait_seconds),
            langfuse=langfuse,
        )

        synthesizer.set_input_token_budget(settings.llm_synthesis_input_token_budget)

        graph_service = GraphQAService(
            store=neo4j_store,
            generator=generator,
            max_attempts=(settings.graph_qa_max_attempts),
            max_rows=(settings.graph_qa_max_rows),
        )

        raw_vector_tool = VectorEvidenceTool(
            qdrant_store,
            limit=(settings.api_evidence_limit),
        )

        raw_graph_tool = GraphEvidenceTool(graph_service)

        vector_tool = TracedEvidenceTool(
            tool=raw_vector_tool,
            langfuse=langfuse,
            name="vector-retrieval",
            observation_type="retriever",
        )

        graph_tool = TracedEvidenceTool(
            tool=raw_graph_tool,
            langfuse=langfuse,
            name="graph-qa",
            observation_type="agent",
        )

        agent = build_agent_graph(
            vector_tool=vector_tool,
            graph_tool=graph_tool,
        )

        agent_service = AgentService(
            agent=agent,
            synthesizer=synthesizer,
            request_timeout_seconds=(settings.api_request_timeout_seconds),
            queue_timeout_seconds=(settings.api_queue_timeout_seconds),
            max_concurrency=(settings.api_max_concurrency),
            langfuse=langfuse,
        )

        return ApplicationResources(
            settings=settings,
            qdrant_store=qdrant_store,
            neo4j_store=neo4j_store,
            state_store=state_store,
            agent=agent,
            agent_service=agent_service,
            langfuse=langfuse,
        )

    except Exception:
        if langfuse is not None:
            langfuse.shutdown()

        try:
            close_client = getattr(
                qdrant_store.client,
                "close",
                None,
            )

            if callable(close_client):
                close_client()

        finally:
            neo4j_store.close()

        raise
