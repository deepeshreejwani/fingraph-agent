"""Smoke test LangGraph with real Qdrant and Neo4j tools."""

import json
from typing import Any

from fingraph.agent import (
    GraphEvidenceTool,
    VectorEvidenceTool,
    build_agent_graph,
)
from fingraph.config import (
    get_settings,
)
from fingraph.graph.qa import (
    GraphQAService,
)
from fingraph.llm import (
    GroqCypherGenerator,
    LLMStateStore,
)
from fingraph.storage.neo4j import (
    Neo4jStore,
)
from fingraph.storage.qdrant import (
    QdrantChunkStore,
)


def initial_state(
    question: str,
) -> dict[str, Any]:
    """Create clean state for one agent invocation."""

    return {
        "question": question,
        "route": None,
        "route_reason": None,
        "vector_evidence": [],
        "graph_evidence": [],
        "evidence": [],
        "errors": [],
    }


def print_result(
    result: dict[str, Any],
) -> None:
    """Print compact agent execution output."""

    payload = {
        "question": (result["question"]),
        "route": (result["route"]),
        "route_reason": (result["route_reason"]),
        "vector_evidence_count": (len(result["vector_evidence"])),
        "graph_evidence_count": (len(result["graph_evidence"])),
        "evidence": [evidence.model_dump(mode="python") for evidence in result["evidence"]],
    }

    print(
        json.dumps(
            payload,
            indent=2,
            default=str,
        )
    )

    print()


def main() -> None:
    settings = get_settings()

    if not settings.groq_api_key:
        raise SystemExit("FINGRAPH_GROQ_API_KEY is required.")

    if not settings.neo4j_password:
        raise SystemExit("FINGRAPH_NEO4J_PASSWORD is required.")

    qdrant_store = QdrantChunkStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=(settings.qdrant_collection),
    )

    qdrant_store.health_check()

    vector_tool = VectorEvidenceTool(
        qdrant_store,
        limit=(settings.api_evidence_limit),
    )

    state_store = LLMStateStore(settings.llm_state_db_path)

    generator = GroqCypherGenerator(
        api_key=(settings.groq_api_key),
        model=(settings.llm_model),
        state_store=(state_store),
        reasoning_effort=(settings.llm_reasoning_effort),
        infrastructure_max_attempts=(settings.llm_infrastructure_max_attempts),
        max_retry_wait_seconds=(settings.llm_max_retry_wait_seconds),
    )

    with Neo4jStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=(settings.neo4j_password),
        database=(settings.neo4j_database),
        read_timeout_seconds=(settings.neo4j_read_timeout_seconds),
        connection_timeout_seconds=(settings.neo4j_connection_timeout_seconds),
        max_transaction_retry_time_seconds=(settings.neo4j_max_transaction_retry_time_seconds),
    ) as neo4j_store:
        graph_service = GraphQAService(
            store=(neo4j_store),
            generator=(generator),
            max_attempts=(settings.graph_qa_max_attempts),
            max_rows=(settings.graph_qa_max_rows),
        )

        graph_tool = GraphEvidenceTool(graph_service)

        agent = build_agent_graph(
            vector_tool=(vector_tool),
            graph_tool=(graph_tool),
        )

        questions = [
            ("What was Microsoft's revenue in 2026?"),
            ("What risks does NVIDIA describe in its 10-K?"),
            (
                "How did Microsoft's "
                "revenue change from "
                "2024 to 2026 and "
                "explain what management "
                "said drove the trend?"
            ),
        ]

        for question in questions:
            result = agent.invoke(initial_state(question))

            print_result(result)


if __name__ == "__main__":
    main()
