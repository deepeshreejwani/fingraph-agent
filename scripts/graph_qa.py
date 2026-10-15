"""Run safe resumable Text-to-Cypher graph QA."""

import argparse
import json

from fingraph.config import (
    get_settings,
)
from fingraph.graph.qa import (
    GraphQAService,
)
from fingraph.llm import (
    GroqCypherGenerator,
    LLMDeferredError,
    LLMPermanentError,
    LLMStateStore,
)
from fingraph.storage.neo4j import (
    Neo4jStore,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "question",
    )

    args = parser.parse_args()

    settings = get_settings()

    if not settings.neo4j_password:
        raise SystemExit("FINGRAPH_NEO4J_PASSWORD is required.")

    if not settings.groq_api_key:
        raise SystemExit("FINGRAPH_GROQ_API_KEY is required.")

    if settings.llm_provider != "groq":
        raise SystemExit(
            "This Phase-1 graph QA entry point currently expects FINGRAPH_LLM_PROVIDER=groq."
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

    try:
        with Neo4jStore(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=(settings.neo4j_password),
            database=(settings.neo4j_database),
        ) as store:
            service = GraphQAService(
                store=store,
                generator=generator,
                max_attempts=(settings.graph_qa_max_attempts),
                max_rows=(settings.graph_qa_max_rows),
            )

            result = service.ask(args.question)

    except LLMDeferredError as exc:
        raise SystemExit(f"DEFERRED: {exc}") from exc

    except LLMPermanentError as exc:
        raise SystemExit(f"LLM FAILED: {exc}") from exc

    print(
        json.dumps(
            result.model_dump(mode="python"),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
