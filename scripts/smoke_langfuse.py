"""Verify Langfuse connectivity without using Groq."""

from fingraph.api.observability import (
    build_langfuse_client,
)
from fingraph.config import get_settings


def main() -> None:
    """Emit one standalone Langfuse smoke trace."""

    settings = get_settings()

    langfuse = build_langfuse_client(settings)

    if langfuse is None:
        raise RuntimeError(
            "Langfuse is disabled. Set "
            "FINGRAPH_LANGFUSE_ENABLED=true "
            "and configure Langfuse credentials."
        )

    trace_id: str | None = None

    trace_url: str | None = None

    try:
        with langfuse.start_as_current_observation(
            name="fingraph-langfuse-smoke",
            as_type="agent",
            input={"purpose": ("Validate FinGraph Langfuse integration.")},
        ) as root:
            trace_id = langfuse.get_current_trace_id()

            with langfuse.start_as_current_observation(
                name="smoke-retrieval",
                as_type="retriever",
                input={"query": ("Langfuse connectivity smoke test")},
            ) as child:
                child.update(
                    output={
                        "result_count": 1,
                        "status": "ok",
                    }
                )

            root.update(
                output={
                    "status": "ok",
                }
            )

            trace_url = langfuse.get_trace_url(trace_id=trace_id)

        langfuse.flush()

    finally:
        langfuse.shutdown()

    print("Langfuse smoke test: OK")
    print(f"Trace ID: {trace_id}")

    if trace_url is not None:
        print(f"Trace URL: {trace_url}")


if __name__ == "__main__":
    main()
