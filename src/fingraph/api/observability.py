"""Optional Langfuse observability integration."""

from langfuse import Langfuse

from fingraph.config import Settings


def build_langfuse_client(
    settings: Settings,
) -> Langfuse | None:
    """Create a Langfuse client only when explicitly enabled."""

    if not settings.langfuse_enabled:
        return None

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        environment=settings.environment,
        tracing_enabled=True,
    )
