"""Environment-driven application configuration."""

import logging
from functools import lru_cache
from typing import Literal

from pydantic import (
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

Environment = Literal[
    "development",
    "test",
    "production",
]


class Settings(BaseSettings):
    """Application settings loaded from environment variables and an optional .env file."""

    environment: Environment = "development"

    log_level: str = Field(
        default="INFO",
        min_length=1,
    )

    sec_user_agent: str | None = None

    neo4j_uri: str = "bolt://localhost:7687"

    neo4j_user: str = "neo4j"

    neo4j_password: str | None = None

    neo4j_database: str = "neo4j"

    neo4j_read_timeout_seconds: float = Field(
        default=10.0,
        gt=0.0,
        le=60.0,
    )

    neo4j_connection_timeout_seconds: float = Field(
        default=3.0,
        gt=0.0,
        le=30.0,
    )

    neo4j_max_transaction_retry_time_seconds: float = Field(
        default=3.0,
        ge=0.0,
        le=30.0,
    )

    qdrant_url: str = "http://localhost:6333"

    qdrant_api_key: str | None = None

    qdrant_collection: str = "fingraph_chunks"

    groq_api_key: str | None = None

    llm_provider: str = "groq"

    llm_model: str = "openai/gpt-oss-120b"

    llm_reasoning_effort: str = "low"

    llm_state_db_path: str = "data/state/fingraph_runs.db"

    llm_infrastructure_max_attempts: int = Field(
        default=4,
        ge=1,
        le=10,
    )

    llm_max_retry_wait_seconds: float = Field(
        default=30.0,
        ge=0.0,
        le=300.0,
    )

    llm_synthesis_input_token_budget: int = Field(
        default=16000,
        ge=4096,
        le=128000,
    )

    graph_qa_max_attempts: int = Field(
        default=2,
        ge=1,
        le=3,
    )

    graph_qa_max_rows: int = Field(
        default=25,
        ge=1,
        le=100,
    )

    api_host: str = "127.0.0.1"

    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
    )

    api_request_timeout_seconds: float = Field(
        default=90.0,
        gt=0.0,
        le=600.0,
    )

    api_queue_timeout_seconds: float = Field(
        default=5.0,
        gt=0.0,
        le=60.0,
    )

    api_readiness_timeout_seconds: float = Field(
        default=3.0,
        gt=0.0,
        le=30.0,
    )

    api_max_concurrency: int = Field(
        default=4,
        ge=1,
        le=64,
    )

    api_evidence_limit: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    api_access_key: str | None = None

    api_rate_limit_requests_per_minute: int = Field(
        default=30,
        ge=1,
        le=10_000,
    )

    langfuse_enabled: bool = False

    langfuse_public_key: str | None = None

    langfuse_secret_key: str | None = None

    langfuse_base_url: str = "https://cloud.langfuse.com"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FINGRAPH_",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(
        cls,
        value: str,
    ) -> str:
        """Normalize and validate Python logging levels."""

        normalized = value.upper()

        if normalized not in logging.getLevelNamesMapping():
            raise ValueError(f"Invalid logging level: {value}")

        return normalized

    @model_validator(mode="after")
    def validate_runtime_configuration(
        self,
    ) -> "Settings":
        """Validate environment-dependent application configuration."""

        if self.environment == "production" and not self.api_access_key:
            raise ValueError("FINGRAPH_API_ACCESS_KEY is required in production.")

        if not self.langfuse_enabled:
            return self

        if not self.langfuse_public_key:
            raise ValueError("FINGRAPH_LANGFUSE_PUBLIC_KEY is required when Langfuse is enabled.")

        if not self.langfuse_secret_key:
            raise ValueError("FINGRAPH_LANGFUSE_SECRET_KEY is required when Langfuse is enabled.")

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""

    return Settings()
