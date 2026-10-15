"""Public API schemas."""

from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from fingraph.agent import Evidence
from fingraph.answer.models import (
    AnswerCitation,
    DeterministicCalculation,
)

RouteName = Literal[
    "vector",
    "graph",
    "hybrid",
]


class AskRequest(BaseModel):
    """Question submitted to the FinGraph agent."""

    model_config = ConfigDict(
        extra="forbid",
    )

    question: str = Field(
        min_length=1,
        max_length=2000,
    )

    @field_validator("question")
    @classmethod
    def normalize_question(
        cls,
        value: str,
    ) -> str:
        """Trim and validate user questions."""

        normalized = value.strip()

        if len(normalized) < 3:
            raise ValueError("Question must contain at least 3 non-whitespace characters.")

        return normalized


class AskResponse(BaseModel):
    """Completed FinGraph execution."""

    model_config = ConfigDict(
        extra="forbid",
    )

    request_id: str

    question: str

    route: RouteName

    route_reason: str

    answer: str

    insufficient_evidence: bool

    citations: list[AnswerCitation]

    calculations: list[DeterministicCalculation]

    evidence: list[Evidence]

    errors: list[str]

    latency_ms: float


class HealthResponse(BaseModel):
    """Liveness response."""

    status: Literal["ok"]

    service: str

    version: str


class ReadinessResponse(BaseModel):
    """Dependency-readiness response."""

    ready: bool

    checks: dict[
        str,
        bool,
    ]


class ApiErrorResponse(BaseModel):
    """Stable API error envelope."""

    model_config = ConfigDict(
        extra="forbid",
    )

    request_id: str

    error_code: str

    message: str

    details: Any | None = None
