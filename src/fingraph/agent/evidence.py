"""Canonical evidence models shared across FinGraph tools."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EvidenceSource = Literal[
    "vector",
    "graph",
]


class Evidence(BaseModel):
    """One normalized piece of evidence used by the agent."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    evidence_id: str = Field(
        min_length=1,
    )

    source_type: EvidenceSource

    ticker: str | None = None
    filing_id: str | None = None
    section_name: str | None = None

    text: str | None = None

    structured_payload: dict[str, Any] | None = None

    source_url: str | None = None

    score: float | None = None

    rank: int | None = Field(
        default=None,
        ge=1,
    )

    provenance: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )
