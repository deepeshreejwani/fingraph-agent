"""Canonical retrieval models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RetrievalMethod = Literal[
    "sparse",
    "dense",
    "hybrid",
]


class RetrievalFilter(BaseModel):
    """Optional metadata restrictions for retrieval."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    tickers: tuple[str, ...] = ()
    fiscal_years: tuple[int, ...] = ()
    section_names: tuple[str, ...] = ()


class RetrievalResult(BaseModel):
    """One ranked retrieval result."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    chunk_id: str
    filing_id: str
    section_id: str

    cik: str
    ticker: str
    fiscal_year: int | None

    section_name: str | None = None

    text: str
    source_url: str

    score: float

    method: RetrievalMethod

    rank: int = Field(
        ge=1,
    )
