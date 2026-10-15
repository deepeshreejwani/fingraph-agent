"""Framework-independent canonical domain models."""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class DomainModel(BaseModel):
    """Base model shared by canonical FinGraph entities."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


class Company(DomainModel):
    """SEC company identity and classification."""

    cik: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    name: str = Field(min_length=1)

    industry: str | None = None
    sic: str | None = None


class Filing(DomainModel):
    """Canonical SEC filing metadata."""

    accession_number: str = Field(
        min_length=1,
    )

    cik: str = Field(
        min_length=1,
    )

    form: str = Field(
        min_length=1,
    )

    filing_date: date

    report_date: date | None = None

    fiscal_year: int | None = Field(
        default=None,
        ge=1900,
        le=2200,
    )

    source_url: str = Field(
        min_length=1,
    )

    content_hash: str = Field(
        min_length=1,
    )


class Section(DomainModel):
    """Narrative section extracted from an SEC filing."""

    section_id: str = Field(
        min_length=1,
    )

    filing_id: str = Field(
        min_length=1,
    )

    section_name: str = Field(
        min_length=1,
    )

    text: str = Field(
        min_length=1,
    )


class Chunk(DomainModel):
    """Retrieval-ready chunk preserving filing provenance."""

    chunk_id: str = Field(
        min_length=1,
    )

    filing_id: str = Field(
        min_length=1,
    )

    section_id: str = Field(
        min_length=1,
    )

    section_name: str = Field(
        min_length=1,
    )

    cik: str = Field(
        min_length=1,
    )

    ticker: str = Field(
        min_length=1,
    )

    fiscal_year: int | None = Field(
        default=None,
        ge=1900,
        le=2200,
    )

    text: str = Field(
        min_length=1,
    )

    content_hash: str = Field(
        min_length=1,
    )

    source_url: str = Field(
        min_length=1,
    )


class FinancialFact(DomainModel):
    """Normalized SEC XBRL financial observation."""

    fact_id: str = Field(
        min_length=1,
    )

    cik: str = Field(
        min_length=1,
    )

    taxonomy: str = Field(
        min_length=1,
    )

    concept: str = Field(
        min_length=1,
    )

    value: Decimal

    unit: str = Field(
        min_length=1,
    )

    # Actual economic period represented by this fact.
    start_date: date | None = None

    end_date: date

    # SEC filing-context metadata.
    #
    # This must not be assumed to equal the actual year represented by
    # end_date. Comparative facts from prior periods can appear inside
    # a newer annual filing.
    fiscal_year: int | None = Field(
        default=None,
        ge=1900,
        le=2200,
    )

    fiscal_period: str | None = None

    accession_number: str = Field(
        min_length=1,
    )

    # Date on which the SEC filing containing this observation was filed.
    filed_date: date | None = None

    # SEC XBRL frame when supplied, for example CY2025.
    frame: str | None = None

    @property
    def period_year(self) -> int:
        """Calendar year associated with the actual fact end date."""

        return self.end_date.year
