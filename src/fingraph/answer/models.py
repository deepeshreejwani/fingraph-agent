"""Grounded answer models."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class SynthesisPlan(BaseModel):
    """Structured output returned by the grounded answer LLM."""

    model_config = ConfigDict(
        extra="forbid",
    )

    answer: str = Field(
        min_length=1,
    )

    cited_evidence_ids: list[str]

    insufficient_evidence: bool

    @model_validator(mode="after")
    def normalize_insufficient_citations(
        self,
    ) -> "SynthesisPlan":
        """
        Unsupported answers never expose evidence as a factual citation.

        Empty graph/vector results may still exist internally to explain
        why the request was unsupported, but they are not valid SEC
        provenance for a factual answer.
        """

        if self.insufficient_evidence:
            self.cited_evidence_ids = []

        return self


class AnswerCitation(BaseModel):
    """One deterministic user-facing citation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    label: int = Field(
        ge=1,
    )

    evidence_id: str

    source_type: Literal[
        "vector",
        "graph",
    ]

    ticker: str | None = None

    filing_id: str | None = None

    section_name: str | None = None

    source_url: str | None = None


class DeterministicCalculation(BaseModel):
    """One financial calculation performed outside the LLM."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    calculation_type: Literal[
        "change",
        "cagr",
        "margin",
    ] = "change"

    metric: str

    ticker: str = Field(
        min_length=1,
    )
    unit: str = Field(
        min_length=1,
    )

    source_evidence_ids: tuple[str, ...] = Field(
        min_length=1,
    )

    start_year: int | None = None
    end_year: int | None = None

    start_value: str | None = None
    end_value: str | None = None

    absolute_change: str | None = None
    percentage_change: str | None = None

    start_fact_id: str | None = None
    end_fact_id: str | None = None

    start_accession_number: str | None = None
    end_accession_number: str | None = None

    start_period_start_date: str | None = None
    start_period_end_date: str | None = None

    end_period_start_date: str | None = None
    end_period_end_date: str | None = None

    periods: int | None = None
    cagr_percentage: str | None = None

    year: int | None = None

    numerator_metric: str | None = None
    numerator_value: str | None = None

    denominator_metric: str | None = None
    denominator_value: str | None = None

    numerator_fact_id: str | None = None
    denominator_fact_id: str | None = None

    numerator_accession_number: str | None = None
    denominator_accession_number: str | None = None

    period_start_date: str | None = None
    period_end_date: str | None = None

    margin_percentage: str | None = None

    @model_validator(mode="after")
    def validate_calculation_shape(
        self,
    ) -> "DeterministicCalculation":
        """Require fields appropriate to the calculation type."""

        if self.calculation_type == "change":
            required = {
                "start_year": self.start_year,
                "end_year": self.end_year,
                "start_value": self.start_value,
                "end_value": self.end_value,
                "absolute_change": self.absolute_change,
                "start_fact_id": self.start_fact_id,
                "end_fact_id": self.end_fact_id,
                "start_accession_number": self.start_accession_number,
                "end_accession_number": self.end_accession_number,
                "start_period_end_date": self.start_period_end_date,
                "end_period_end_date": self.end_period_end_date,
            }

        elif self.calculation_type == "cagr":
            required = {
                "start_year": self.start_year,
                "end_year": self.end_year,
                "start_value": self.start_value,
                "end_value": self.end_value,
                "periods": self.periods,
                "cagr_percentage": self.cagr_percentage,
                "start_fact_id": self.start_fact_id,
                "end_fact_id": self.end_fact_id,
                "start_accession_number": self.start_accession_number,
                "end_accession_number": self.end_accession_number,
                "start_period_end_date": self.start_period_end_date,
                "end_period_end_date": self.end_period_end_date,
            }

        else:
            required = {
                "year": self.year,
                "numerator_metric": self.numerator_metric,
                "numerator_value": self.numerator_value,
                "denominator_metric": self.denominator_metric,
                "denominator_value": self.denominator_value,
                "numerator_fact_id": self.numerator_fact_id,
                "denominator_fact_id": self.denominator_fact_id,
                "numerator_accession_number": self.numerator_accession_number,
                "denominator_accession_number": self.denominator_accession_number,
                "period_start_date": self.period_start_date,
                "period_end_date": self.period_end_date,
            }

        missing = [name for name, value in required.items() if value is None]

        if missing:
            raise ValueError(
                f"{self.calculation_type} calculation is "
                "missing required fields: " + ", ".join(missing)
            )

        return self
