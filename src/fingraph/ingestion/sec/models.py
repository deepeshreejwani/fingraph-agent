"""SEC ingestion-specific models."""

from datetime import date

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from fingraph.utils.hashing import sha256_text


class SecModel(BaseModel):
    """Base model for normalized SEC metadata."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


class CompanyReference(SecModel):
    """Company reference obtained from the SEC ticker universe."""

    cik: str = Field(
        min_length=10,
        max_length=10,
    )

    ticker: str = Field(
        min_length=1,
    )

    name: str = Field(
        min_length=1,
    )


class FilingReference(SecModel):
    """Metadata required to locate an SEC filing."""

    cik: str = Field(
        min_length=10,
        max_length=10,
    )

    accession_number: str = Field(
        min_length=1,
    )

    form: str = Field(
        min_length=1,
    )

    filing_date: date

    report_date: date | None = None

    primary_document: str = Field(
        min_length=1,
    )

    primary_document_url: str = Field(
        min_length=1,
    )

    @property
    def accession_number_compact(
        self,
    ) -> str:
        """Return accession number without dashes."""

        return self.accession_number.replace(
            "-",
            "",
        )


class DownloadedFiling(BaseModel):
    """Downloaded filing document with byte-preserving content."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=False,
    )

    reference: FilingReference

    source_url: str = Field(
        min_length=1,
    )

    content: str

    content_hash: str = Field(
        min_length=64,
        max_length=64,
    )

    @model_validator(mode="after")
    def validate_content_hash(
        self,
    ) -> "DownloadedFiling":
        """Require the stored hash to describe the exact filing content."""

        actual_hash = sha256_text(self.content)

        if actual_hash != self.content_hash:
            raise ValueError("Downloaded filing content hash does not match exact content.")

        return self
