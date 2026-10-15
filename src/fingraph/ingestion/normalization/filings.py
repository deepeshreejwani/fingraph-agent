"""Normalize SEC filing references."""

from fingraph.domain.models import Filing
from fingraph.ingestion.sec.models import (
    FilingReference,
)


def normalize_filing(
    filing: FilingReference,
    *,
    content_hash: str,
) -> Filing:
    """Convert SEC filing metadata into the canonical model."""

    return Filing(
        accession_number=(filing.accession_number),
        cik=filing.cik,
        form=filing.form,
        filing_date=(filing.filing_date),
        report_date=(filing.report_date),
        fiscal_year=(filing.report_date.year if filing.report_date else None),
        source_url=(filing.primary_document_url),
        content_hash=(content_hash),
    )
