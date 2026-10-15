"""10-K discovery and filing acquisition."""

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from fingraph.ingestion.sec.client import SecClient
from fingraph.ingestion.sec.models import (
    CompanyReference,
    DownloadedFiling,
    FilingReference,
)
from fingraph.utils.hashing import sha256_text

SUBMISSIONS_BASE_URL = "https://data.sec.gov/submissions"

ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"


def submissions_url(
    cik: str,
) -> str:
    """Build the SEC submissions URL for a CIK."""

    return f"{SUBMISSIONS_BASE_URL}/CIK{cik}.json"


def historical_submissions_url(
    filename: str,
) -> str:
    """Build the URL for an SEC historical submissions file."""

    normalized = filename.strip()

    if not normalized:
        raise ValueError("Historical submissions filename must not be empty.")

    if "/" in normalized or "\\" in normalized or not normalized.endswith(".json"):
        raise ValueError("Invalid SEC historical submissions filename.")

    return f"{SUBMISSIONS_BASE_URL}/{normalized}"


def filing_document_url(
    *,
    cik: str,
    accession_number: str,
    primary_document: str,
) -> str:
    """Build the canonical SEC filing document URL."""

    cik_without_leading_zeros = str(int(cik))

    accession_compact = accession_number.replace(
        "-",
        "",
    )

    return f"{ARCHIVES_BASE_URL}/{cik_without_leading_zeros}/{accession_compact}/{primary_document}"


def discover_10k_filings(
    client: SecClient,
    company: CompanyReference,
    *,
    limit: int | None = None,
) -> list[FilingReference]:
    """Discover recent and historical non-amended 10-K filings."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided.")

    payload = client.get_json(submissions_url(company.cik))

    filings = payload.get("filings")

    if not isinstance(
        filings,
        Mapping,
    ):
        return []

    references: list[FilingReference] = []

    recent = filings.get("recent")

    if isinstance(
        recent,
        Mapping,
    ):
        references.extend(
            _parse_filing_table(
                company.cik,
                recent,
            )
        )

    references = _deduplicate_filings(references)

    if _enough_10ks(
        references,
        limit=limit,
    ):
        return _select_10ks(
            references,
            limit=limit,
        )

    historical_files = _historical_file_entries(filings.get("files"))

    for entry in historical_files:
        filename = entry.get("name")

        if not isinstance(
            filename,
            str,
        ):
            continue

        historical_payload = client.get_json(historical_submissions_url(filename))

        historical_table = _extract_filing_table(historical_payload)

        if historical_table is None:
            continue

        references.extend(
            _parse_filing_table(
                company.cik,
                historical_table,
            )
        )

        references = _deduplicate_filings(references)

        if _enough_10ks(
            references,
            limit=limit,
        ):
            break

    return _select_10ks(
        references,
        limit=limit,
    )


def download_filing(
    client: SecClient,
    reference: FilingReference,
) -> DownloadedFiling:
    """Download a filing's primary document."""

    content = client.get_text(reference.primary_document_url)

    return DownloadedFiling(
        reference=reference,
        source_url=(reference.primary_document_url),
        content=content,
        content_hash=(sha256_text(content)),
    )


def _extract_filing_table(
    payload: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Extract the columnar filing table from an SEC submissions payload."""

    if _looks_like_filing_table(payload):
        return payload

    filings = payload.get("filings")

    if not isinstance(
        filings,
        Mapping,
    ):
        return None

    recent = filings.get("recent")

    if isinstance(
        recent,
        Mapping,
    ) and _looks_like_filing_table(recent):
        return recent

    return None


def _looks_like_filing_table(
    value: Mapping[str, Any],
) -> bool:
    """Return whether a mapping resembles an SEC filing table."""

    required = {
        "accessionNumber",
        "form",
        "filingDate",
        "primaryDocument",
    }

    return required.issubset(value.keys())


def _historical_file_entries(
    value: object,
) -> list[Mapping[str, Any]]:
    """Normalize SEC filings.files entries newest-first."""

    if not isinstance(
        value,
        Sequence,
    ) or isinstance(
        value,
        (
            str,
            bytes,
        ),
    ):
        return []

    entries = [
        entry
        for entry in value
        if isinstance(
            entry,
            Mapping,
        )
    ]

    return sorted(
        entries,
        key=_historical_file_sort_key,
        reverse=True,
    )


def _historical_file_sort_key(
    entry: Mapping[str, Any],
) -> str:
    """Sort historical files using their latest covered filing date."""

    filing_to = entry.get("filingTo")

    if filing_to is None:
        return ""

    return str(filing_to)


def _parse_filing_table(
    cik: str,
    table: Mapping[str, Any],
) -> list[FilingReference]:
    """Parse an SEC columnar filing table."""

    accessions = _as_sequence(table.get("accessionNumber"))

    forms = _as_sequence(table.get("form"))

    filing_dates = _as_sequence(table.get("filingDate"))

    report_dates = _as_sequence(table.get("reportDate"))

    primary_documents = _as_sequence(table.get("primaryDocument"))

    required_lengths = {
        len(accessions),
        len(forms),
        len(filing_dates),
        len(primary_documents),
    }

    if len(required_lengths) != 1:
        raise ValueError("SEC filing arrays have inconsistent lengths.")

    expected_length = len(accessions)

    if not report_dates:
        report_dates = [None] * expected_length

    elif len(report_dates) != expected_length:
        raise ValueError("SEC reportDate array has inconsistent length.")

    references: list[FilingReference] = []

    for (
        accession,
        form,
        filing_date_raw,
        report_date_raw,
        primary_document,
    ) in zip(
        accessions,
        forms,
        filing_dates,
        report_dates,
        primary_documents,
        strict=True,
    ):
        if not all(
            (
                accession,
                form,
                filing_date_raw,
                primary_document,
            )
        ):
            continue

        accession_text = str(accession).strip()

        form_text = str(form).strip()

        primary_document_text = str(primary_document).strip()

        if not (accession_text and form_text and primary_document_text):
            continue

        references.append(
            FilingReference(
                cik=cik,
                accession_number=(accession_text),
                form=form_text,
                filing_date=(date.fromisoformat(str(filing_date_raw))),
                report_date=(date.fromisoformat(str(report_date_raw)) if report_date_raw else None),
                primary_document=(primary_document_text),
                primary_document_url=(
                    filing_document_url(
                        cik=cik,
                        accession_number=(accession_text),
                        primary_document=(primary_document_text),
                    )
                ),
            )
        )

    return references


def _deduplicate_filings(
    references: Sequence[FilingReference],
) -> list[FilingReference]:
    """Deduplicate filing references by accession number."""

    by_accession: dict[
        str,
        FilingReference,
    ] = {}

    for reference in references:
        existing = by_accession.get(reference.accession_number)

        if existing is None or reference.filing_date > existing.filing_date:
            by_accession[reference.accession_number] = reference

    return list(by_accession.values())


def _select_10ks(
    references: Sequence[FilingReference],
    *,
    limit: int | None,
) -> list[FilingReference]:
    """Return exact 10-K filings newest-first."""

    ten_ks = [filing for filing in references if filing.form == "10-K"]

    ten_ks.sort(
        key=lambda filing: (
            filing.filing_date,
            filing.accession_number,
        ),
        reverse=True,
    )

    if limit is not None:
        return ten_ks[:limit]

    return ten_ks


def _enough_10ks(
    references: Sequence[FilingReference],
    *,
    limit: int | None,
) -> bool:
    """Return whether enough exact 10-Ks are available."""

    if limit is None:
        return False

    count = sum(1 for reference in references if reference.form == "10-K")

    return count >= limit


def _as_sequence(
    value: object,
) -> Sequence[object]:
    """Normalize a non-string sequence."""

    if isinstance(
        value,
        Sequence,
    ) and not isinstance(
        value,
        (
            str,
            bytes,
        ),
    ):
        return value

    return []
