"""Tests for recent and historical SEC filing discovery."""

from typing import Any

import pytest

from fingraph.ingestion.sec.filings import (
    discover_10k_filings,
    historical_submissions_url,
)
from fingraph.ingestion.sec.models import (
    CompanyReference,
)


class FakeSecClient:
    """Minimal deterministic SEC client for filing discovery tests."""

    def __init__(
        self,
        responses: dict[
            str,
            dict[str, Any],
        ],
    ) -> None:
        self.responses = responses
        self.requests: list[str] = []

    def get_json(
        self,
        url: str,
    ) -> dict[str, Any]:
        self.requests.append(url)

        return self.responses[url]


def _company() -> CompanyReference:
    return CompanyReference(
        cik="0000789019",
        ticker="MSFT",
        name="Microsoft Corp",
    )


def _table(
    *,
    accessions: list[str],
    forms: list[str],
    filing_dates: list[str],
    report_dates: list[str],
    primary_documents: list[str],
) -> dict[str, Any]:
    return {
        "accessionNumber": (accessions),
        "form": forms,
        "filingDate": (filing_dates),
        "reportDate": (report_dates),
        "primaryDocument": (primary_documents),
    }


def test_historical_submissions_url() -> None:
    assert historical_submissions_url("CIK0000789019-submissions-001.json") == (
        "https://data.sec.gov/submissions/CIK0000789019-submissions-001.json"
    )


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "../bad.json",
        "nested/bad.json",
        r"nested\bad.json",
        "not-json.txt",
    ],
)
def test_historical_submissions_url_rejects_invalid_names(
    filename: str,
) -> None:
    with pytest.raises(ValueError):
        historical_submissions_url(filename)


def test_recent_filings_are_returned_without_historical_request() -> None:
    main_url = "https://data.sec.gov/submissions/CIK0000789019.json"

    client = FakeSecClient(
        {
            main_url: {
                "filings": {
                    "recent": _table(
                        accessions=[
                            "0000789019-26-000001",
                            "0000789019-25-000001",
                        ],
                        forms=[
                            "10-K",
                            "10-K",
                        ],
                        filing_dates=[
                            "2026-07-29",
                            "2025-07-30",
                        ],
                        report_dates=[
                            "2026-06-30",
                            "2025-06-30",
                        ],
                        primary_documents=[
                            "msft-20260630.htm",
                            "msft-20250630.htm",
                        ],
                    ),
                    "files": [
                        {
                            "name": ("CIK0000789019-submissions-001.json"),
                            "filingFrom": ("2020-01-01"),
                            "filingTo": ("2024-12-31"),
                        }
                    ],
                }
            }
        }
    )

    result = discover_10k_filings(
        client,  # type: ignore[arg-type]
        _company(),
        limit=2,
    )

    assert len(result) == 2

    assert [filing.filing_date.year for filing in result] == [
        2026,
        2025,
    ]

    assert client.requests == [main_url]


def test_historical_file_is_loaded_when_recent_is_insufficient() -> None:
    main_url = "https://data.sec.gov/submissions/CIK0000789019.json"

    historical_name = "CIK0000789019-submissions-001.json"

    historical_url = f"https://data.sec.gov/submissions/{historical_name}"

    client = FakeSecClient(
        {
            main_url: {
                "filings": {
                    "recent": _table(
                        accessions=[
                            "0000789019-26-000001",
                        ],
                        forms=[
                            "10-K",
                        ],
                        filing_dates=[
                            "2026-07-29",
                        ],
                        report_dates=[
                            "2026-06-30",
                        ],
                        primary_documents=[
                            "msft-20260630.htm",
                        ],
                    ),
                    "files": [
                        {
                            "name": (historical_name),
                            "filingFrom": ("2020-01-01"),
                            "filingTo": ("2025-12-31"),
                        }
                    ],
                }
            },
            historical_url: _table(
                accessions=[
                    "0000789019-25-000001",
                    "0000789019-25-000002",
                    "0000789019-24-000001",
                ],
                forms=[
                    "10-K",
                    "10-K/A",
                    "10-K",
                ],
                filing_dates=[
                    "2025-07-30",
                    "2025-08-05",
                    "2024-07-30",
                ],
                report_dates=[
                    "2025-06-30",
                    "2025-06-30",
                    "2024-06-30",
                ],
                primary_documents=[
                    "msft-20250630.htm",
                    "msft-20250630x10ka.htm",
                    "msft-20240630.htm",
                ],
            ),
        }
    )

    result = discover_10k_filings(
        client,  # type: ignore[arg-type]
        _company(),
        limit=3,
    )

    assert [filing.filing_date.year for filing in result] == [
        2026,
        2025,
        2024,
    ]

    assert all(filing.form == "10-K" for filing in result)

    assert client.requests == [
        main_url,
        historical_url,
    ]


def test_duplicate_accessions_are_removed() -> None:
    main_url = "https://data.sec.gov/submissions/CIK0000789019.json"

    historical_name = "CIK0000789019-submissions-001.json"

    historical_url = f"https://data.sec.gov/submissions/{historical_name}"

    duplicate_accession = "0000789019-25-000001"

    client = FakeSecClient(
        {
            main_url: {
                "filings": {
                    "recent": _table(
                        accessions=[
                            duplicate_accession,
                        ],
                        forms=[
                            "10-K",
                        ],
                        filing_dates=[
                            "2025-07-30",
                        ],
                        report_dates=[
                            "2025-06-30",
                        ],
                        primary_documents=[
                            "msft-20250630.htm",
                        ],
                    ),
                    "files": [
                        {
                            "name": (historical_name),
                            "filingFrom": ("2024-01-01"),
                            "filingTo": ("2025-12-31"),
                        }
                    ],
                }
            },
            historical_url: _table(
                accessions=[
                    duplicate_accession,
                    "0000789019-24-000001",
                ],
                forms=[
                    "10-K",
                    "10-K",
                ],
                filing_dates=[
                    "2025-07-30",
                    "2024-07-30",
                ],
                report_dates=[
                    "2025-06-30",
                    "2024-06-30",
                ],
                primary_documents=[
                    "msft-20250630.htm",
                    "msft-20240630.htm",
                ],
            ),
        }
    )

    result = discover_10k_filings(
        client,  # type: ignore[arg-type]
        _company(),
        limit=2,
    )

    accessions = [filing.accession_number for filing in result]

    assert accessions == [
        duplicate_accession,
        "0000789019-24-000001",
    ]


def test_limit_must_be_positive() -> None:
    client = FakeSecClient({})

    with pytest.raises(ValueError):
        discover_10k_filings(
            client,  # type: ignore[arg-type]
            _company(),
            limit=0,
        )
