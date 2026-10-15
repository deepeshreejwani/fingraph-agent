"""Focused tests for Neo4j loader hardening."""

import json
from pathlib import Path

from pytest import MonkeyPatch

import fingraph.storage.neo4j.loader as loader
from fingraph.ingestion.publication import (
    PUBLISHING_MARKER,
)
from fingraph.storage.neo4j.loader import (
    period_id_for_fact,
)


def _write_json(
    path: Path,
    payload: object,
) -> None:
    """Write one deterministic processed test artifact."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_company_snapshot(
    *,
    root: Path,
    ticker: str,
    cik: str,
    accession_number: str,
    section_id: str,
    fact_id: str,
) -> None:
    """Write the canonical processed artifacts needed by graph-ID discovery."""

    company_root = root / ticker

    _write_json(
        company_root / "company.json",
        {
            "cik": cik,
            "ticker": ticker,
            "name": ticker,
            "sic": None,
            "industry": None,
        },
    )

    _write_json(
        company_root / "filings.json",
        [
            {
                "cik": cik,
                "accession_number": (accession_number),
                "form": "10-K",
                "filing_date": ("2025-02-01"),
                "report_date": ("2024-12-31"),
                "fiscal_year": 2024,
                "source_url": (f"https://example.com/{ticker}"),
                "content_hash": (f"hash-{ticker}"),
            }
        ],
    )

    _write_json(
        company_root / "chunks.json",
        [
            {
                "chunk_id": (f"chunk-{ticker}"),
                "filing_id": (accession_number),
                "section_id": (section_id),
                "section_name": ("Item 1. Business"),
                "cik": cik,
                "ticker": ticker,
                "fiscal_year": 2024,
                "text": (f"{ticker} business"),
                "source_url": (f"https://example.com/{ticker}"),
            }
        ],
    )

    _write_json(
        company_root / "financial_facts.json",
        [
            {
                "fact_id": fact_id,
                "cik": cik,
                "taxonomy": "us-gaap",
                "concept": "Revenue",
                "value": 100.0,
                "unit": "USD",
                "start_date": ("2024-01-01"),
                "end_date": ("2024-12-31"),
                "fiscal_year": 2024,
                "fiscal_period": "FY",
                "accession_number": (accession_number),
                "filed_date": ("2025-02-01"),
                "frame": None,
            }
        ],
    )


def test_period_id_uses_economic_dates_not_filing_context_year() -> None:
    first = {
        "cik": "0000789019",
        "start_date": "2024-07-01",
        "end_date": "2025-06-30",
        "fiscal_year": 2026,
    }

    second = {
        **first,
        "fiscal_year": 2025,
    }

    assert period_id_for_fact(first) == period_id_for_fact(second)


def test_company_metadata_ignores_in_progress_partition(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Publishing partitions must be invisible to canonical metadata discovery."""

    root = tmp_path / "processed" / "sec"

    _write_company_snapshot(
        root=root,
        ticker="AAPL",
        cik="0000320193",
        accession_number=("aapl-filing"),
        section_id=("section-aapl"),
        fact_id=("fact-aapl"),
    )

    _write_company_snapshot(
        root=root,
        ticker="MSFT",
        cik="0000789019",
        accession_number=("msft-filing"),
        section_id=("section-msft"),
        fact_id=("fact-msft"),
    )

    (root / "MSFT" / PUBLISHING_MARKER).touch()

    monkeypatch.setattr(
        loader,
        "PROCESSED_ROOT",
        root,
    )

    metadata = loader.company_metadata_by_ticker()

    assert set(metadata) == {
        "AAPL",
    }

    assert metadata["AAPL"]["cik"] == "0000320193"


def test_expected_graph_ids_ignore_in_progress_partition(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Graph audit and pruning inputs must exclude an incompletely published ticker.
    """

    root = tmp_path / "processed" / "sec"

    _write_company_snapshot(
        root=root,
        ticker="AAPL",
        cik="0000320193",
        accession_number=("aapl-filing"),
        section_id=("section-aapl"),
        fact_id=("fact-aapl"),
    )

    _write_company_snapshot(
        root=root,
        ticker="MSFT",
        cik="0000789019",
        accession_number=("msft-filing"),
        section_id=("section-msft"),
        fact_id=("fact-msft"),
    )

    (root / "MSFT" / PUBLISHING_MARKER).touch()

    monkeypatch.setattr(
        loader,
        "PROCESSED_ROOT",
        root,
    )

    expected = loader.expected_graph_ids()

    assert expected["companies"] == [
        "0000320193",
    ]

    assert expected["filings"] == [
        "aapl-filing",
    ]

    assert expected["sections"] == [
        "section-aapl",
    ]

    assert expected["financial_facts"] == [
        "fact-aapl",
    ]

    expected_period_id = period_id_for_fact(
        {
            "cik": "0000320193",
            "start_date": ("2024-01-01"),
            "end_date": ("2024-12-31"),
        }
    )

    assert expected["fiscal_periods"] == [
        expected_period_id,
    ]


def test_expected_graph_ids_restore_company_after_marker_removed(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Removing the publication barrier should make the completed snapshot visible."""

    root = tmp_path / "processed" / "sec"

    _write_company_snapshot(
        root=root,
        ticker="AAPL",
        cik="0000320193",
        accession_number=("aapl-filing"),
        section_id=("section-aapl"),
        fact_id=("fact-aapl"),
    )

    _write_company_snapshot(
        root=root,
        ticker="MSFT",
        cik="0000789019",
        accession_number=("msft-filing"),
        section_id=("section-msft"),
        fact_id=("fact-msft"),
    )

    marker = root / "MSFT" / PUBLISHING_MARKER

    marker.touch()

    monkeypatch.setattr(
        loader,
        "PROCESSED_ROOT",
        root,
    )

    hidden = loader.expected_graph_ids()

    assert hidden["companies"] == [
        "0000320193",
    ]

    marker.unlink()

    published = loader.expected_graph_ids()

    assert published["companies"] == [
        "0000320193",
        "0000789019",
    ]

    assert published["filings"] == [
        "aapl-filing",
        "msft-filing",
    ]

    assert published["sections"] == [
        "section-aapl",
        "section-msft",
    ]

    assert published["financial_facts"] == [
        "fact-aapl",
        "fact-msft",
    ]

    assert len(published["fiscal_periods"]) == 2
