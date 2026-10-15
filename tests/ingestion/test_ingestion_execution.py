"""Tests for scalable and crash-safe SEC ingestion execution."""

from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any

import pytest
import scripts.ingest_sec as ingest_sec_module
from pytest import MonkeyPatch
from scripts.ingest_sec import (
    _run_companies,
    build_summary,
    load_config,
    partition_companies,
)

from fingraph.ingestion.sec.manifest import (
    ManifestEntry,
)
from fingraph.ingestion.sec.models import (
    CompanyReference,
)


def company(
    ticker: str,
    cik: int,
) -> CompanyReference:
    """Create a company reference for partition tests."""

    return CompanyReference(
        cik=str(cik).zfill(10),
        ticker=ticker,
        name=ticker,
    )


class TrackingManifest:
    """Minimal manifest double exposing coordinator commits."""

    def __init__(
        self,
    ) -> None:
        self.entries: dict[
            str,
            ManifestEntry,
        ] = {}

        self.flush_count = 0

        self.flushed = Event()

    def get(
        self,
        resource_id: str,
    ) -> ManifestEntry | None:
        """Return one staged test entry."""

        return self.entries.get(resource_id)

    def upsert_many(
        self,
        entries: list[ManifestEntry],
        *,
        persist: bool = True,
    ) -> int:
        """Record coordinator-owned manifest mutations."""

        changed = 0

        for entry in entries:
            previous = self.entries.get(entry.resource_id)

            if previous == entry:
                continue

            self.entries[entry.resource_id] = entry

            changed += 1

        if persist:
            self.flush()

        return changed

    def flush(
        self,
    ) -> None:
        """Record one coordinator manifest flush."""

        self.flush_count += 1

        self.flushed.set()


def _manifest_entry(
    *,
    ticker: str,
) -> ManifestEntry:
    """Create one deterministic manifest entry."""

    normalized = ticker.upper()

    return ManifestEntry(
        resource_id=(f"resource-{normalized}"),
        resource_type="10-K",
        cik=("0000000001" if normalized == "AAPL" else "0000000002"),
        accession_number=(f"{normalized}-filing"),
        source_url=(f"https://example.com/{normalized}"),
        local_path=(f"data/raw/sec/{normalized}/filing.html"),
        content_hash=(f"hash-{normalized}"),
    )


def _company_result() -> dict[
    str,
    int,
]:
    """Return one successful company-ingestion counter payload."""

    return {
        "filings": 1,
        "filings_downloaded": 1,
        "filings_cached": 0,
        "filing_cache_repairs": 0,
        "sections": 4,
        "chunks": 10,
        "financial_facts": 20,
        "companyfacts_refreshed": 1,
    }


def test_partition_companies_is_deterministic() -> None:
    companies = [
        company(
            "META",
            4,
        ),
        company(
            "AAPL",
            1,
        ),
        company(
            "MSFT",
            3,
        ),
        company(
            "AMZN",
            2,
        ),
    ]

    first = partition_companies(
        companies,
        partition_count=2,
        partition_index=0,
    )

    second = partition_companies(
        list(reversed(companies)),
        partition_count=2,
        partition_index=0,
    )

    assert [item.ticker for item in first] == [item.ticker for item in second]

    assert [item.ticker for item in first] == [
        "AAPL",
        "META",
    ]


def test_partitions_cover_each_company_once() -> None:
    companies = [
        company(
            "AAPL",
            1,
        ),
        company(
            "AMZN",
            2,
        ),
        company(
            "META",
            3,
        ),
        company(
            "MSFT",
            4,
        ),
        company(
            "NVDA",
            5,
        ),
    ]

    selected = []

    for partition_index in range(3):
        selected.extend(
            partition_companies(
                companies,
                partition_count=3,
                partition_index=(partition_index),
            )
        )

    assert sorted(item.ticker for item in selected) == sorted(item.ticker for item in companies)


@pytest.mark.parametrize(
    (
        "partition_count",
        "partition_index",
    ),
    [
        (
            0,
            0,
        ),
        (
            1,
            -1,
        ),
        (
            2,
            2,
        ),
    ],
)
def test_invalid_partition_configuration_is_rejected(
    partition_count: int,
    partition_index: int,
) -> None:
    with pytest.raises(ValueError):
        partition_companies(
            [],
            partition_count=(partition_count),
            partition_index=(partition_index),
        )


def test_development_config_preserves_single_worker() -> None:
    config = load_config(Path("configs/development.toml"))

    assert config["execution"]["workers"] == 1

    assert config["execution"]["partition_count"] == 1

    assert config["execution"]["partition_index"] == 0


def test_ingestion_summary_rounds_metrics_to_three_decimals() -> None:
    totals = {
        "companies": 2,
        "filings": 10,
        "filings_downloaded": 4,
        "filings_cached": 6,
        "filing_cache_repairs": 0,
        "sections": 40,
        "chunks": 500,
        "financial_facts": 1000,
        "companyfacts_refreshed": 2,
    }

    summary = build_summary(
        totals=totals,
        elapsed_seconds=12.34567,
        workers=4,
        partition_count=1,
        partition_index=0,
    )

    assert summary["elapsed_seconds"] == 12.346

    assert summary["filings_per_minute"] == 48.6

    assert summary["companies_per_second"] == 0.162


def test_successful_company_manifest_entries_are_committed_by_coordinator(
    monkeypatch: MonkeyPatch,
) -> None:
    """Successful workers should return entries for coordinator-owned commit."""

    manifest = TrackingManifest()

    entry = _manifest_entry(ticker="AAPL")

    def fake_ingest_company(
        **_: Any,
    ) -> tuple[
        dict[str, int],
        list[ManifestEntry],
    ]:
        return (
            _company_result(),
            [
                entry,
            ],
        )

    monkeypatch.setattr(
        ingest_sec_module,
        "ingest_company",
        fake_ingest_company,
    )

    totals = _run_companies(
        client=object(),  # type: ignore[arg-type]
        companies=[
            company(
                "AAPL",
                1,
            )
        ],
        workers=1,
        filings_per_company=1,
        chunk_size_words=100,
        overlap_words=10,
        manifest=manifest,  # type: ignore[arg-type]
    )

    assert totals["companies"] == 1

    assert list(manifest.entries) == [entry.resource_id]

    assert manifest.flush_count == 1


def test_failed_parallel_company_cannot_leak_manifest_entries_into_other_flush(
    monkeypatch: MonkeyPatch,
) -> None:
    """
    A successful company may flush while another worker is still running.

    The still-running worker's local manifest entries must remain invisible
    and must never be persisted if that company later fails.
    """

    manifest = TrackingManifest()

    successful_entry = _manifest_entry(ticker="AAPL")

    failed_entry = _manifest_entry(ticker="MSFT")

    failure_worker_started = Event()

    def fake_ingest_company(
        *,
        company: CompanyReference,
        **_: Any,
    ) -> tuple[
        dict[str, int],
        list[ManifestEntry],
    ]:
        if company.ticker == "AAPL":
            assert failure_worker_started.wait(timeout=2.0)

            return (
                _company_result(),
                [
                    successful_entry,
                ],
            )

        local_entries = [failed_entry]

        assert local_entries

        failure_worker_started.set()

        assert manifest.flushed.wait(timeout=2.0)

        raise RuntimeError("simulated company failure")

    monkeypatch.setattr(
        ingest_sec_module,
        "ingest_company",
        fake_ingest_company,
    )

    with pytest.raises(
        RuntimeError,
        match=("simulated company failure"),
    ):
        _run_companies(
            client=object(),  # type: ignore[arg-type]
            companies=[
                company(
                    "AAPL",
                    1,
                ),
                company(
                    "MSFT",
                    2,
                ),
            ],
            workers=2,
            filings_per_company=1,
            chunk_size_words=100,
            overlap_words=10,
            manifest=manifest,  # type: ignore[arg-type]
        )

    assert successful_entry.resource_id in manifest.entries

    assert failed_entry.resource_id not in manifest.entries

    assert manifest.flush_count == 1
