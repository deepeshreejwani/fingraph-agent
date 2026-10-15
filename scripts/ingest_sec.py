"""Acquire and normalize the configured SEC corpus."""

from __future__ import annotations

import argparse
import json
import logging
import time
import tomllib
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Any

from fingraph.config import get_settings
from fingraph.domain.models import Filing
from fingraph.ingestion.normalization import (
    normalize_company_facts,
)
from fingraph.ingestion.normalization.filings import (
    normalize_filing,
)
from fingraph.ingestion.parsing import (
    chunk_section,
    extract_sections,
    html_to_text,
)
from fingraph.ingestion.publication import (
    publish_company_snapshot,
)
from fingraph.ingestion.sec import (
    SecClient,
    discover_10k_filings,
    download_filing,
    fetch_company_facts,
    fetch_company_universe,
    select_companies,
)
from fingraph.ingestion.sec.manifest import (
    Manifest,
    ManifestEntry,
)
from fingraph.ingestion.sec.models import (
    CompanyReference,
)
from fingraph.logging import configure_logging
from fingraph.utils.hashing import (
    sha256_bytes,
    sha256_text,
)
from fingraph.utils.ids import stable_id

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/development.toml")

RAW_ROOT = Path("data/raw/sec")

PROCESSED_ROOT = Path("data/processed/sec")

MANIFEST_PATH = Path("data/raw/sec/manifest.json")


def load_config(
    path: Path,
) -> dict[str, Any]:
    """Load TOML configuration."""

    with path.open("rb") as handle:
        return tomllib.load(handle)


def write_text_atomic(
    path: Path,
    content: str,
) -> None:
    """Atomically persist the exact UTF-8 byte representation of text."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_bytes(content.encode("utf-8"))

    temporary.replace(path)


def write_json(
    path: Path,
    payload: object,
) -> None:
    """Write deterministic pretty JSON atomically."""

    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        default=str,
    )

    write_text_atomic(
        path,
        serialized,
    )


def canonical_json_hash(
    payload: object,
) -> str:
    """Hash JSON independently of pretty-print formatting."""

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    return sha256_text(serialized)


def cached_text_if_valid(
    *,
    path: Path,
    expected_hash: str,
) -> str | None:
    """Return cached UTF-8 text only when persisted bytes match the manifest hash."""

    if not path.exists():
        return None

    try:
        raw = path.read_bytes()

    except OSError:
        return None

    if sha256_bytes(raw) != expected_hash:
        return None

    try:
        return raw.decode("utf-8")

    except UnicodeDecodeError:
        return None


def cached_json_is_valid(
    *,
    path: Path,
    expected_hash: str,
) -> bool:
    """Validate a cached JSON artifact against its canonical hash."""

    if not path.exists():
        return False

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return False

    return canonical_json_hash(payload) == expected_hash


def company_payload(
    company: CompanyReference,
) -> dict[str, Any]:
    """Persist canonical company metadata available from SEC discovery."""

    raw = company.model_dump(mode="json")

    name = raw.get("name") or raw.get("title") or company.ticker

    return {
        "cik": company.cik,
        "ticker": company.ticker,
        "name": name,
        "sic": raw.get("sic"),
        "industry": raw.get("industry"),
    }


def partition_companies(
    companies: list[CompanyReference],
    *,
    partition_count: int,
    partition_index: int,
) -> list[CompanyReference]:
    """
    Deterministically select one company partition.

    Companies are first sorted by ticker/CIK and then assigned round-robin by
    stable position. The same universe and partition configuration therefore
    always produce the same selection.
    """

    if partition_count < 1:
        raise ValueError("partition_count must be at least 1.")

    if partition_index < 0:
        raise ValueError("partition_index cannot be negative.")

    if partition_index >= partition_count:
        raise ValueError("partition_index must be smaller than partition_count.")

    ordered = sorted(
        companies,
        key=lambda company: (
            company.ticker,
            company.cik,
        ),
    )

    return [
        company
        for index, company in enumerate(ordered)
        if index % partition_count == partition_index
    ]


def ingest_company(
    *,
    client: SecClient,
    company: CompanyReference,
    filings_per_company: int,
    chunk_size_words: int,
    overlap_words: int,
    manifest: Manifest,
) -> tuple[
    dict[str, int],
    list[ManifestEntry],
]:
    """
    Ingest one company and publish one complete processed snapshot.

    Manifest mutations are returned to the coordinator rather than applied to
    the shared manifest from worker threads. A company's manifest mutations are
    therefore committed only after its processed snapshot has published
    successfully.
    """

    ticker = company.ticker
    cik = company.cik

    logger.info(
        "Starting company ingestion",
        extra={
            "ticker": ticker,
            "cik": cik,
        },
    )

    filings = discover_10k_filings(
        client,
        company,
        limit=filings_per_company,
    )

    allowed_accessions = {filing.accession_number for filing in filings}

    company_root = RAW_ROOT / ticker

    pending_manifest_entries: list[ManifestEntry] = []

    filing_count = 0
    filing_downloaded_count = 0
    filing_cache_hit_count = 0
    filing_cache_repair_count = 0
    section_count = 0

    all_chunks = []

    normalized_filings: list[Filing] = []

    for filing in filings:
        resource_id = stable_id(
            "filing",
            cik,
            filing.accession_number,
        )

        filing_path = company_root / "filings" / f"{filing.accession_number}.html"

        existing = manifest.get(resource_id)

        html: str | None = None

        cache_was_invalid = False

        if existing is not None:
            html = cached_text_if_valid(
                path=filing_path,
                expected_hash=(existing.content_hash),
            )

            if html is None:
                cache_was_invalid = True

        if html is not None:
            content_hash = existing.content_hash if existing is not None else sha256_text(html)

            filing_cache_hit_count += 1

            logger.info(
                "Using verified cached filing",
                extra={
                    "ticker": ticker,
                    "accession_number": (filing.accession_number),
                },
            )

        else:
            downloaded = download_filing(
                client,
                filing,
            )

            html = downloaded.content
            content_hash = downloaded.content_hash

            write_text_atomic(
                filing_path,
                html,
            )

            persisted_hash = sha256_bytes(filing_path.read_bytes())

            if persisted_hash != content_hash:
                raise RuntimeError(
                    "Persisted SEC filing hash does not "
                    "match downloaded content hash: "
                    f"{filing.accession_number}"
                )

            pending_manifest_entries.append(
                ManifestEntry(
                    resource_id=resource_id,
                    resource_type="10-K",
                    cik=cik,
                    accession_number=(filing.accession_number),
                    source_url=(filing.primary_document_url),
                    local_path=str(filing_path),
                    content_hash=content_hash,
                )
            )

            filing_downloaded_count += 1

            if cache_was_invalid:
                filing_cache_repair_count += 1

                logger.warning(
                    "Repaired invalid cached filing",
                    extra={
                        "ticker": ticker,
                        "accession_number": (filing.accession_number),
                    },
                )

        normalized_filings.append(
            normalize_filing(
                filing,
                content_hash=content_hash,
            )
        )

        filing_count += 1

        plain_text = html_to_text(html)

        sections = extract_sections(
            plain_text,
            filing_id=(filing.accession_number),
        )

        fiscal_year = filing.report_date.year if filing.report_date is not None else None

        for section in sections:
            section_count += 1

            all_chunks.extend(
                chunk_section(
                    section,
                    cik=cik,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    source_url=(filing.primary_document_url),
                    chunk_size_words=(chunk_size_words),
                    overlap_words=(overlap_words),
                )
            )

    chunk_count = len(all_chunks)

    facts_resource_id = stable_id(
        "companyfacts",
        cik,
    )

    facts_cache_path = company_root / "companyfacts.json"

    facts_manifest_entry = manifest.get(facts_resource_id)

    raw_facts = dict(
        fetch_company_facts(
            client,
            cik,
        )
    )

    facts_hash = canonical_json_hash(raw_facts)

    cache_valid = (
        facts_manifest_entry is not None
        and (facts_manifest_entry.content_hash == facts_hash)
        and cached_json_is_valid(
            path=facts_cache_path,
            expected_hash=facts_hash,
        )
    )

    companyfacts_refreshed = 0

    if not cache_valid:
        write_json(
            facts_cache_path,
            raw_facts,
        )

        pending_manifest_entries.append(
            ManifestEntry(
                resource_id=(facts_resource_id),
                resource_type=("companyfacts"),
                cik=cik,
                source_url=(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"),
                local_path=str(facts_cache_path),
                content_hash=facts_hash,
            )
        )

        companyfacts_refreshed = 1

        logger.info(
            "Refreshed SEC companyfacts",
            extra={
                "ticker": ticker,
                "cik": cik,
            },
        )

    else:
        logger.info(
            "SEC companyfacts unchanged; keeping verified cache",
            extra={
                "ticker": ticker,
                "cik": cik,
            },
        )

    facts = normalize_company_facts(
        raw_facts,
        allowed_accessions=(allowed_accessions),
    )

    processed_artifacts: dict[
        str,
        object,
    ] = {
        "company.json": (company_payload(company)),
        "filings.json": [filing.model_dump(mode="json") for filing in normalized_filings],
        "chunks.json": [chunk.model_dump(mode="json") for chunk in all_chunks],
        "financial_facts.json": [fact.model_dump(mode="json") for fact in facts],
    }

    publish_company_snapshot(
        processed_root=(PROCESSED_ROOT),
        ticker=ticker,
        artifacts=(processed_artifacts),
    )

    logger.info(
        "Completed company ingestion",
        extra={
            "ticker": ticker,
            "filings": filing_count,
            "filings_downloaded": (filing_downloaded_count),
            "filings_cached": (filing_cache_hit_count),
            "filing_cache_repairs": (filing_cache_repair_count),
            "sections": section_count,
            "chunks": chunk_count,
            "financial_facts": (len(facts)),
            "companyfacts_refreshed": (companyfacts_refreshed),
        },
    )

    return (
        {
            "filings": filing_count,
            "filings_downloaded": (filing_downloaded_count),
            "filings_cached": (filing_cache_hit_count),
            "filing_cache_repairs": (filing_cache_repair_count),
            "sections": section_count,
            "chunks": chunk_count,
            "financial_facts": (len(facts)),
            "companyfacts_refreshed": (companyfacts_refreshed),
        },
        pending_manifest_entries,
    )


def _empty_totals() -> dict[str, int]:
    """Return zeroed ingestion counters."""

    return {
        "companies": 0,
        "filings": 0,
        "filings_downloaded": 0,
        "filings_cached": 0,
        "filing_cache_repairs": 0,
        "sections": 0,
        "chunks": 0,
        "financial_facts": 0,
        "companyfacts_refreshed": 0,
    }


def _merge_company_result(
    totals: dict[str, int],
    result: dict[str, int],
) -> None:
    """Merge one company's counters into run totals."""

    totals["companies"] += 1

    for key, value in result.items():
        totals[key] += value


def _commit_company_manifest_entries(
    manifest: Manifest,
    entries: list[ManifestEntry],
) -> None:
    """
    Commit one successfully published company's raw-resource metadata.

    The coordinator is the only execution context that persists the shared
    manifest. Failed or still-running company workers therefore cannot leak
    manifest mutations into another company's commit.
    """

    if entries:
        manifest.upsert_many(
            entries,
            persist=False,
        )

    manifest.flush()


def _run_companies(
    *,
    client: SecClient,
    companies: list[CompanyReference],
    workers: int,
    filings_per_company: int,
    chunk_size_words: int,
    overlap_words: int,
    manifest: Manifest,
) -> dict[str, int]:
    """Execute company ingestion sequentially or with bounded threads."""

    if workers < 1:
        raise ValueError("workers must be at least 1.")

    totals = _empty_totals()

    if not companies:
        return totals

    effective_workers = min(
        workers,
        len(companies),
    )

    if effective_workers == 1:
        for company in companies:
            (
                result,
                manifest_entries,
            ) = ingest_company(
                client=client,
                company=company,
                filings_per_company=(filings_per_company),
                chunk_size_words=(chunk_size_words),
                overlap_words=(overlap_words),
                manifest=manifest,
            )

            _merge_company_result(
                totals,
                result,
            )

            _commit_company_manifest_entries(
                manifest,
                manifest_entries,
            )

        return totals

    with ThreadPoolExecutor(
        max_workers=effective_workers,
        thread_name_prefix="sec-company",
    ) as executor:
        futures: dict[
            Future[
                tuple[
                    dict[str, int],
                    list[ManifestEntry],
                ]
            ],
            CompanyReference,
        ] = {}

        for company in companies:
            future = executor.submit(
                ingest_company,
                client=client,
                company=company,
                filings_per_company=(filings_per_company),
                chunk_size_words=(chunk_size_words),
                overlap_words=(overlap_words),
                manifest=manifest,
            )

            futures[future] = company

        for future in as_completed(futures):
            company = futures[future]

            try:
                (
                    result,
                    manifest_entries,
                ) = future.result()

            except Exception:
                logger.exception(
                    "Company ingestion failed",
                    extra={
                        "ticker": (company.ticker),
                        "cik": company.cik,
                    },
                )

                raise

            _merge_company_result(
                totals,
                result,
            )

            _commit_company_manifest_entries(
                manifest,
                manifest_entries,
            )

    return totals


def build_summary(
    *,
    totals: dict[str, int],
    elapsed_seconds: float,
    workers: int,
    partition_count: int,
    partition_index: int,
) -> dict[str, int | float]:
    """Add reproducible execution metrics to ingestion counters."""

    elapsed = max(
        elapsed_seconds,
        0.0,
    )

    filings = totals["filings"]

    companies = totals["companies"]

    filings_per_minute = filings / elapsed * 60.0 if elapsed > 0.0 else 0.0

    companies_per_second = companies / elapsed if elapsed > 0.0 else 0.0

    return {
        **totals,
        "elapsed_seconds": round(
            elapsed,
            3,
        ),
        "filings_per_minute": round(
            filings_per_minute,
            3,
        ),
        "companies_per_second": round(
            companies_per_second,
            3,
        ),
        "workers": workers,
        "partition_count": (partition_count),
        "partition_index": (partition_index),
    }


def main() -> None:
    """Run configured SEC ingestion."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=("Override [execution].workers from the TOML configuration."),
    )

    parser.add_argument(
        "--partition-count",
        type=int,
        default=None,
        help=("Override [execution].partition_count."),
    )

    parser.add_argument(
        "--partition-index",
        type=int,
        default=None,
        help=("Override [execution].partition_index."),
    )

    args = parser.parse_args()

    settings = get_settings()

    configure_logging(settings.log_level)

    if not settings.sec_user_agent:
        raise SystemExit("FINGRAPH_SEC_USER_AGENT is required. Set it in your local .env file.")

    config = load_config(args.config)

    dataset_config = config["dataset"]

    sec_config = config["sec"]

    chunking_config = config["chunking"]

    execution_config = config.get(
        "execution",
        {},
    )

    workers = (
        args.workers
        if args.workers is not None
        else int(
            execution_config.get(
                "workers",
                1,
            )
        )
    )

    partition_count = (
        args.partition_count
        if args.partition_count is not None
        else int(
            execution_config.get(
                "partition_count",
                1,
            )
        )
    )

    partition_index = (
        args.partition_index
        if args.partition_index is not None
        else int(
            execution_config.get(
                "partition_index",
                0,
            )
        )
    )

    if workers < 1:
        raise SystemExit("workers must be at least 1.")

    requested_tickers = set(dataset_config["tickers"])

    manifest = Manifest(MANIFEST_PATH)

    started_at = time.perf_counter()

    with SecClient(
        user_agent=(settings.sec_user_agent),
        requests_per_second=float(sec_config["requests_per_second"]),
        timeout_seconds=float(sec_config["timeout_seconds"]),
        max_retries=int(sec_config["max_retries"]),
    ) as client:
        universe = fetch_company_universe(client)

        companies = select_companies(
            universe,
            requested_tickers,
        )

        try:
            companies = partition_companies(
                companies,
                partition_count=(partition_count),
                partition_index=(partition_index),
            )

        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        logger.info(
            "Configured SEC ingestion execution",
            extra={
                "requested_companies": (len(requested_tickers)),
                "selected_companies": (len(companies)),
                "workers": workers,
                "partition_count": (partition_count),
                "partition_index": (partition_index),
                "requests_per_second": (float(sec_config["requests_per_second"])),
            },
        )

        totals = _run_companies(
            client=client,
            companies=companies,
            workers=workers,
            filings_per_company=int(dataset_config["filings_per_company"]),
            chunk_size_words=int(chunking_config["chunk_size_words"]),
            overlap_words=int(chunking_config["overlap_words"]),
            manifest=manifest,
        )

    manifest.flush()

    elapsed_seconds = time.perf_counter() - started_at

    summary = build_summary(
        totals=totals,
        elapsed_seconds=(elapsed_seconds),
        workers=workers,
        partition_count=(partition_count),
        partition_index=(partition_index),
    )

    write_json(
        PROCESSED_ROOT / "ingestion_summary.json",
        summary,
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
