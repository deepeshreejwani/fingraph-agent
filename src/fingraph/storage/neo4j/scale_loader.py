"""Bounded-memory Neo4j loading for the v1.1 scale corpus."""

from __future__ import annotations

import json
from collections.abc import (
    Iterator,
)
from pathlib import Path
from typing import Any

from fingraph.ingestion.publication import (
    published_artifact_paths,
)
from fingraph.storage.neo4j.loader import (
    COMPANY_QUERY,
    FACT_QUERY,
    FILING_QUERY,
    PERIOD_QUERY,
    SECTION_QUERY,
    period_id_for_fact,
)
from fingraph.storage.neo4j.loader import (
    PROCESSED_ROOT as DEFAULT_PROCESSED_ROOT,
)
from fingraph.storage.neo4j.store import (
    Neo4jStore,
)

PROCESSED_ROOT = DEFAULT_PROCESSED_ROOT


def _artifact_paths(
    filename: str,
) -> list[Path]:
    """Return published canonical artifact paths in deterministic ticker order."""

    return published_artifact_paths(
        PROCESSED_ROOT,
        filename,
    )


def _read_json(
    path: Path,
) -> Any:
    """Read one canonical processed artifact."""

    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_list(
    path: Path,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Read and validate one list-shaped canonical artifact."""

    payload = _read_json(path)

    if not isinstance(
        payload,
        list,
    ):
        raise ValueError(f"Expected JSON array: {path}")

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for item in payload:
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(f"Expected JSON objects in artifact: {path}")

        rows.append(item)

    return rows


def _read_json_dict(
    path: Path,
) -> dict[
    str,
    Any,
]:
    """Read and validate one object-shaped canonical artifact."""

    payload = _read_json(path)

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(f"Expected JSON object: {path}")

    return payload


def iter_company_rows() -> Iterator[
    dict[
        str,
        Any,
    ]
]:
    """
    Yield canonical company rows one ticker at a time.

    Company metadata is preferred. The historical chunk-based fallback remains
    available for compatibility with older processed artifacts.
    """

    metadata_paths = {path.parent.name: path for path in _artifact_paths("company.json")}

    chunk_paths = {path.parent.name: path for path in _artifact_paths("chunks.json")}

    tickers = sorted(set(metadata_paths) | set(chunk_paths))

    for ticker in tickers:
        metadata_path = metadata_paths.get(ticker)

        if metadata_path is not None:
            company = _read_json_dict(metadata_path)

            cik = company.get("cik")

            if cik is None:
                raise ValueError(f"Company metadata missing CIK: {ticker}")

            yield {
                "cik": str(cik),
                "ticker": ticker,
                "name": (company.get("name") or ticker),
                "sic": (company.get("sic")),
                "industry": (company.get("industry")),
            }

            continue

        chunk_path = chunk_paths.get(ticker)

        if chunk_path is None:
            continue

        chunks = _read_json_list(chunk_path)

        if not chunks:
            continue

        first = chunks[0]

        yield {
            "cik": str(first["cik"]),
            "ticker": ticker,
            "name": ticker,
            "sic": None,
            "industry": None,
        }


def load_companies_scaled(
    store: Neo4jStore,
    *,
    batch_size: int,
) -> int:
    """Load companies using bounded UNWIND transactions."""

    return store.execute_batch(
        COMPANY_QUERY,
        iter_company_rows(),
        batch_size=batch_size,
    )


def load_filings_and_sections_scaled(
    store: Neo4jStore,
    *,
    batch_size: int,
) -> tuple[
    int,
    int,
]:
    """
    Load filings and sections one company artifact at a time.

    Only the currently processed ticker's artifact is retained in memory.
    """

    filing_count = 0

    for path in _artifact_paths("filings.json"):
        filings = _read_json_list(path)

        filing_count += store.execute_batch(
            FILING_QUERY,
            filings,
            batch_size=(batch_size),
        )

    section_count = 0

    for path in _artifact_paths("chunks.json"):
        chunks = _read_json_list(path)

        sections: dict[
            str,
            dict[
                str,
                Any,
            ],
        ] = {}

        for chunk in chunks:
            section_id = str(chunk["section_id"])

            sections.setdefault(
                section_id,
                {
                    "section_id": (section_id),
                    "filing_id": (chunk["filing_id"]),
                    "section_name": (chunk["section_name"]),
                },
            )

        section_count += store.execute_batch(
            SECTION_QUERY,
            sections.values(),
            batch_size=(batch_size),
        )

    return (
        filing_count,
        section_count,
    )


def _period_rows_for_facts(
    facts: list[
        dict[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:
    """Build unique fiscal-period rows for one company's facts."""

    periods: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for fact in facts:
        period_id = period_id_for_fact(fact)

        end_date = str(fact["end_date"])

        periods.setdefault(
            period_id,
            {
                "period_id": (period_id),
                "start_date": (fact.get("start_date")),
                "end_date": (end_date),
                "period_year": int(end_date[:4]),
            },
        )

    return periods


def _iter_fact_rows(
    facts: list[
        dict[
            str,
            Any,
        ]
    ],
) -> Iterator[
    dict[
        str,
        Any,
    ]
]:
    """Yield normalized graph fact rows for one company artifact."""

    for fact in facts:
        yield {
            **fact,
            "period_id": (period_id_for_fact(fact)),
            "value": float(fact["value"]),
        }


def load_facts_scaled(
    store: Neo4jStore,
    *,
    batch_size: int,
) -> tuple[
    int,
    int,
]:
    """
    Load fiscal periods and financial facts company by company.

    The corpus-wide fact list is never materialized. Each company's raw fact
    artifact is released before the next ticker is loaded.
    """

    fact_count = 0
    period_count = 0

    for path in _artifact_paths("financial_facts.json"):
        facts = _read_json_list(path)

        periods = _period_rows_for_facts(facts)

        period_count += store.execute_batch(
            PERIOD_QUERY,
            periods.values(),
            batch_size=(batch_size),
        )

        fact_count += store.execute_batch(
            FACT_QUERY,
            _iter_fact_rows(facts),
            batch_size=(batch_size),
        )

    return (
        fact_count,
        period_count,
    )
