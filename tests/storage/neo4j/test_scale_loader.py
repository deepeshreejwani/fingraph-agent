"""Tests for bounded Neo4j scale loading."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest
from pytest import MonkeyPatch

import fingraph.storage.neo4j.scale_loader as scale_loader
from fingraph.ingestion.publication import (
    PUBLISHING_MARKER,
)
from fingraph.storage.neo4j.loader import (
    FACT_QUERY,
    PERIOD_QUERY,
)
from fingraph.storage.neo4j.store import (
    Neo4jStore,
)


def test_execute_batch_splits_transactions() -> None:
    """Rows should be written in bounded UNWIND transactions."""

    store = object.__new__(Neo4jStore)

    execute_write = Mock()

    cast(
        Any,
        store,
    ).execute_write = execute_write

    rows = [
        {
            "id": index,
        }
        for index in range(5)
    ]

    written = store.execute_batch(
        "UNWIND $rows AS row RETURN row",
        rows,
        batch_size=2,
    )

    assert written == 5
    assert execute_write.call_count == 3

    batch_sizes = [len(call.kwargs["parameters"]["rows"]) for call in execute_write.call_args_list]

    assert batch_sizes == [
        2,
        2,
        1,
    ]


def test_execute_batch_accepts_generator() -> None:
    """Batch writes should consume generators without requiring a list."""

    store = object.__new__(Neo4jStore)

    execute_write = Mock()

    cast(
        Any,
        store,
    ).execute_write = execute_write

    rows = (
        {
            "id": index,
        }
        for index in range(3)
    )

    written = store.execute_batch(
        "UNWIND $rows AS row RETURN row",
        rows,
        batch_size=2,
    )

    assert written == 3
    assert execute_write.call_count == 2


def test_execute_batch_rejects_invalid_size() -> None:
    """Transaction batch size must be positive."""

    store = object.__new__(Neo4jStore)

    execute_write = Mock()

    cast(
        Any,
        store,
    ).execute_write = execute_write

    with pytest.raises(
        ValueError,
        match=("batch_size must be greater than zero"),
    ):
        store.execute_batch(
            "RETURN 1",
            [],
            batch_size=0,
        )


def _write_json(
    path: Path,
    payload: object,
) -> None:
    """Write one deterministic test artifact."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_company_loader_uses_canonical_metadata(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Company scale loading should preserve canonical metadata."""

    root = tmp_path / "processed" / "sec"

    _write_json(
        root / "AAPL" / "company.json",
        {
            "cik": "0000320193",
            "name": "Apple Inc.",
            "sic": "3571",
            "industry": None,
        },
    )

    _write_json(
        root / "MSFT" / "company.json",
        {
            "cik": "0000789019",
            "name": "Microsoft Corporation",
            "sic": "7372",
            "industry": None,
        },
    )

    monkeypatch.setattr(
        scale_loader,
        "PROCESSED_ROOT",
        root,
    )

    rows = list(scale_loader.iter_company_rows())

    assert [row["ticker"] for row in rows] == [
        "AAPL",
        "MSFT",
    ]

    assert rows[0]["cik"] == "0000320193"

    assert rows[1]["name"] == "Microsoft Corporation"


def test_company_loader_ignores_in_progress_partition(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A publishing marker should hide a company from scale loading."""

    root = tmp_path / "processed" / "sec"

    _write_json(
        root / "AAPL" / "company.json",
        {
            "cik": "0000320193",
            "name": "Apple Inc.",
            "sic": "3571",
            "industry": None,
        },
    )

    _write_json(
        root / "MSFT" / "company.json",
        {
            "cik": "0000789019",
            "name": "Microsoft Corporation",
            "sic": "7372",
            "industry": None,
        },
    )

    (root / "MSFT" / PUBLISHING_MARKER).touch()

    monkeypatch.setattr(
        scale_loader,
        "PROCESSED_ROOT",
        root,
    )

    rows = list(scale_loader.iter_company_rows())

    assert [row["ticker"] for row in rows] == [
        "AAPL",
    ]


def test_fact_loader_processes_company_artifacts_separately(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Fact loading should write each company independently rather than
    materializing a corpus-wide fact list.
    """

    root = tmp_path / "processed" / "sec"

    common = {
        "taxonomy": "us-gaap",
        "concept": "Revenue",
        "unit": "USD",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "fiscal_year": 2024,
        "fiscal_period": "FY",
        "filed_date": "2025-02-01",
        "frame": None,
    }

    _write_json(
        root / "AAPL" / "financial_facts.json",
        [
            {
                **common,
                "fact_id": "fact-aapl",
                "cik": "0000320193",
                "accession_number": ("aapl-filing"),
                "value": 100.0,
            }
        ],
    )

    _write_json(
        root / "MSFT" / "financial_facts.json",
        [
            {
                **common,
                "fact_id": "fact-msft",
                "cik": "0000789019",
                "accession_number": ("msft-filing"),
                "value": 200.0,
            }
        ],
    )

    monkeypatch.setattr(
        scale_loader,
        "PROCESSED_ROOT",
        root,
    )

    calls: list[
        tuple[
            str,
            list[
                dict[
                    str,
                    object,
                ]
            ],
            int,
        ]
    ] = []

    store = Mock()

    def execute_batch(
        query: str,
        rows: Iterable[
            dict[
                str,
                object,
            ]
        ],
        *,
        batch_size: int,
    ) -> int:
        materialized = list(rows)

        calls.append(
            (
                query,
                materialized,
                batch_size,
            )
        )

        return len(materialized)

    store.execute_batch.side_effect = execute_batch

    (
        facts,
        periods,
    ) = scale_loader.load_facts_scaled(
        store,
        batch_size=500,
    )

    assert facts == 2
    assert periods == 2
    assert len(calls) == 4

    assert [call[0] for call in calls] == [
        PERIOD_QUERY,
        FACT_QUERY,
        PERIOD_QUERY,
        FACT_QUERY,
    ]

    assert all(call[2] == 500 for call in calls)

    fact_calls = [call for call in calls if call[0] == FACT_QUERY]

    assert [len(call[1]) for call in fact_calls] == [
        1,
        1,
    ]


def test_fact_loader_ignores_in_progress_partition(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Financial facts from a publishing company must not reach Neo4j."""

    root = tmp_path / "processed" / "sec"

    common = {
        "taxonomy": "us-gaap",
        "concept": "Revenue",
        "unit": "USD",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "fiscal_year": 2024,
        "fiscal_period": "FY",
        "filed_date": "2025-02-01",
        "frame": None,
    }

    _write_json(
        root / "AAPL" / "financial_facts.json",
        [
            {
                **common,
                "fact_id": "fact-aapl",
                "cik": "0000320193",
                "accession_number": ("aapl-filing"),
                "value": 100.0,
            }
        ],
    )

    _write_json(
        root / "MSFT" / "financial_facts.json",
        [
            {
                **common,
                "fact_id": "fact-msft",
                "cik": "0000789019",
                "accession_number": ("msft-filing"),
                "value": 200.0,
            }
        ],
    )

    (root / "MSFT" / PUBLISHING_MARKER).touch()

    monkeypatch.setattr(
        scale_loader,
        "PROCESSED_ROOT",
        root,
    )

    calls: list[
        tuple[
            str,
            list[
                dict[
                    str,
                    object,
                ]
            ],
        ]
    ] = []

    store = Mock()

    def execute_batch(
        query: str,
        rows: Iterable[
            dict[
                str,
                object,
            ]
        ],
        *,
        batch_size: int,
    ) -> int:
        del batch_size

        materialized = list(rows)

        calls.append(
            (
                query,
                materialized,
            )
        )

        return len(materialized)

    store.execute_batch.side_effect = execute_batch

    (
        facts,
        periods,
    ) = scale_loader.load_facts_scaled(
        store,
        batch_size=500,
    )

    assert facts == 1
    assert periods == 1

    fact_calls = [
        rows
        for (
            query,
            rows,
        ) in calls
        if query == FACT_QUERY
    ]

    assert len(fact_calls) == 1

    assert fact_calls[0][0]["fact_id"] == "fact-aapl"
