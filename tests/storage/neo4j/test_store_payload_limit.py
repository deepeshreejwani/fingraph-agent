"""Regression tests for Neo4j read-result payload limits."""

from typing import Any, cast
from unittest.mock import Mock

import pytest

from fingraph.storage.neo4j import (
    Neo4jResultPayloadError,
    Neo4jStore,
)


class FakeRecord:
    """Minimal Neo4j record test double."""

    def __init__(
        self,
        payload: dict[str, Any],
    ) -> None:
        self.payload = payload

    def data(
        self,
    ) -> dict[str, Any]:
        return dict(self.payload)


def _store_with_records(
    *,
    records: list[FakeRecord],
    max_read_payload_bytes: int,
) -> Neo4jStore:
    """Build a Neo4jStore test double without opening a real connection."""

    store = object.__new__(Neo4jStore)

    driver = Mock()

    driver.execute_query.return_value = (
        records,
        Mock(),
        [],
    )

    cast(
        Any,
        store,
    ).driver = driver

    cast(
        Any,
        store,
    ).database = "neo4j"

    cast(
        Any,
        store,
    ).read_timeout_seconds = None

    cast(
        Any,
        store,
    ).max_read_payload_bytes = max_read_payload_bytes

    return store


def test_execute_read_accepts_payload_within_limit() -> None:
    store = _store_with_records(
        records=[
            FakeRecord(
                {
                    "ticker": "MSFT",
                }
            ),
            FakeRecord(
                {
                    "ticker": "AAPL",
                }
            ),
        ],
        max_read_payload_bytes=1_024,
    )

    rows = store.execute_read("MATCH (c:Company) RETURN c.ticker AS ticker LIMIT 25")

    assert rows == [
        {
            "ticker": "MSFT",
        },
        {
            "ticker": "AAPL",
        },
    ]


def test_execute_read_rejects_payload_exceeding_limit() -> None:
    store = _store_with_records(
        records=[
            FakeRecord(
                {
                    "text": "x" * 200,
                }
            )
        ],
        max_read_payload_bytes=64,
    )

    with pytest.raises(
        Neo4jResultPayloadError,
        match="serialized payload limit",
    ):
        store.execute_read("MATCH (s:Section) RETURN s.text AS text LIMIT 25")


def test_execute_read_counts_payload_across_multiple_rows() -> None:
    store = _store_with_records(
        records=[
            FakeRecord(
                {
                    "value": "x" * 30,
                }
            ),
            FakeRecord(
                {
                    "value": "y" * 30,
                }
            ),
        ],
        max_read_payload_bytes=60,
    )

    with pytest.raises(
        Neo4jResultPayloadError,
        match="serialized payload limit",
    ):
        store.execute_read("MATCH (s:Section) RETURN s.text AS value LIMIT 25")


def test_execute_read_counts_utf8_bytes_not_python_characters() -> None:
    store = _store_with_records(
        records=[
            FakeRecord(
                {
                    "text": "₹" * 20,
                }
            )
        ],
        max_read_payload_bytes=50,
    )

    with pytest.raises(
        Neo4jResultPayloadError,
        match="serialized payload limit",
    ):
        store.execute_read("MATCH (s:Section) RETURN s.text AS text LIMIT 25")


def test_execute_read_uses_read_routing_and_parameters() -> None:
    store = _store_with_records(
        records=[
            FakeRecord(
                {
                    "ticker": "MSFT",
                }
            )
        ],
        max_read_payload_bytes=1_024,
    )

    store.execute_read(
        "MATCH (c:Company) WHERE c.ticker = $ticker RETURN c.ticker AS ticker LIMIT 25",
        parameters={
            "ticker": "MSFT",
        },
    )

    cast(
        Any,
        store,
    ).driver.execute_query.assert_called_once_with(
        ("MATCH (c:Company) WHERE c.ticker = $ticker RETURN c.ticker AS ticker LIMIT 25"),
        parameters_={
            "ticker": "MSFT",
        },
        database_="neo4j",
        routing_="r",
    )
