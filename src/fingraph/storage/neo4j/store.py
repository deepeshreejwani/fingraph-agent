"""Neo4j connection and query adapter."""

import json
from collections.abc import (
    Iterable,
    Mapping,
)
from itertools import islice
from typing import Any

from neo4j import (
    Driver,
    GraphDatabase,
    Query,
)

from fingraph.graph.schema import (
    CONSTRAINT_QUERIES,
    INDEX_QUERIES,
)

DEFAULT_MAX_READ_PAYLOAD_BYTES = 1_048_576


class Neo4jQueryValidationError(RuntimeError):
    """Generated Cypher violated the Neo4j-backed query safety contract."""


class Neo4jResultPayloadError(Neo4jQueryValidationError):
    """Generated Cypher produced a result exceeding the payload safety limit."""


def _serialized_json_size_bytes(
    value: object,
) -> int:
    """Return compact UTF-8 JSON size for one result value."""

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
        default=str,
    )

    return len(serialized.encode("utf-8"))


class Neo4jStore:
    """Thin adapter around the official Neo4j driver."""

    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        read_timeout_seconds: float | None = None,
        connection_timeout_seconds: float = 3.0,
        max_transaction_retry_time_seconds: float = 3.0,
        max_read_payload_bytes: int = DEFAULT_MAX_READ_PAYLOAD_BYTES,
    ) -> None:
        if read_timeout_seconds is not None and read_timeout_seconds <= 0:
            raise ValueError("read_timeout_seconds must be positive.")

        if connection_timeout_seconds <= 0:
            raise ValueError("connection_timeout_seconds must be positive.")

        if max_transaction_retry_time_seconds < 0:
            raise ValueError("max_transaction_retry_time_seconds must be non-negative.")

        if max_read_payload_bytes <= 0:
            raise ValueError("max_read_payload_bytes must be positive.")

        self.database = database
        self.read_timeout_seconds = read_timeout_seconds
        self.max_read_payload_bytes = max_read_payload_bytes

        self.driver: Driver = GraphDatabase.driver(
            uri,
            auth=(
                user,
                password,
            ),
            connection_timeout=(connection_timeout_seconds),
            max_transaction_retry_time=(max_transaction_retry_time_seconds),
        )

    def close(
        self,
    ) -> None:
        """Close underlying Neo4j driver."""

        self.driver.close()

    def __enter__(
        self,
    ) -> "Neo4jStore":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()

    def verify_connectivity(
        self,
    ) -> None:
        """Verify Neo4j connectivity."""

        self.driver.verify_connectivity()

    def health_check(
        self,
    ) -> None:
        """Verify authenticated read access to the configured database."""

        rows = self.execute_read("RETURN 1 AS ok")

        if len(rows) != 1 or rows[0].get("ok") != 1:
            raise RuntimeError("Neo4j health check returned an unexpected result.")

    def initialize_schema(
        self,
    ) -> None:
        """Create required constraints and indexes."""

        for query in CONSTRAINT_QUERIES:
            self.driver.execute_query(
                query,
                database_=self.database,
            )

        for query in INDEX_QUERIES:
            self.driver.execute_query(
                query,
                database_=self.database,
            )

    def execute_write(
        self,
        query: str,
        *,
        parameters: (Mapping[str, Any] | None) = None,
    ) -> None:
        """Execute one parameterized write query."""

        self.driver.execute_query(
            query,
            parameters_=dict(parameters or {}),
            database_=self.database,
        )

    def _read_query(
        self,
        query: str,
    ) -> str | Query:
        """Attach the configured transaction timeout to a read query."""

        if self.read_timeout_seconds is None:
            return query

        return Query(
            query,
            timeout=(self.read_timeout_seconds),
        )

    def explain_read(
        self,
        query: str,
        *,
        parameters: (Mapping[str, Any] | None) = None,
    ) -> None:
        """Parse and plan a query, rejecting anything classified as non-read."""

        explained = f"EXPLAIN\n{query}"

        _, summary, _ = self.driver.execute_query(
            self._read_query(explained),
            parameters_=dict(parameters or {}),
            database_=self.database,
            routing_="r",
        )

        query_type = summary.query_type

        if query_type != "r":
            rendered_type = "unknown" if query_type is None else str(query_type)

            raise Neo4jQueryValidationError(
                "Neo4j EXPLAIN classified "
                "generated Cypher as non-read "
                f"(query_type={rendered_type})."
            )

    def execute_read(
        self,
        query: str,
        *,
        parameters: (Mapping[str, Any] | None) = None,
    ) -> list[
        dict[
            str,
            Any,
        ]
    ]:
        """Execute a read query with a bounded serialized result payload."""

        records, _, _ = self.driver.execute_query(
            self._read_query(query),
            parameters_=dict(parameters or {}),
            database_=self.database,
            routing_="r",
        )

        rows: list[
            dict[
                str,
                Any,
            ]
        ] = []

        # Account for the surrounding JSON array brackets.
        payload_bytes = 2

        for record in records:
            row = record.data()

            row_bytes = _serialized_json_size_bytes(row)

            # Compact JSON arrays use one comma between rows.
            if rows:
                payload_bytes += 1

            payload_bytes += row_bytes

            if payload_bytes > self.max_read_payload_bytes:
                raise Neo4jResultPayloadError(
                    "Neo4j read result exceeded the configured "
                    "serialized payload limit "
                    f"of {self.max_read_payload_bytes} bytes."
                )

            rows.append(row)

        return rows

    def execute_batch(
        self,
        query: str,
        rows: Iterable[
            Mapping[
                str,
                Any,
            ]
        ],
        *,
        batch_size: int = 1_000,
    ) -> int:
        """
        Execute bounded UNWIND transactions.

        Rows are consumed incrementally from the supplied iterable so callers
        do not need to materialize the complete corpus before writing.

        Returns the total number of rows submitted successfully.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        iterator = iter(rows)
        written = 0

        while True:
            batch = list(
                islice(
                    iterator,
                    batch_size,
                )
            )

            if not batch:
                break

            self.execute_write(
                query,
                parameters={"rows": [dict(row) for row in batch]},
            )

            written += len(batch)

        return written
