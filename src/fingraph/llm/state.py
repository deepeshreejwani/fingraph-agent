"""Persistent resumable state for LLM requests."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fingraph.llm.usage import LLMUsage, usage_from_counts


@dataclass(frozen=True)
class LLMRequestRecord:
    """Persisted state for one deterministic LLM request."""

    request_id: str
    status: str
    attempts: int
    response_json: str | None
    last_error: str | None
    model: str | None = None
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    input_cost_usd: float | None = None
    cached_input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    total_cost_usd: float | None = None
    execution_owner: str | None = None
    lease_expires_at: str | None = None

    def usage(self) -> LLMUsage | None:
        """Return persisted usage when this request has token accounting."""

        if (
            self.model is None
            or self.input_tokens is None
            or self.cached_input_tokens is None
            or self.output_tokens is None
        ):
            return None

        return usage_from_counts(
            model=self.model,
            input_tokens=self.input_tokens,
            cached_input_tokens=self.cached_input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
        )


class LLMStateStore:
    """SQLite-backed durable LLM request checkpoint store."""

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize()

    def _connect(
        self,
    ) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=30.0,
        )

        connection.row_factory = sqlite3.Row

        return connection

    @staticmethod
    def _now_datetime() -> datetime:
        return datetime.now(UTC)

    @classmethod
    def _now(cls) -> str:
        return cls._now_datetime().isoformat()

    @classmethod
    def _lease_expiry(
        cls,
        lease_seconds: float,
    ) -> str:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive.")

        return (cls._now_datetime() + timedelta(seconds=lease_seconds)).isoformat()

    def _initialize(
        self,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_requests (
                    request_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    input_hash TEXT NOT NULL,

                    status TEXT NOT NULL
                        CHECK (
                            status IN (
                                'pending',
                                'running',
                                'succeeded',
                                'failed'
                            )
                        ),

                    attempts INTEGER NOT NULL DEFAULT 0,

                    request_json TEXT NOT NULL,
                    response_json TEXT,
                    last_error TEXT,

                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_llm_requests_status
                ON llm_requests(status)
                """
            )

            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(llm_requests)").fetchall()
            }

            migrations = {
                "input_tokens": "INTEGER",
                "cached_input_tokens": "INTEGER",
                "output_tokens": "INTEGER",
                "total_tokens": "INTEGER",
                "input_cost_usd": "REAL",
                "cached_input_cost_usd": "REAL",
                "output_cost_usd": "REAL",
                "total_cost_usd": "REAL",
                "execution_owner": "TEXT",
                "lease_expires_at": "TEXT",
            }

            for name, sql_type in migrations.items():
                if name in columns:
                    continue

                connection.execute(f"ALTER TABLE llm_requests ADD COLUMN {name} {sql_type}")

    def get(
        self,
        request_id: str,
    ) -> LLMRequestRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    request_id,
                    status,
                    attempts,
                    response_json,
                    last_error,
                    model,
                    input_tokens,
                    cached_input_tokens,
                    output_tokens,
                    total_tokens,
                    input_cost_usd,
                    cached_input_cost_usd,
                    output_cost_usd,
                    total_cost_usd,
                    execution_owner,
                    lease_expires_at
                FROM llm_requests
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()

        if row is None:
            return None

        return LLMRequestRecord(
            request_id=row["request_id"],
            status=row["status"],
            attempts=int(row["attempts"]),
            response_json=row["response_json"],
            last_error=row["last_error"],
            model=row["model"],
            input_tokens=row["input_tokens"],
            cached_input_tokens=row["cached_input_tokens"],
            output_tokens=row["output_tokens"],
            total_tokens=row["total_tokens"],
            input_cost_usd=row["input_cost_usd"],
            cached_input_cost_usd=row["cached_input_cost_usd"],
            output_cost_usd=row["output_cost_usd"],
            total_cost_usd=row["total_cost_usd"],
            execution_owner=row["execution_owner"],
            lease_expires_at=row["lease_expires_at"],
        )

    def register(
        self,
        *,
        request_id: str,
        task_type: str,
        provider: str,
        model: str,
        prompt_version: str,
        input_hash: str,
        request_payload: dict[
            str,
            Any,
        ],
    ) -> None:
        now = self._now()

        request_json = json.dumps(
            request_payload,
            sort_keys=True,
            separators=(",", ":"),
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO llm_requests (
                    request_id,
                    task_type,
                    provider,
                    model,
                    prompt_version,
                    input_hash,
                    status,
                    attempts,
                    request_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)
                """,
                (
                    request_id,
                    task_type,
                    provider,
                    model,
                    prompt_version,
                    input_hash,
                    request_json,
                    now,
                    now,
                ),
            )

    def claim_execution(
        self,
        *,
        request_id: str,
        owner_id: str,
        lease_seconds: float = 120.0,
    ) -> bool:
        """
        Atomically claim one logical provider execution.

        Pending work may be claimed immediately. Running work may be reclaimed
        only when its durable lease has expired (or when a legacy running row
        has no lease metadata).
        """

        if not owner_id.strip():
            raise ValueError("owner_id must be non-empty.")

        now = self._now()
        lease_expires_at = self._lease_expiry(lease_seconds)

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE llm_requests
                SET
                    status = 'running',
                    execution_owner = ?,
                    lease_expires_at = ?,
                    updated_at = ?
                WHERE request_id = ?
                  AND (
                      status = 'pending'
                      OR (
                          status = 'running'
                          AND (
                              execution_owner = ?
                              OR lease_expires_at IS NULL
                              OR lease_expires_at <= ?
                          )
                      )
                  )
                """,
                (
                    owner_id,
                    lease_expires_at,
                    now,
                    request_id,
                    owner_id,
                    now,
                ),
            )

            return cursor.rowcount == 1

    def begin_attempt(
        self,
        request_id: str,
        *,
        owner_id: str | None = None,
        lease_seconds: float = 120.0,
    ) -> bool:
        """
        Start one provider attempt and optionally renew an execution lease.

        owner_id=None preserves the legacy helper behavior.
        Production provider paths pass owner_id.
        """

        now = self._now()

        if owner_id is None:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE llm_requests
                    SET
                        status = 'running',
                        attempts = attempts + 1,
                        updated_at = ?
                    WHERE request_id = ?
                    """,
                    (
                        now,
                        request_id,
                    ),
                )

                return cursor.rowcount == 1

        if not owner_id.strip():
            raise ValueError("owner_id must be non-empty.")

        lease_expires_at = self._lease_expiry(lease_seconds)

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE llm_requests
                SET
                    attempts = attempts + 1,
                    lease_expires_at = ?,
                    updated_at = ?
                WHERE request_id = ?
                  AND status = 'running'
                  AND execution_owner = ?
                """,
                (
                    lease_expires_at,
                    now,
                    request_id,
                    owner_id,
                ),
            )

            return cursor.rowcount == 1

    def mark_succeeded(
        self,
        *,
        request_id: str,
        response_json: str,
        usage: LLMUsage | None = None,
        owner_id: str | None = None,
    ) -> bool:
        parameters: tuple[Any, ...] = (
            response_json,
            usage.input_tokens if usage is not None else None,
            usage.cached_input_tokens if usage is not None else None,
            usage.output_tokens if usage is not None else None,
            usage.total_tokens if usage is not None else None,
            usage.input_cost_usd if usage is not None else None,
            usage.cached_input_cost_usd if usage is not None else None,
            usage.output_cost_usd if usage is not None else None,
            usage.total_cost_usd if usage is not None else None,
            self._now(),
            request_id,
        )

        owner_clause = ""

        if owner_id is not None:
            if not owner_id.strip():
                raise ValueError("owner_id must be non-empty.")

            owner_clause = """
                  AND status = 'running'
                  AND execution_owner = ?
            """

            parameters = (*parameters, owner_id)

        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE llm_requests
                SET
                    status = 'succeeded',
                    response_json = ?,
                    last_error = NULL,
                    input_tokens = ?,
                    cached_input_tokens = ?,
                    output_tokens = ?,
                    total_tokens = ?,
                    input_cost_usd = ?,
                    cached_input_cost_usd = ?,
                    output_cost_usd = ?,
                    total_cost_usd = ?,
                    execution_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE request_id = ?
                {owner_clause}
                """,
                parameters,
            )

            return cursor.rowcount == 1

    def mark_pending(
        self,
        *,
        request_id: str,
        error: str,
        owner_id: str | None = None,
        expected_status: str | None = None,
    ) -> bool:
        """
        Return interrupted work to resumable state.

        Production execution passes owner_id. Pre-execution reset paths may
        instead use expected_status so stale callers cannot clear a newer claim.
        """

        if owner_id is not None and expected_status is not None:
            raise ValueError("owner_id and expected_status cannot be used together.")

        parameters: tuple[Any, ...] = (
            error,
            self._now(),
            request_id,
        )

        condition_clause = ""

        if owner_id is not None:
            if not owner_id.strip():
                raise ValueError("owner_id must be non-empty.")

            condition_clause = """
                  AND status = 'running'
                  AND execution_owner = ?
            """

            parameters = (*parameters, owner_id)

        elif expected_status is not None:
            if expected_status not in {
                "pending",
                "running",
                "succeeded",
                "failed",
            }:
                raise ValueError("expected_status is invalid.")

            condition_clause = """
                  AND status = ?
            """

            parameters = (*parameters, expected_status)

        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE llm_requests
                SET
                    status = 'pending',
                    last_error = ?,
                    execution_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE request_id = ?
                {condition_clause}
                """,
                parameters,
            )

            return cursor.rowcount == 1

    def mark_failed(
        self,
        *,
        request_id: str,
        error: str,
        owner_id: str | None = None,
    ) -> bool:
        parameters: tuple[Any, ...] = (
            error,
            self._now(),
            request_id,
        )

        owner_clause = ""

        if owner_id is not None:
            if not owner_id.strip():
                raise ValueError("owner_id must be non-empty.")

            owner_clause = """
                  AND status = 'running'
                  AND execution_owner = ?
            """

            parameters = (*parameters, owner_id)

        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE llm_requests
                SET
                    status = 'failed',
                    last_error = ?,
                    execution_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE request_id = ?
                {owner_clause}
                """,
                parameters,
            )

            return cursor.rowcount == 1
