"""Persistent manifest for downloaded SEC artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ManifestEntry(BaseModel):
    """One locally materialized SEC resource."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    resource_id: str
    resource_type: str
    cik: str
    accession_number: str | None = None
    source_url: str
    local_path: str
    content_hash: str
    downloaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Manifest:
    """Thread-safe JSON-backed resource manifest."""

    def __init__(
        self,
        path: Path,
    ) -> None:
        self.path = path

        self._lock = RLock()

        self._entries = self._load()

        self._dirty = False

    def get(
        self,
        resource_id: str,
    ) -> ManifestEntry | None:
        """Return an entry by deterministic resource ID."""

        with self._lock:
            return self._entries.get(resource_id)

    def upsert(
        self,
        entry: ManifestEntry,
        *,
        persist: bool = True,
    ) -> None:
        """
        Insert or replace one entry.

        persist=False stages the mutation in memory. This is used by parallel
        ingestion workers so the coordinator can commit several changes with
        one atomic manifest write.
        """

        with self._lock:
            self._entries[entry.resource_id] = entry
            self._dirty = True

            if persist:
                self._write_locked()

    def upsert_many(
        self,
        entries: Iterable[ManifestEntry],
        *,
        persist: bool = True,
    ) -> int:
        """Insert or replace multiple entries with at most one disk write."""

        changed = 0

        with self._lock:
            for entry in entries:
                existing = self._entries.get(entry.resource_id)

                if existing == entry:
                    continue

                self._entries[entry.resource_id] = entry
                self._dirty = True
                changed += 1

            if persist and self._dirty:
                self._write_locked()

        return changed

    def flush(
        self,
    ) -> None:
        """Atomically persist staged changes when the manifest is dirty."""

        with self._lock:
            if not self._dirty:
                return

            self._write_locked()

    def all(
        self,
    ) -> list[ManifestEntry]:
        """Return all manifest entries."""

        with self._lock:
            return sorted(
                self._entries.values(),
                key=lambda entry: entry.resource_id,
            )

    def _load(
        self,
    ) -> dict[str, ManifestEntry]:
        if not self.path.exists():
            return {}

        raw = json.loads(
            self.path.read_text(
                encoding="utf-8",
            )
        )

        if not isinstance(raw, list):
            raise ValueError(f"Invalid manifest structure: {self.path}")

        entries = [ManifestEntry.model_validate(item) for item in raw]

        return {entry.resource_id: entry for entry in entries}

    def _write_locked(
        self,
    ) -> None:
        """
        Persist a complete manifest atomically.

        Caller must hold self._lock.
        """

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        ordered = sorted(
            self._entries.values(),
            key=lambda entry: entry.resource_id,
        )

        payload: list[dict[str, Any]] = [entry.model_dump(mode="json") for entry in ordered]

        temporary = self.path.with_suffix(self.path.suffix + ".tmp")

        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        temporary.replace(self.path)

        self._dirty = False
