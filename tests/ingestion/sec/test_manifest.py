"""Tests for batched SEC manifest persistence."""

from __future__ import annotations

import json
from pathlib import Path

from fingraph.ingestion.sec.manifest import (
    Manifest,
    ManifestEntry,
)


def entry(
    resource_id: str,
) -> ManifestEntry:
    """Create one deterministic manifest entry."""

    return ManifestEntry(
        resource_id=resource_id,
        resource_type="10-K",
        cik="0000789019",
        accession_number=(f"0000789019-26-{resource_id[-6:]}"),
        source_url=("https://www.sec.gov/example.htm"),
        local_path=(f"data/raw/sec/MSFT/{resource_id}.html"),
        content_hash="a" * 64,
    )


def test_staged_upsert_does_not_write_until_flush(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"

    manifest = Manifest(path)

    manifest.upsert(
        entry("resource-000001"),
        persist=False,
    )

    assert not path.exists()

    manifest.flush()

    assert path.exists()

    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    assert len(payload) == 1

    assert payload[0]["resource_id"] == "resource-000001"


def test_upsert_many_persists_multiple_entries_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"

    manifest = Manifest(path)

    changed = manifest.upsert_many(
        [
            entry("resource-000001"),
            entry("resource-000002"),
        ]
    )

    assert changed == 2

    reloaded = Manifest(path)

    assert [item.resource_id for item in reloaded.all()] == [
        "resource-000001",
        "resource-000002",
    ]


def test_flush_is_idempotent_when_nothing_changed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"

    manifest = Manifest(path)

    manifest.upsert(entry("resource-000001"))

    before = path.read_bytes()

    manifest.flush()

    after = path.read_bytes()

    assert after == before
