"""Tests for published-corpus discovery in retrieval indexing."""

from pathlib import Path

import pytest
import scripts.build_retrieval_index as retrieval_index
from pytest import MonkeyPatch

from fingraph.ingestion.publication import (
    PUBLISHING_MARKER,
)


def _write_chunks(
    *,
    root: Path,
    ticker: str,
) -> Path:
    """Create one minimal chunks artifact for discovery tests."""

    path = root / ticker / "chunks.json"

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "[]",
        encoding="utf-8",
    )

    return path


def test_discovery_returns_only_published_chunk_partitions(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """In-progress processed snapshots must not be discovered for indexing."""

    root = tmp_path / "processed" / "sec"

    aapl_path = _write_chunks(
        root=root,
        ticker="AAPL",
    )

    _write_chunks(
        root=root,
        ticker="MSFT",
    )

    (root / "MSFT" / PUBLISHING_MARKER).touch()

    monkeypatch.setattr(
        retrieval_index,
        "PROCESSED_ROOT",
        root,
    )

    discovered = retrieval_index.discover_chunk_files()

    assert discovered == [
        (
            "AAPL",
            aapl_path,
        )
    ]


def test_requested_in_progress_ticker_is_reported_as_unavailable(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Explicit ticker requests must fail closed while publication is incomplete."""

    root = tmp_path / "processed" / "sec"

    _write_chunks(
        root=root,
        ticker="MSFT",
    )

    (root / "MSFT" / PUBLISHING_MARKER).touch()

    monkeypatch.setattr(
        retrieval_index,
        "PROCESSED_ROOT",
        root,
    )

    with pytest.raises(
        SystemExit,
        match=("No published processed chunks found for: MSFT"),
    ):
        retrieval_index.discover_chunk_files(("msft",))


def test_requested_published_ticker_is_discovered_normally(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Published partitions should preserve explicit ticker discovery behavior."""

    root = tmp_path / "processed" / "sec"

    msft_path = _write_chunks(
        root=root,
        ticker="MSFT",
    )

    monkeypatch.setattr(
        retrieval_index,
        "PROCESSED_ROOT",
        root,
    )

    discovered = retrieval_index.discover_chunk_files((" msft ",))

    assert discovered == [
        (
            "MSFT",
            msft_path,
        )
    ]
