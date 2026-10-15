"""Focused tests for crash-safe SEC ingestion cache behavior."""

import json
from pathlib import Path

import pytest
from pytest import MonkeyPatch
from scripts.ingest_sec import (
    cached_json_is_valid,
    cached_text_if_valid,
    canonical_json_hash,
    write_json,
    write_text_atomic,
)

from fingraph.ingestion.publication import (
    CANONICAL_COMPANY_ARTIFACTS,
    CURRENT_SNAPSHOT_POINTER,
    PUBLISHING_MARKER,
    SNAPSHOT_DIRECTORY,
    publish_company_snapshot,
    published_artifact_paths,
)
from fingraph.utils.hashing import sha256_text


def _snapshot_artifacts(
    *,
    version: int,
) -> dict[str, object]:
    """Build one complete deterministic test company snapshot."""

    return {
        "company.json": {
            "cik": "0000320193",
            "ticker": "AAPL",
            "name": f"Apple v{version}",
            "sic": "3571",
            "industry": None,
        },
        "filings.json": [
            {
                "version": version,
                "accession_number": f"filing-{version}",
            }
        ],
        "chunks.json": [
            {
                "version": version,
                "chunk_id": f"chunk-{version}",
            }
        ],
        "financial_facts.json": [
            {
                "version": version,
                "fact_id": f"fact-{version}",
            }
        ],
    }


def _single_published_artifact(
    *,
    processed_root: Path,
    filename: str,
) -> Path:
    """Return the only published artifact path in a single-company fixture."""

    paths = published_artifact_paths(
        processed_root,
        filename,
    )

    assert len(paths) == 1

    return paths[0]


def test_cached_text_requires_matching_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "filing.html"

    write_text_atomic(
        path,
        "<html>valid</html>",
    )

    expected = sha256_text("<html>valid</html>")

    assert (
        cached_text_if_valid(
            path=path,
            expected_hash=expected,
        )
        == "<html>valid</html>"
    )

    write_text_atomic(
        path,
        "<html>corrupt</html>",
    )

    assert (
        cached_text_if_valid(
            path=path,
            expected_hash=expected,
        )
        is None
    )


def test_cached_text_preserves_crlf_hash_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "filing.html"

    content = "<html>\r\n<body>SEC filing</body>\r\n</html>\r\n"

    expected = sha256_text(content)

    write_text_atomic(
        path,
        content,
    )

    assert (
        cached_text_if_valid(
            path=path,
            expected_hash=expected,
        )
        == content
    )


def test_cached_json_hash_is_format_independent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "facts.json"

    payload = {
        "b": [
            3,
            2,
            1,
        ],
        "a": {
            "value": 5,
        },
    }

    expected = canonical_json_hash(payload)

    path.write_text(
        json.dumps(
            payload,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    assert cached_json_is_valid(
        path=path,
        expected_hash=expected,
    )


def test_atomic_json_writer_replaces_complete_document(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"

    write_json(
        path,
        {
            "version": 1,
        },
    )

    write_json(
        path,
        {
            "version": 2,
            "values": [
                1,
                2,
                3,
            ],
        },
    )

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "version": 2,
        "values": [
            1,
            2,
            3,
        ],
    }

    assert not path.with_suffix(".json.tmp").exists()


def test_publish_company_snapshot_publishes_complete_artifact_set(
    tmp_path: Path,
) -> None:
    processed_root = tmp_path / "processed" / "sec"

    artifacts = _snapshot_artifacts(version=1)

    company_root = publish_company_snapshot(
        processed_root=processed_root,
        ticker=" aapl ",
        artifacts=artifacts,
    )

    assert company_root == (processed_root / "AAPL")

    assert not (company_root / PUBLISHING_MARKER).exists()

    pointer_path = company_root / CURRENT_SNAPSHOT_POINTER

    assert pointer_path.is_file()

    generation = pointer_path.read_text(
        encoding="utf-8",
    ).strip()

    assert generation

    snapshot_root = processed_root / SNAPSHOT_DIRECTORY / generation / "AAPL"

    assert snapshot_root.is_dir()

    for filename in CANONICAL_COMPANY_ARTIFACTS:
        assert (
            json.loads(
                (snapshot_root / filename).read_text(
                    encoding="utf-8",
                )
            )
            == artifacts[filename]
        )


def test_published_artifact_paths_resolve_current_generation(
    tmp_path: Path,
) -> None:
    processed_root = tmp_path / "processed" / "sec"

    artifacts = _snapshot_artifacts(version=1)

    publish_company_snapshot(
        processed_root=processed_root,
        ticker="AAPL",
        artifacts=artifacts,
    )

    chunk_path = _single_published_artifact(
        processed_root=processed_root,
        filename="chunks.json",
    )

    assert chunk_path.parent.name == "AAPL"

    assert SNAPSHOT_DIRECTORY in chunk_path.parts

    assert (
        json.loads(
            chunk_path.read_text(
                encoding="utf-8",
            )
        )
        == artifacts["chunks.json"]
    )


def test_published_artifact_paths_ignore_legacy_in_progress_partition(
    tmp_path: Path,
) -> None:
    processed_root = tmp_path / "processed" / "sec"

    aapl_root = processed_root / "AAPL"

    aapl_root.mkdir(
        parents=True,
    )

    (aapl_root / "chunks.json").write_text(
        "[]",
        encoding="utf-8",
    )

    msft_root = processed_root / "MSFT"

    msft_root.mkdir(
        parents=True,
    )

    (msft_root / "chunks.json").write_text(
        "[]",
        encoding="utf-8",
    )

    (msft_root / PUBLISHING_MARKER).touch()

    paths = published_artifact_paths(
        processed_root,
        "chunks.json",
    )

    assert [path.parent.name for path in paths] == [
        "AAPL",
    ]


def test_generation_backed_snapshot_remains_visible_despite_legacy_marker(
    tmp_path: Path,
) -> None:
    """
    A stale legacy marker must not hide a valid generation-backed snapshot.

    The generation pointer is the publication boundary for the new layout.
    """

    processed_root = tmp_path / "processed" / "sec"

    artifacts = _snapshot_artifacts(version=1)

    company_root = publish_company_snapshot(
        processed_root=processed_root,
        ticker="AAPL",
        artifacts=artifacts,
    )

    (company_root / PUBLISHING_MARKER).touch()

    path = _single_published_artifact(
        processed_root=processed_root,
        filename="chunks.json",
    )

    assert (
        json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
        == artifacts["chunks.json"]
    )


def test_discovered_generation_remains_stable_across_republication(
    tmp_path: Path,
) -> None:
    """
    A reader must retain an immutable snapshot after later publication.

    This is the regression for the discovery/read TOCTOU race: once a reader
    receives a generation path, republishing the ticker must not change the
    contents behind that already-discovered path.
    """

    processed_root = tmp_path / "processed" / "sec"

    first = _snapshot_artifacts(version=1)

    publish_company_snapshot(
        processed_root=processed_root,
        ticker="AAPL",
        artifacts=first,
    )

    discovered_before = _single_published_artifact(
        processed_root=processed_root,
        filename="chunks.json",
    )

    assert (
        json.loads(
            discovered_before.read_text(
                encoding="utf-8",
            )
        )
        == first["chunks.json"]
    )

    second = _snapshot_artifacts(version=2)

    publish_company_snapshot(
        processed_root=processed_root,
        ticker="AAPL",
        artifacts=second,
    )

    discovered_after = _single_published_artifact(
        processed_root=processed_root,
        filename="chunks.json",
    )

    assert discovered_after != discovered_before

    assert (
        json.loads(
            discovered_before.read_text(
                encoding="utf-8",
            )
        )
        == first["chunks.json"]
    )

    assert (
        json.loads(
            discovered_after.read_text(
                encoding="utf-8",
            )
        )
        == second["chunks.json"]
    )


def test_interrupted_publication_preserves_previous_generation(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Failure before pointer publication must leave the previous snapshot visible.

    A fully materialized replacement generation is not public until the
    ``.current`` pointer swap succeeds.
    """

    processed_root = tmp_path / "processed" / "sec"

    original = _snapshot_artifacts(version=1)

    publish_company_snapshot(
        processed_root=processed_root,
        ticker="AAPL",
        artifacts=original,
    )

    old_chunk_path = _single_published_artifact(
        processed_root=processed_root,
        filename="chunks.json",
    )

    company_root = processed_root / "AAPL"

    original_generation = (
        (company_root / CURRENT_SNAPSHOT_POINTER)
        .read_text(
            encoding="utf-8",
        )
        .strip()
    )

    replacement = _snapshot_artifacts(version=2)

    original_replace = Path.replace

    def failing_replace(
        self: Path,
        target: Path,
    ) -> Path:
        if target == (company_root / CURRENT_SNAPSHOT_POINTER):
            raise OSError("simulated pointer publication failure")

        return original_replace(
            self,
            target,
        )

    monkeypatch.setattr(
        Path,
        "replace",
        failing_replace,
    )

    with pytest.raises(
        OSError,
        match="simulated pointer publication failure",
    ):
        publish_company_snapshot(
            processed_root=processed_root,
            ticker="AAPL",
            artifacts=replacement,
        )

    assert (company_root / CURRENT_SNAPSHOT_POINTER).read_text(
        encoding="utf-8",
    ).strip() == original_generation

    visible_chunk_path = _single_published_artifact(
        processed_root=processed_root,
        filename="chunks.json",
    )

    assert visible_chunk_path == old_chunk_path

    assert (
        json.loads(
            visible_chunk_path.read_text(
                encoding="utf-8",
            )
        )
        == original["chunks.json"]
    )


def test_failed_pointer_publication_cleans_unpublished_generation(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A failed pointer swap must not leave its unpublished generation behind."""

    processed_root = tmp_path / "processed" / "sec"

    publish_company_snapshot(
        processed_root=processed_root,
        ticker="AAPL",
        artifacts=_snapshot_artifacts(version=1),
    )

    company_root = processed_root / "AAPL"

    snapshots_root = processed_root / SNAPSHOT_DIRECTORY

    generations_before = {path.name for path in snapshots_root.iterdir() if path.is_dir()}

    original_replace = Path.replace

    def failing_replace(
        self: Path,
        target: Path,
    ) -> Path:
        if target == (company_root / CURRENT_SNAPSHOT_POINTER):
            raise OSError("simulated pointer publication failure")

        return original_replace(
            self,
            target,
        )

    monkeypatch.setattr(
        Path,
        "replace",
        failing_replace,
    )

    with pytest.raises(
        OSError,
        match="simulated pointer publication failure",
    ):
        publish_company_snapshot(
            processed_root=processed_root,
            ticker="AAPL",
            artifacts=_snapshot_artifacts(version=2),
        )

    generations_after = {path.name for path in snapshots_root.iterdir() if path.is_dir()}

    assert generations_after == generations_before


def test_successful_republication_replaces_current_generation(
    tmp_path: Path,
) -> None:
    processed_root = tmp_path / "processed" / "sec"

    first = _snapshot_artifacts(version=1)

    company_root = publish_company_snapshot(
        processed_root=processed_root,
        ticker="AAPL",
        artifacts=first,
    )

    first_generation = (
        (company_root / CURRENT_SNAPSHOT_POINTER)
        .read_text(
            encoding="utf-8",
        )
        .strip()
    )

    second = _snapshot_artifacts(version=2)

    publish_company_snapshot(
        processed_root=processed_root,
        ticker="AAPL",
        artifacts=second,
    )

    second_generation = (
        (company_root / CURRENT_SNAPSHOT_POINTER)
        .read_text(
            encoding="utf-8",
        )
        .strip()
    )

    assert second_generation != first_generation

    current_chunk_path = _single_published_artifact(
        processed_root=processed_root,
        filename="chunks.json",
    )

    assert (
        json.loads(
            current_chunk_path.read_text(
                encoding="utf-8",
            )
        )
        == second["chunks.json"]
    )


def test_successful_republication_recovers_legacy_interrupted_partition(
    tmp_path: Path,
) -> None:
    processed_root = tmp_path / "processed" / "sec"

    company_root = processed_root / "AAPL"

    company_root.mkdir(
        parents=True,
    )

    (company_root / PUBLISHING_MARKER).touch()

    artifacts = _snapshot_artifacts(version=3)

    publish_company_snapshot(
        processed_root=processed_root,
        ticker="AAPL",
        artifacts=artifacts,
    )

    assert not (company_root / PUBLISHING_MARKER).exists()

    assert (company_root / CURRENT_SNAPSHOT_POINTER).is_file()

    chunk_path = _single_published_artifact(
        processed_root=processed_root,
        filename="chunks.json",
    )

    assert chunk_path.parent.name == "AAPL"

    assert (
        json.loads(
            chunk_path.read_text(
                encoding="utf-8",
            )
        )
        == artifacts["chunks.json"]
    )


def test_historical_direct_layout_remains_backward_compatible(
    tmp_path: Path,
) -> None:
    """Partitions created before generation publication remain readable."""

    processed_root = tmp_path / "processed" / "sec"

    company_root = processed_root / "AAPL"

    company_root.mkdir(
        parents=True,
    )

    chunks = [
        {
            "version": 1,
            "chunk_id": "legacy-chunk",
        }
    ]

    (company_root / "chunks.json").write_text(
        json.dumps(chunks),
        encoding="utf-8",
    )

    paths = published_artifact_paths(
        processed_root,
        "chunks.json",
    )

    assert paths == [
        company_root / "chunks.json",
    ]

    assert (
        json.loads(
            paths[0].read_text(
                encoding="utf-8",
            )
        )
        == chunks
    )


def test_invalid_generation_pointer_fails_closed(
    tmp_path: Path,
) -> None:
    """Malformed pointer contents must never escape the snapshot directory."""

    processed_root = tmp_path / "processed" / "sec"

    company_root = processed_root / "AAPL"

    company_root.mkdir(
        parents=True,
    )

    (company_root / CURRENT_SNAPSHOT_POINTER).write_text(
        "../outside\n",
        encoding="utf-8",
    )

    assert (
        published_artifact_paths(
            processed_root,
            "chunks.json",
        )
        == []
    )


def test_missing_generation_target_fails_closed(
    tmp_path: Path,
) -> None:
    """A pointer to a missing generation must not expose legacy files."""

    processed_root = tmp_path / "processed" / "sec"

    company_root = processed_root / "AAPL"

    company_root.mkdir(
        parents=True,
    )

    (company_root / "chunks.json").write_text(
        "[]",
        encoding="utf-8",
    )

    (company_root / CURRENT_SNAPSHOT_POINTER).write_text(
        "missing-generation\n",
        encoding="utf-8",
    )

    assert (
        published_artifact_paths(
            processed_root,
            "chunks.json",
        )
        == []
    )


def test_internal_snapshot_directory_is_not_discovered_as_company(
    tmp_path: Path,
) -> None:
    """Internal publication directories must never appear as ticker partitions."""

    processed_root = tmp_path / "processed" / "sec"

    publish_company_snapshot(
        processed_root=processed_root,
        ticker="AAPL",
        artifacts=_snapshot_artifacts(version=1),
    )

    paths = published_artifact_paths(
        processed_root,
        "chunks.json",
    )

    assert len(paths) == 1

    assert paths[0].parent.name == "AAPL"


def test_publish_company_snapshot_rejects_incomplete_artifact_set(
    tmp_path: Path,
) -> None:
    processed_root = tmp_path / "processed" / "sec"

    artifacts = _snapshot_artifacts(version=1)

    del artifacts["financial_facts.json"]

    with pytest.raises(
        ValueError,
        match="missing=financial_facts.json",
    ):
        publish_company_snapshot(
            processed_root=processed_root,
            ticker="AAPL",
            artifacts=artifacts,
        )

    assert not (processed_root / "AAPL").exists()
