"""Crash-safe publication of canonical processed company snapshots."""

from __future__ import annotations

import json
import shutil
from collections.abc import (
    Iterator,
    Mapping,
)
from pathlib import Path
from uuid import uuid4

PUBLISHING_MARKER = ".publishing"

CURRENT_SNAPSHOT_POINTER = ".current"

SNAPSHOT_DIRECTORY = ".snapshots"

CANONICAL_COMPANY_ARTIFACTS = (
    "company.json",
    "filings.json",
    "chunks.json",
    "financial_facts.json",
)


def normalize_ticker(
    ticker: str,
) -> str:
    """Normalize and validate a ticker used as a filesystem partition."""

    normalized = ticker.strip().upper()

    if not normalized:
        raise ValueError("ticker must not be empty.")

    if normalized in {
        ".",
        "..",
    }:
        raise ValueError("ticker must not be a relative path component.")

    if "/" in normalized or "\\" in normalized:
        raise ValueError("ticker must not contain path separators.")

    return normalized


def _snapshot_generation_is_valid(
    generation: str,
) -> bool:
    """Return whether a stored generation ID is a safe path component."""

    if not generation:
        return False

    if generation in {
        ".",
        "..",
    }:
        return False

    if "/" in generation or "\\" in generation:
        return False

    return Path(generation).name == generation


def _current_snapshot_root(
    company_root: Path,
) -> Path | None:
    """
    Resolve one immutable published company snapshot.

    Generation-backed publications are resolved through an atomically replaced
    pointer file. Historical direct-layout partitions remain readable when no
    generation pointer exists.

    A legacy publication marker still hides direct-layout snapshots created by
    earlier publication implementations.
    """

    if not company_root.is_dir():
        return None

    pointer_path = company_root / CURRENT_SNAPSHOT_POINTER

    if pointer_path.is_file():
        try:
            generation = pointer_path.read_text(encoding="utf-8").strip()

        except OSError:
            return None

        if not _snapshot_generation_is_valid(generation):
            return None

        snapshot_root = company_root.parent / SNAPSHOT_DIRECTORY / generation / company_root.name

        if snapshot_root.is_dir():
            return snapshot_root

        return None

    if (company_root / PUBLISHING_MARKER).exists():
        return None

    return company_root


def company_snapshot_is_published(
    company_root: Path,
) -> bool:
    """
    Return whether a company partition has a readable published snapshot.

    Generation-backed publications resolve to immutable directories selected by
    an atomically replaced pointer. Historical direct-layout partitions remain
    supported for backward compatibility.
    """

    return _current_snapshot_root(company_root) is not None


def iter_published_company_roots(
    processed_root: Path,
) -> Iterator[Path]:
    """
    Yield immutable published company snapshot roots in ticker order.

    Each generation-backed company pointer is resolved exactly once. The
    returned path therefore remains tied to that immutable generation even if
    the company is republished after discovery.

    Hidden internal directories such as ``.snapshots`` are never interpreted
    as company partitions.
    """

    if not processed_root.exists():
        return

    company_roots = sorted(
        (
            path
            for path in processed_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=lambda path: (
            path.name.upper(),
            path.name,
        ),
    )

    for company_root in company_roots:
        snapshot_root = _current_snapshot_root(company_root)

        if snapshot_root is not None:
            yield snapshot_root


def published_artifact_paths(
    processed_root: Path,
    filename: str,
) -> list[Path]:
    """
    Return one artifact type from immutable published company snapshots.

    Generation-backed paths remain valid for the lifetime of the generation,
    preventing publication from changing the contents behind a path that a
    reader has already discovered.
    """

    if not filename:
        raise ValueError("filename must not be empty.")

    if Path(filename).name != filename:
        raise ValueError("filename must be a single path component.")

    return [
        company_root / filename
        for company_root in iter_published_company_roots(processed_root)
        if (company_root / filename).is_file()
    ]


def _serialize_json(
    payload: object,
) -> bytes:
    """Serialize one canonical artifact deterministically."""

    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        default=str,
    ).encode("utf-8")


def _validate_artifact_set(
    artifacts: Mapping[
        str,
        object,
    ],
) -> None:
    """Require exactly the canonical company snapshot artifact set."""

    supplied = set(artifacts)

    required = set(CANONICAL_COMPANY_ARTIFACTS)

    missing = sorted(required - supplied)

    unexpected = sorted(supplied - required)

    if not missing and not unexpected:
        return

    details: list[str] = []

    if missing:
        details.append("missing=" + ",".join(missing))

    if unexpected:
        details.append("unexpected=" + ",".join(unexpected))

    raise ValueError("Invalid canonical company artifact set: " + "; ".join(details))


def _write_snapshot_pointer(
    *,
    company_root: Path,
    generation: str,
) -> None:
    """Atomically publish one immutable generation pointer."""

    pointer_path = company_root / CURRENT_SNAPSHOT_POINTER

    temporary_pointer = company_root / (
        f".{CURRENT_SNAPSHOT_POINTER.lstrip('.')}.{uuid4().hex}.tmp"
    )

    try:
        temporary_pointer.write_text(
            generation + "\n",
            encoding="utf-8",
        )

        temporary_pointer.replace(pointer_path)

    finally:
        temporary_pointer.unlink(missing_ok=True)


def publish_company_snapshot(
    *,
    processed_root: Path,
    ticker: str,
    artifacts: Mapping[
        str,
        object,
    ],
) -> Path:
    """
    Atomically publish one complete processed-company generation.

    The complete artifact set is first materialized outside the canonical
    processed root. It is then moved into an immutable generation directory.

    The visible company snapshot changes only when ``.current`` is atomically
    replaced with the new generation ID. Readers that resolved the previous
    generation before that swap continue reading the complete previous
    generation. Readers resolving afterwards receive the complete new
    generation.

    Publication failures before the pointer swap therefore leave the previously
    published snapshot visible and unchanged.

    Historical direct-layout partitions remain readable until their first
    successful generation-backed publication. A stale legacy ``.publishing``
    marker is removed only after a new generation becomes visible.
    """

    normalized_ticker = normalize_ticker(ticker)

    _validate_artifact_set(artifacts)

    processed_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_root = processed_root.parent / f".{processed_root.name}_staging"

    staging_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    generation = uuid4().hex

    staging_company_root = staging_root / (f"{normalized_ticker}-{generation}")

    staging_company_root.mkdir()

    company_root = processed_root / normalized_ticker

    generation_parent = processed_root / SNAPSHOT_DIRECTORY / generation

    generation_company_root = generation_parent / normalized_ticker

    generation_installed = False
    pointer_published = False

    try:
        for filename in CANONICAL_COMPANY_ARTIFACTS:
            staged_path = staging_company_root / filename

            staged_path.write_bytes(_serialize_json(artifacts[filename]))

        generation_parent.mkdir(
            parents=True,
            exist_ok=False,
        )

        staging_company_root.replace(generation_company_root)

        generation_installed = True

        company_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        _write_snapshot_pointer(
            company_root=company_root,
            generation=generation,
        )

        pointer_published = True

        (company_root / PUBLISHING_MARKER).unlink(missing_ok=True)

        return company_root

    finally:
        shutil.rmtree(
            staging_company_root,
            ignore_errors=True,
        )

        if generation_installed and not pointer_published:
            shutil.rmtree(
                generation_parent,
                ignore_errors=True,
            )
