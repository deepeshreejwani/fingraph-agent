"""Content hashing utilities."""

import hashlib


def sha256_text(value: str) -> str:
    """Return a deterministic SHA-256 hex digest for text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Return a deterministic SHA-256 hex digest for bytes."""

    return hashlib.sha256(value).hexdigest()
