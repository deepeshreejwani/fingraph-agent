"""Deterministic identifier utilities."""

import hashlib


def stable_id(namespace: str, *parts: object) -> str:
    """Create a deterministic identifier from a namespace and ordered values."""

    normalized = "|".join(str(part).strip() for part in parts if part is not None)

    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    return f"{namespace}_{digest[:24]}"
