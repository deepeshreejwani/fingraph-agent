"""Canonical FinGraph evaluation failure taxonomy."""

from typing import Literal

FailureCategory = Literal[
    "route_mismatch",
    "graph_evidence_missing",
    "vector_evidence_missing",
    "graph_value_mismatch",
    "vector_metadata_mismatch",
    "grounding_failure",
    "execution_failure",
    "unsupported_question",
]


def classify_failures(
    *,
    route_passed: bool,
    graph_passed: bool,
    vector_passed: bool,
    values_passed: bool,
    metadata_passed: bool,
    grounding_passed: bool,
) -> list[FailureCategory]:
    """Return deterministic failure categories for one evaluation case."""

    failures: list[FailureCategory] = []

    if not route_passed:
        failures.append("route_mismatch")

    if not graph_passed:
        failures.append("graph_evidence_missing")

    if not vector_passed:
        failures.append("vector_evidence_missing")

    if not values_passed:
        failures.append("graph_value_mismatch")

    if not metadata_passed:
        failures.append("vector_metadata_mismatch")

    if not grounding_passed:
        failures.append("grounding_failure")

    return failures
