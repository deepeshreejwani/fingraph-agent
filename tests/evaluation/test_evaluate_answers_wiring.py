"""Regression coverage for answer-evaluator storage wiring."""

from types import SimpleNamespace
from typing import Any

import scripts.evaluate_answers as evaluate_answers


class FakeQdrantStore:
    """Capture evaluator Qdrant constructor arguments."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeNeo4jStore:
    """Capture evaluator Neo4j constructor arguments."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        qdrant_url="https://qdrant.example",
        qdrant_api_key="qdrant-secret",
        qdrant_collection="fingraph-test",
        neo4j_uri="neo4j+s://neo4j.example",
        neo4j_user="neo4j-user",
        neo4j_password="neo4j-secret",
        neo4j_database="fingraph",
        neo4j_read_timeout_seconds=11.0,
        neo4j_connection_timeout_seconds=4.0,
        neo4j_max_transaction_retry_time_seconds=5.0,
    )


def test_answer_evaluator_passes_qdrant_api_key(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        evaluate_answers,
        "QdrantChunkStore",
        FakeQdrantStore,
    )

    store = evaluate_answers._build_qdrant_store(_settings())

    assert isinstance(store, FakeQdrantStore)
    assert store.kwargs == {
        "url": "https://qdrant.example",
        "api_key": "qdrant-secret",
        "collection_name": "fingraph-test",
    }


def test_answer_evaluator_passes_neo4j_timeout_settings(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        evaluate_answers,
        "Neo4jStore",
        FakeNeo4jStore,
    )

    store = evaluate_answers._build_neo4j_store(_settings())

    assert isinstance(store, FakeNeo4jStore)
    assert store.kwargs == {
        "uri": "neo4j+s://neo4j.example",
        "user": "neo4j-user",
        "password": "neo4j-secret",
        "database": "fingraph",
        "read_timeout_seconds": 11.0,
        "connection_timeout_seconds": 4.0,
        "max_transaction_retry_time_seconds": 5.0,
    }
