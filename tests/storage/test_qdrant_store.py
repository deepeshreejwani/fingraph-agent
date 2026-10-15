"""Tests for Qdrant storage configuration and indexing behavior."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from pytest import MonkeyPatch

import fingraph.storage.qdrant.store as qdrant_store_module
from fingraph.config import Settings
from fingraph.domain.models import Chunk
from fingraph.storage.qdrant import QdrantChunkStore


def _chunk(
    *,
    chunk_id: str,
    ticker: str = "AAPL",
    fiscal_year: int = 2025,
) -> Chunk:
    """Create one deterministic chunk for Qdrant tests."""

    return cast(
        Chunk,
        SimpleNamespace(
            chunk_id=chunk_id,
            filing_id="filing-1",
            section_id="section-1",
            section_name="Item 1. Business",
            cik="0000320193",
            ticker=ticker,
            fiscal_year=fiscal_year,
            text=f"text for {chunk_id}",
            source_url="https://example.com",
        ),
    )


def _dense_embedder() -> Mock:
    """Return a deterministic dense embedder double."""

    embedder = Mock()

    embedder.embed_documents.side_effect = lambda texts, *, batch_size: [
        [
            float(index + 1),
            0.0,
        ]
        for index, _ in enumerate(texts)
    ]

    return embedder


def _sparse_embedder() -> Mock:
    """Return a deterministic sparse embedder double."""

    embedder = Mock()

    embedder.embed_documents.side_effect = lambda texts, *, batch_size: [
        (
            [
                0,
            ],
            [
                1.0,
            ],
        )
        for _ in texts
    ]

    return embedder


def test_qdrant_store_supports_local_client_without_api_key(
    monkeypatch: MonkeyPatch,
) -> None:
    """Local Qdrant should remain usable without authentication."""

    client = Mock()

    client_factory = Mock(
        return_value=client,
    )

    monkeypatch.setattr(
        qdrant_store_module,
        "QdrantClient",
        client_factory,
    )

    store = QdrantChunkStore(
        url="http://localhost:6333",
        collection_name="test_chunks",
        dense_embedder=Mock(),
        sparse_embedder=Mock(),
    )

    client_factory.assert_called_once_with(
        url="http://localhost:6333",
        api_key=None,
    )

    assert store.client is client


def test_qdrant_store_forwards_cloud_api_key(
    monkeypatch: MonkeyPatch,
) -> None:
    """Managed Qdrant credentials must be forwarded to the client."""

    client = Mock()

    client_factory = Mock(
        return_value=client,
    )

    monkeypatch.setattr(
        qdrant_store_module,
        "QdrantClient",
        client_factory,
    )

    store = QdrantChunkStore(
        url="https://example.qdrant.io",
        api_key="test-qdrant-key",
        collection_name="test_chunks",
        dense_embedder=Mock(),
        sparse_embedder=Mock(),
    )

    client_factory.assert_called_once_with(
        url="https://example.qdrant.io",
        api_key="test-qdrant-key",
    )

    assert store.client is client


def test_settings_load_qdrant_api_key_from_environment(
    monkeypatch: MonkeyPatch,
) -> None:
    """Qdrant Cloud credentials should use the FinGraph environment prefix."""

    monkeypatch.setenv(
        "FINGRAPH_QDRANT_API_KEY",
        "test-qdrant-key",
    )

    settings = Settings(
        environment="test",
        langfuse_enabled=False,
        _env_file=None,
    )

    assert settings.qdrant_api_key == "test-qdrant-key"


def test_index_chunks_uses_bounded_batches(
    monkeypatch: MonkeyPatch,
) -> None:
    """Embedding and upload work should be bounded by batch_size."""

    client = Mock()

    client_factory = Mock(
        return_value=client,
    )

    monkeypatch.setattr(
        qdrant_store_module,
        "QdrantClient",
        client_factory,
    )

    dense_embedder = _dense_embedder()

    sparse_embedder = _sparse_embedder()

    store = QdrantChunkStore(
        url="http://localhost:6333",
        collection_name="test_chunks",
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
    )

    chunks = [
        _chunk(
            chunk_id=f"chunk-{index}",
            ticker="TEST",
        )
        for index in range(5)
    ]

    indexed = store.index_chunks(
        chunks,
        batch_size=2,
    )

    assert indexed == 5

    assert dense_embedder.embed_documents.call_count == 3

    assert sparse_embedder.embed_documents.call_count == 3

    assert client.upload_points.call_count == 3

    uploaded_sizes = [len(call.kwargs["points"]) for call in client.upload_points.call_args_list]

    assert uploaded_sizes == [
        2,
        2,
        1,
    ]


def test_index_chunks_rejects_non_positive_batch_size(
    monkeypatch: MonkeyPatch,
) -> None:
    """Invalid batch sizes should fail before embedding or upload."""

    client = Mock()

    monkeypatch.setattr(
        qdrant_store_module,
        "QdrantClient",
        Mock(
            return_value=client,
        ),
    )

    store = QdrantChunkStore(
        dense_embedder=Mock(),
        sparse_embedder=Mock(),
    )

    with pytest.raises(
        ValueError,
        match=("batch_size must be greater than zero"),
    ):
        store.index_chunks(
            [],
            batch_size=0,
        )


def test_replace_ticker_requires_explicit_maintenance_mode(
    monkeypatch: MonkeyPatch,
) -> None:
    """Destructive incremental replacement must fail closed by default."""

    client = Mock()

    monkeypatch.setattr(
        qdrant_store_module,
        "QdrantClient",
        Mock(
            return_value=client,
        ),
    )

    dense_embedder = _dense_embedder()

    sparse_embedder = _sparse_embedder()

    store = QdrantChunkStore(
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
    )

    with pytest.raises(
        RuntimeError,
        match="offline-maintenance operation",
    ):
        store.replace_ticker(
            "AAPL",
            [
                _chunk(
                    chunk_id="replacement",
                )
            ],
        )

    dense_embedder.embed_documents.assert_not_called()
    sparse_embedder.embed_documents.assert_not_called()
    client.scroll.assert_not_called()
    client.delete.assert_not_called()
    client.upload_points.assert_not_called()


def test_replace_ticker_embedding_failure_does_not_delete_existing_partition(
    monkeypatch: MonkeyPatch,
) -> None:
    """Embedding failure must occur before any destructive index mutation."""

    client = Mock()

    monkeypatch.setattr(
        qdrant_store_module,
        "QdrantClient",
        Mock(
            return_value=client,
        ),
    )

    dense_embedder = Mock()
    dense_embedder.embed_documents.side_effect = RuntimeError("embedding failed")

    store = QdrantChunkStore(
        dense_embedder=dense_embedder,
        sparse_embedder=_sparse_embedder(),
    )

    with pytest.raises(
        RuntimeError,
        match="embedding failed",
    ):
        store.replace_ticker(
            "AAPL",
            [
                _chunk(
                    chunk_id="replacement",
                )
            ],
            maintenance_mode=True,
        )

    client.delete.assert_not_called()
    client.scroll.assert_not_called()
    client.upload_points.assert_not_called()


def test_replace_ticker_rejects_mismatched_chunk_before_destructive_work(
    monkeypatch: MonkeyPatch,
) -> None:
    """Ticker mismatches must fail before snapshot or deletion."""

    client = Mock()

    monkeypatch.setattr(
        qdrant_store_module,
        "QdrantClient",
        Mock(
            return_value=client,
        ),
    )

    dense_embedder = _dense_embedder()
    sparse_embedder = _sparse_embedder()

    store = QdrantChunkStore(
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
    )

    with pytest.raises(
        ValueError,
        match="does not match requested ticker",
    ):
        store.replace_ticker(
            "AAPL",
            [
                _chunk(
                    chunk_id="replacement",
                    ticker="MSFT",
                )
            ],
            maintenance_mode=True,
        )

    dense_embedder.embed_documents.assert_not_called()
    sparse_embedder.embed_documents.assert_not_called()
    client.scroll.assert_not_called()
    client.delete.assert_not_called()
    client.upload_points.assert_not_called()


def test_replace_ticker_prepares_then_replaces_partition(
    monkeypatch: MonkeyPatch,
) -> None:
    """Replacement should prepare all new points before deleting the old ticker."""

    client = Mock()

    client.scroll.return_value = (
        [],
        None,
    )

    call_order: list[str] = []

    def record_scroll(**_: object) -> tuple[list[object], None]:
        call_order.append("snapshot")
        return (
            [],
            None,
        )

    client.scroll.side_effect = record_scroll

    def record_delete(**_: object) -> None:
        call_order.append("delete")

    client.delete.side_effect = record_delete

    def record_upload(**_: object) -> None:
        call_order.append("upload")

    client.upload_points.side_effect = record_upload

    monkeypatch.setattr(
        qdrant_store_module,
        "QdrantClient",
        Mock(
            return_value=client,
        ),
    )

    dense_embedder = _dense_embedder()
    sparse_embedder = _sparse_embedder()

    dense_original = dense_embedder.embed_documents.side_effect
    sparse_original = sparse_embedder.embed_documents.side_effect

    def dense_with_order(
        texts: list[str],
        *,
        batch_size: int,
    ) -> list[list[float]]:
        call_order.append("dense")

        assert callable(dense_original)

        return cast(
            list[list[float]],
            dense_original(
                texts,
                batch_size=batch_size,
            ),
        )

    def sparse_with_order(
        texts: list[str],
        *,
        batch_size: int,
    ) -> list[tuple[list[int], list[float]]]:
        call_order.append("sparse")

        assert callable(sparse_original)

        return cast(
            list[tuple[list[int], list[float]]],
            sparse_original(
                texts,
                batch_size=batch_size,
            ),
        )

    dense_embedder.embed_documents.side_effect = dense_with_order
    sparse_embedder.embed_documents.side_effect = sparse_with_order

    store = QdrantChunkStore(
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
    )

    indexed = store.replace_ticker(
        "AAPL",
        [
            _chunk(
                chunk_id="chunk-1",
            ),
            _chunk(
                chunk_id="chunk-2",
            ),
        ],
        batch_size=1,
        maintenance_mode=True,
    )

    assert indexed == 2

    first_snapshot = call_order.index("snapshot")
    first_delete = call_order.index("delete")

    assert all(
        call_order.index(name) < first_snapshot
        for name in (
            "dense",
            "sparse",
        )
    )

    assert first_snapshot < first_delete
    assert first_delete < call_order.index("upload")


def test_replace_ticker_upload_failure_restores_previous_partition(
    monkeypatch: MonkeyPatch,
) -> None:
    """Failed replacement should delete partial new state and restore old points."""

    client = Mock()

    old_point = SimpleNamespace(
        id="old-id",
        vector={
            "dense": [
                1.0,
                0.0,
            ]
        },
        payload={
            "chunk_id": "old-chunk",
            "ticker": "AAPL",
        },
    )

    client.scroll.return_value = (
        [
            old_point,
        ],
        None,
    )

    upload_calls = 0

    def upload_points(**_: object) -> None:
        nonlocal upload_calls

        upload_calls += 1

        if upload_calls == 1:
            raise RuntimeError("replacement upload failed")

    client.upload_points.side_effect = upload_points

    monkeypatch.setattr(
        qdrant_store_module,
        "QdrantClient",
        Mock(
            return_value=client,
        ),
    )

    store = QdrantChunkStore(
        dense_embedder=_dense_embedder(),
        sparse_embedder=_sparse_embedder(),
    )

    with pytest.raises(
        RuntimeError,
        match="replacement upload failed",
    ):
        store.replace_ticker(
            "AAPL",
            [
                _chunk(
                    chunk_id="replacement",
                )
            ],
            maintenance_mode=True,
        )

    assert client.delete.call_count == 2

    assert client.upload_points.call_count == 2

    rollback_points = client.upload_points.call_args_list[-1].kwargs["points"]

    assert len(rollback_points) == 1
    assert rollback_points[0].id == "old-id"


def test_replace_ticker_reports_incomplete_rollback(
    monkeypatch: MonkeyPatch,
) -> None:
    """Rollback failure must surface loss of index-integrity guarantees."""

    client = Mock()

    old_point = SimpleNamespace(
        id="old-id",
        vector={
            "dense": [
                1.0,
                0.0,
            ]
        },
        payload={
            "chunk_id": "old-chunk",
            "ticker": "AAPL",
        },
    )

    client.scroll.return_value = (
        [
            old_point,
        ],
        None,
    )

    upload_calls = 0

    def upload_points(**_: object) -> None:
        nonlocal upload_calls

        upload_calls += 1

        raise RuntimeError(f"upload failure {upload_calls}")

    client.upload_points.side_effect = upload_points

    delete_calls = 0

    def delete(**_: object) -> None:
        nonlocal delete_calls

        delete_calls += 1

        if delete_calls == 2:
            raise RuntimeError("rollback delete failed")

    client.delete.side_effect = delete

    monkeypatch.setattr(
        qdrant_store_module,
        "QdrantClient",
        Mock(
            return_value=client,
        ),
    )

    store = QdrantChunkStore(
        dense_embedder=_dense_embedder(),
        sparse_embedder=_sparse_embedder(),
    )

    with pytest.raises(
        RuntimeError,
        match="rollback was incomplete",
    ) as captured:
        store.replace_ticker(
            "AAPL",
            [
                _chunk(
                    chunk_id="replacement",
                )
            ],
            maintenance_mode=True,
        )

    assert "rollback delete failed" in str(captured.value)
    assert "upload failure 2" in str(captured.value)


def test_replace_ticker_rejects_non_positive_batch_size(
    monkeypatch: MonkeyPatch,
) -> None:
    """Invalid replacement batch size must fail before maintenance work."""

    client = Mock()

    monkeypatch.setattr(
        qdrant_store_module,
        "QdrantClient",
        Mock(
            return_value=client,
        ),
    )

    store = QdrantChunkStore(
        dense_embedder=Mock(),
        sparse_embedder=Mock(),
    )

    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero",
    ):
        store.replace_ticker(
            "AAPL",
            [],
            batch_size=0,
            maintenance_mode=True,
        )

    client.scroll.assert_not_called()
    client.delete.assert_not_called()
    client.upload_points.assert_not_called()
