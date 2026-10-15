"""Qdrant-backed filing retrieval."""

from collections.abc import Iterable, Iterator
from itertools import islice
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from fingraph.domain.models import Chunk
from fingraph.retrieval.dense import DenseEmbedder
from fingraph.retrieval.models import (
    RetrievalFilter,
    RetrievalMethod,
    RetrievalResult,
)
from fingraph.retrieval.sparse import BM25Embedder

DEFAULT_COLLECTION = "fingraph_chunks"

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"


def qdrant_point_id(
    chunk_id: str,
) -> str:
    """Map a canonical chunk ID to a deterministic Qdrant-compatible UUID."""

    return str(
        uuid5(
            NAMESPACE_URL,
            f"fingraph:{chunk_id}",
        )
    )


def _batched(
    items: Iterable[Chunk],
    *,
    batch_size: int,
) -> Iterator[list[Chunk]]:
    """Yield bounded chunk batches without materializing the full iterable."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    iterator = iter(items)

    while True:
        batch = list(
            islice(
                iterator,
                batch_size,
            )
        )

        if not batch:
            return

        yield batch


def _batched_points(
    items: list[models.PointStruct],
    *,
    batch_size: int,
) -> Iterator[list[models.PointStruct]]:
    """Yield bounded point batches from a prepared ticker replacement."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    for start in range(
        0,
        len(items),
        batch_size,
    ):
        yield items[start : start + batch_size]


class QdrantChunkStore:
    """Store and retrieve filing chunks in Qdrant."""

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        dense_embedder: DenseEmbedder | None = None,
        sparse_embedder: BM25Embedder | None = None,
    ) -> None:
        self.collection_name = collection_name

        self.client = QdrantClient(
            url=url,
            api_key=api_key,
        )

        self.dense_embedder = dense_embedder or DenseEmbedder()

        self.sparse_embedder = sparse_embedder or BM25Embedder()

        self._available_fiscal_years_cache: dict[
            str,
            tuple[int, ...],
        ] = {}

    def health_check(
        self,
    ) -> None:
        """Verify Qdrant is reachable."""

        self.client.get_collections()

    def collection_exists(
        self,
    ) -> bool:
        """Return whether the configured collection currently exists."""

        return bool(self.client.collection_exists(self.collection_name))

    def points_count(
        self,
    ) -> int:
        """Return the current number of indexed points."""

        collection = self.client.get_collection(self.collection_name)

        return int(collection.points_count or 0)

    def recreate_collection(
        self,
        *,
        dense_vector_size: int,
    ) -> None:
        """Create a fresh development retrieval collection."""

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=(self.collection_name),
            vectors_config={
                DENSE_VECTOR_NAME: (
                    models.VectorParams(
                        size=dense_vector_size,
                        distance=(models.Distance.COSINE),
                    )
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: (
                    models.SparseVectorParams(
                        modifier=(models.Modifier.IDF),
                    )
                )
            },
        )

        for field in (
            "ticker",
            "fiscal_year",
            "section_name",
            "filing_id",
        ):
            self.client.create_payload_index(
                collection_name=(self.collection_name),
                field_name=field,
                field_schema=(
                    models.PayloadSchemaType.INTEGER
                    if field == "fiscal_year"
                    else models.PayloadSchemaType.KEYWORD
                ),
            )

        self._available_fiscal_years_cache.clear()

    @staticmethod
    def _normalize_ticker(
        ticker: str,
    ) -> str:
        """Normalize and validate one ticker identifier."""

        normalized = ticker.strip().upper()

        if not normalized:
            raise ValueError("ticker must not be empty.")

        return normalized

    def _ticker_filter(
        self,
        ticker: str,
    ) -> models.Filter:
        """Build the canonical payload filter for one ticker partition."""

        normalized_ticker = self._normalize_ticker(ticker)

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="ticker",
                    match=(models.MatchValue(value=(normalized_ticker))),
                )
            ]
        )

    def delete_ticker(
        self,
        ticker: str,
    ) -> None:
        """
        Delete all currently indexed chunks for one ticker.

        This remains useful as a low-level administrative primitive. Incremental
        production replacement should use replace_ticker() so failures can be
        rolled back.
        """

        normalized_ticker = self._normalize_ticker(ticker)

        self.client.delete(
            collection_name=(self.collection_name),
            points_selector=(
                models.FilterSelector(filter=(self._ticker_filter(normalized_ticker)))
            ),
            wait=True,
        )

        self._available_fiscal_years_cache.pop(
            normalized_ticker,
            None,
        )

    def _prepare_chunk_batch(
        self,
        chunks: list[Chunk],
        *,
        batch_size: int,
    ) -> list[models.PointStruct]:
        """Embed one chunk batch and build Qdrant points without uploading."""

        texts = [chunk.text for chunk in chunks]

        dense_vectors = self.dense_embedder.embed_documents(
            texts,
            batch_size=batch_size,
        )

        sparse_vectors = self.sparse_embedder.embed_documents(
            texts,
            batch_size=batch_size,
        )

        points: list[models.PointStruct] = []

        for (
            chunk,
            dense_vector,
            sparse_vector,
        ) in zip(
            chunks,
            dense_vectors,
            sparse_vectors,
            strict=True,
        ):
            (
                sparse_indices,
                sparse_values,
            ) = sparse_vector

            points.append(
                models.PointStruct(
                    id=(qdrant_point_id(chunk.chunk_id)),
                    vector={
                        DENSE_VECTOR_NAME: (dense_vector),
                        SPARSE_VECTOR_NAME: (
                            models.SparseVector(
                                indices=(sparse_indices),
                                values=(sparse_values),
                            )
                        ),
                    },
                    payload={
                        "chunk_id": (chunk.chunk_id),
                        "filing_id": (chunk.filing_id),
                        "section_id": (chunk.section_id),
                        "section_name": (chunk.section_name),
                        "cik": chunk.cik,
                        "ticker": (chunk.ticker),
                        "fiscal_year": (chunk.fiscal_year),
                        "text": chunk.text,
                        "source_url": (chunk.source_url),
                    },
                )
            )

        return points

    def _upload_prepared_points(
        self,
        points: list[models.PointStruct],
        *,
        batch_size: int,
    ) -> None:
        """Upload already-prepared points using bounded synchronous batches."""

        for batch in _batched_points(
            points,
            batch_size=batch_size,
        ):
            self.client.upload_points(
                collection_name=(self.collection_name),
                points=batch,
                batch_size=batch_size,
                parallel=1,
                max_retries=3,
                wait=True,
            )

    def _snapshot_ticker_points(
        self,
        ticker: str,
    ) -> list[models.PointStruct]:
        """
        Snapshot one ticker partition including vectors and payload.

        The snapshot is used only during incremental replacement rollback, so
        memory remains bounded to one ticker rather than the complete corpus.
        """

        ticker_filter = self._ticker_filter(ticker)

        offset = None

        snapshot: list[models.PointStruct] = []

        while True:
            (
                records,
                next_offset,
            ) = self.client.scroll(
                collection_name=(self.collection_name),
                scroll_filter=(ticker_filter),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            for record in records:
                if record.vector is None:
                    raise RuntimeError("Qdrant rollback snapshot returned a point without vectors.")

                snapshot.append(
                    models.PointStruct(
                        id=record.id,
                        vector=record.vector,
                        payload=(record.payload or {}),
                    )
                )

            if next_offset is None:
                break

            offset = next_offset

        return snapshot

    def _validate_replacement_chunks(
        self,
        *,
        ticker: str,
        chunks: list[Chunk],
    ) -> str:
        """Validate that all replacement chunks belong to the requested ticker."""

        normalized_ticker = self._normalize_ticker(ticker)

        for chunk in chunks:
            if chunk.ticker.strip().upper() != normalized_ticker:
                raise ValueError(
                    "Replacement chunk ticker "
                    f"{chunk.ticker!r} does not "
                    "match requested ticker "
                    f"{normalized_ticker!r}."
                )

        return normalized_ticker

    def replace_ticker(
        self,
        ticker: str,
        chunks: Iterable[Chunk],
        *,
        batch_size: int = 64,
        maintenance_mode: bool = False,
    ) -> int:
        """
        Replace one ticker partition with rollback protection.

        All replacement chunks are validated and embedded before the existing
        partition is touched. The old Qdrant points are then snapshotted with
        payload and vectors.

        If replacement upload fails after deletion, any partially uploaded new
        partition is removed and the old snapshot is restored before the
        original replacement error is re-raised.

        If rollback itself cannot restore the previous partition, a RuntimeError
        is raised because the caller can no longer assume the index is intact.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if not maintenance_mode:
            raise RuntimeError(
                "Qdrant ticker replacement is an offline-maintenance operation. "
                "Set maintenance_mode=True only when concurrent serving is stopped."
            )

        replacement_chunks = list(chunks)

        normalized_ticker = self._validate_replacement_chunks(
            ticker=ticker,
            chunks=(replacement_chunks),
        )

        prepared_points: list[models.PointStruct] = []

        for batch in _batched(
            replacement_chunks,
            batch_size=batch_size,
        ):
            prepared_points.extend(
                self._prepare_chunk_batch(
                    batch,
                    batch_size=batch_size,
                )
            )

        previous_points = self._snapshot_ticker_points(normalized_ticker)

        self.delete_ticker(normalized_ticker)

        try:
            self._upload_prepared_points(
                prepared_points,
                batch_size=batch_size,
            )

        except Exception as replacement_error:
            rollback_errors: list[Exception] = []

            try:
                self.delete_ticker(normalized_ticker)

            except Exception as exc:
                rollback_errors.append(exc)

            try:
                self._upload_prepared_points(
                    previous_points,
                    batch_size=batch_size,
                )

            except Exception as exc:
                rollback_errors.append(exc)

            self._available_fiscal_years_cache.pop(
                normalized_ticker,
                None,
            )

            if rollback_errors:
                rollback_summary = "; ".join(
                    f"{type(error).__name__}: {error}" for error in rollback_errors
                )

                raise RuntimeError(
                    "Qdrant ticker replacement failed "
                    "and rollback was incomplete: "
                    f"{rollback_summary}"
                ) from replacement_error

            raise

        self._available_fiscal_years_cache.pop(
            normalized_ticker,
            None,
        )

        return len(prepared_points)

    def index_chunks(
        self,
        chunks: Iterable[Chunk],
        *,
        batch_size: int = 64,
    ) -> int:
        """
        Generate vectors and upsert chunks using bounded batches.

        The full chunk iterable is never materialized by this method. Dense
        vectors, sparse vectors and Qdrant points exist only for the current
        bounded batch.

        Returns the number of successfully submitted chunks.
        """

        indexed = 0

        for batch in _batched(
            chunks,
            batch_size=batch_size,
        ):
            points = self._prepare_chunk_batch(
                batch,
                batch_size=batch_size,
            )

            self._upload_prepared_points(
                points,
                batch_size=batch_size,
            )

            indexed += len(batch)

        if indexed:
            self._available_fiscal_years_cache.clear()

        return indexed

    def available_fiscal_years(
        self,
        ticker: str,
    ) -> tuple[int, ...]:
        """
        Return sorted indexed fiscal years for one ticker.

        This is a metadata lookup rather than semantic retrieval.
        Results are cached for the lifetime of the store instance.
        """

        normalized_ticker = ticker.strip().upper()

        if not normalized_ticker:
            raise ValueError("ticker must not be empty.")

        existing = self._available_fiscal_years_cache.get(normalized_ticker)

        if existing is not None:
            return existing

        retrieval_filter = RetrievalFilter(tickers=(normalized_ticker,))

        query_filter = self._build_filter(retrieval_filter)

        offset = None

        years: set[int] = set()

        while True:
            (
                points,
                next_offset,
            ) = self.client.scroll(
                collection_name=(self.collection_name),
                scroll_filter=(query_filter),
                limit=256,
                offset=offset,
                with_payload=["fiscal_year"],
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}

                raw_year = payload.get("fiscal_year")

                if raw_year is None:
                    continue

                years.add(int(raw_year))

            if next_offset is None:
                break

            offset = next_offset

        result = tuple(sorted(years))

        self._available_fiscal_years_cache[normalized_ticker] = result

        return result

    def latest_fiscal_year(
        self,
        ticker: str,
    ) -> int | None:
        """Return the latest indexed fiscal year for one ticker."""

        years = self.available_fiscal_years(ticker)

        if not years:
            return None

        return years[-1]

    def dense_search(
        self,
        query: str,
        *,
        filters: RetrievalFilter | None = None,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        """Run dense semantic retrieval."""

        query_vector = self.dense_embedder.embed_query(query)

        response = self.client.query_points(
            collection_name=(self.collection_name),
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            query_filter=(self._build_filter(filters)),
            limit=limit,
            with_payload=True,
        )

        return self._to_results(
            response.points,
            method="dense",
        )

    def sparse_search(
        self,
        query: str,
        *,
        filters: RetrievalFilter | None = None,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        """Run BM25 sparse retrieval."""

        (
            indices,
            values,
        ) = self.sparse_embedder.embed_query(query)

        response = self.client.query_points(
            collection_name=(self.collection_name),
            query=(
                models.SparseVector(
                    indices=indices,
                    values=values,
                )
            ),
            using=SPARSE_VECTOR_NAME,
            query_filter=(self._build_filter(filters)),
            limit=limit,
            with_payload=True,
        )

        return self._to_results(
            response.points,
            method="sparse",
        )

    @staticmethod
    def _build_filter(
        retrieval_filter: RetrievalFilter | None,
    ) -> models.Filter | None:
        """Convert canonical metadata restrictions to Qdrant filters."""

        if retrieval_filter is None:
            return None

        conditions: list[models.FieldCondition] = []

        if retrieval_filter.tickers:
            conditions.append(
                models.FieldCondition(
                    key="ticker",
                    match=(models.MatchAny(any=list(retrieval_filter.tickers))),
                )
            )

        if retrieval_filter.fiscal_years:
            conditions.append(
                models.FieldCondition(
                    key="fiscal_year",
                    match=(models.MatchAny(any=list(retrieval_filter.fiscal_years))),
                )
            )

        if retrieval_filter.section_names:
            conditions.append(
                models.FieldCondition(
                    key="section_name",
                    match=(models.MatchAny(any=list(retrieval_filter.section_names))),
                )
            )

        if not conditions:
            return None

        return models.Filter(
            must=conditions,
        )

    @staticmethod
    def _to_results(
        points: Iterable[models.ScoredPoint],
        *,
        method: RetrievalMethod,
    ) -> list[RetrievalResult]:
        """Convert Qdrant points into canonical retrieval results."""

        results: list[RetrievalResult] = []

        for rank, point in enumerate(
            points,
            start=1,
        ):
            payload = point.payload or {}

            results.append(
                RetrievalResult(
                    chunk_id=str(payload["chunk_id"]),
                    filing_id=str(payload["filing_id"]),
                    section_id=str(payload["section_id"]),
                    section_name=str(payload["section_name"]),
                    cik=str(payload["cik"]),
                    ticker=str(payload["ticker"]),
                    fiscal_year=(
                        int(payload["fiscal_year"])
                        if payload.get("fiscal_year") is not None
                        else None
                    ),
                    text=str(payload["text"]),
                    source_url=str(payload["source_url"]),
                    score=float(point.score),
                    method=method,
                    rank=rank,
                )
            )

        return results
