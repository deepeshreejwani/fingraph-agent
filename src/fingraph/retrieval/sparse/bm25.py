"""BM25 sparse embedding utilities."""

from fastembed import SparseTextEmbedding

DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


class BM25Embedder:
    """Generate BM25-compatible sparse embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_SPARSE_MODEL,
    ) -> None:
        self.model_name = model_name

        self._model = SparseTextEmbedding(
            model_name=model_name,
        )

    def embed_documents(
        self,
        texts: list[str],
        *,
        batch_size: int = 64,
    ) -> list[tuple[list[int], list[float]]]:
        """Create sparse document vectors."""

        embeddings = self._model.embed(
            texts,
            batch_size=batch_size,
        )

        return [
            (
                embedding.indices.tolist(),
                embedding.values.tolist(),
            )
            for embedding in embeddings
        ]

    def embed_query(
        self,
        query: str,
    ) -> tuple[list[int], list[float]]:
        """Create one sparse query vector."""

        embedding = next(
            iter(
                self._model.query_embed(
                    query,
                )
            )
        )

        return (
            [int(index) for index in embedding.indices.tolist()],
            [float(value) for value in embedding.values.tolist()],
        )
