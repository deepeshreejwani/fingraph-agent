"""Local dense embedding generation."""

from collections.abc import Iterable

from fastembed import TextEmbedding

DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"


class DenseEmbedder:
    """Generate local dense text embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_DENSE_MODEL,
    ) -> None:
        self.model_name = model_name

        self._model = TextEmbedding(
            model_name=model_name,
            lazy_load=True,
        )

    def embed_documents(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 64,
    ) -> list[list[float]]:
        """Embed document text."""

        vectors = self._model.embed(
            list(texts),
            batch_size=batch_size,
        )

        return [vector.tolist() for vector in vectors]

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """Embed a single search query."""

        vectors = self._model.query_embed(
            query,
        )

        vector = next(iter(vectors))

        return [float(value) for value in vector.tolist()]
