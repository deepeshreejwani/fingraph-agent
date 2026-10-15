"""Build the FinGraph retrieval index from canonical processed chunks."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

from fingraph.config import get_settings
from fingraph.domain.models import Chunk
from fingraph.ingestion.publication import (
    published_artifact_paths,
)
from fingraph.storage.qdrant import (
    QdrantChunkStore,
)

PROCESSED_ROOT = Path("data/processed/sec")


def discover_chunk_files(
    requested_tickers: tuple[str, ...] | None = None,
) -> list[
    tuple[
        str,
        Path,
    ]
]:
    """Return deterministic published ticker/chunk-file pairs."""

    available = {
        path.parent.name.upper(): path
        for path in published_artifact_paths(
            PROCESSED_ROOT,
            "chunks.json",
        )
    }

    if requested_tickers is None:
        return [
            (
                ticker,
                available[ticker],
            )
            for ticker in sorted(available)
        ]

    normalized = tuple(
        sorted({ticker.strip().upper() for ticker in requested_tickers if ticker.strip()})
    )

    missing = [ticker for ticker in normalized if ticker not in available]

    if missing:
        joined = ", ".join(missing)

        raise SystemExit(f"No published processed chunks found for: {joined}")

    return [
        (
            ticker,
            available[ticker],
        )
        for ticker in normalized
    ]


def load_company_chunks(
    path: Path,
) -> list[Chunk]:
    """
    Load and validate one company's canonical chunks.

    Memory is bounded to one company artifact rather than the entire corpus.
    Embeddings and Qdrant points are bounded separately by the store batch size.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(
        payload,
        list,
    ):
        raise ValueError(f"Expected a JSON array in {path}")

    return [Chunk.model_validate(item) for item in payload]


def main() -> None:
    """Build or incrementally refresh the configured Qdrant collection."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=(
            "rebuild",
            "incremental",
        ),
        default="rebuild",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help=("Optional ticker subset. Example: --tickers AAPL MSFT"),
    )

    parser.add_argument(
        "--maintenance-mode",
        action="store_true",
        help=(
            "Acknowledge that incremental Qdrant replacement is an "
            "offline-maintenance operation and concurrent serving is stopped."
        ),
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero.")

    if args.mode == "incremental" and not args.maintenance_mode:
        raise SystemExit(
            "Incremental mode requires --maintenance-mode because ticker "
            "replacement is not read-atomic while concurrent serving is active."
        )

    if args.mode != "incremental" and args.maintenance_mode:
        raise SystemExit("--maintenance-mode is only valid with --mode incremental.")

    ticker_filter = tuple(args.tickers) if args.tickers else None

    chunk_files = discover_chunk_files(ticker_filter)

    if not chunk_files:
        raise SystemExit("No published processed chunks found.")

    settings = get_settings()

    store = QdrantChunkStore(
        url=settings.qdrant_url,
        api_key=(settings.qdrant_api_key),
        collection_name=(settings.qdrant_collection),
    )

    total_start = perf_counter()

    total_chunks = 0
    total_batches = 0
    load_seconds = 0.0
    indexing_seconds = 0.0

    try:
        store.health_check()

        print("Qdrant connectivity: OK")

        if args.mode == "rebuild":
            sample_vector = store.dense_embedder.embed_query("financial risk")

            print(
                "Dense vector dimensions:",
                len(sample_vector),
            )

            store.recreate_collection(
                dense_vector_size=(len(sample_vector)),
            )

        elif not store.collection_exists():
            raise SystemExit(
                "Incremental mode requires an existing Qdrant collection. Run --mode rebuild first."
            )

        for (
            ticker,
            path,
        ) in chunk_files:
            load_start = perf_counter()

            chunks = load_company_chunks(path)

            load_elapsed = perf_counter() - load_start

            load_seconds += load_elapsed

            index_start = perf_counter()

            if args.mode == "incremental":
                indexed = store.replace_ticker(
                    ticker,
                    chunks,
                    batch_size=(args.batch_size),
                    maintenance_mode=True,
                )

            else:
                indexed = store.index_chunks(
                    chunks,
                    batch_size=(args.batch_size),
                )

            index_elapsed = perf_counter() - index_start

            indexing_seconds += index_elapsed

            batches = math.ceil(indexed / args.batch_size) if indexed else 0

            total_chunks += indexed
            total_batches += batches

            print(
                json.dumps(
                    {
                        "ticker": ticker,
                        "chunks": indexed,
                        "batches": batches,
                        "load_seconds": round(
                            load_elapsed,
                            3,
                        ),
                        "index_seconds": round(
                            index_elapsed,
                            3,
                        ),
                    },
                    sort_keys=True,
                )
            )

        elapsed_seconds = perf_counter() - total_start

        chunks_per_second = total_chunks / indexing_seconds if indexing_seconds > 0 else 0.0

        summary = {
            "mode": args.mode,
            "companies": len(chunk_files),
            "chunks": total_chunks,
            "batches": total_batches,
            "batch_size": (args.batch_size),
            "load_seconds": round(
                load_seconds,
                3,
            ),
            "embedding_upsert_seconds": round(
                indexing_seconds,
                3,
            ),
            "elapsed_seconds": round(
                elapsed_seconds,
                3,
            ),
            "chunks_per_second": round(
                chunks_per_second,
                3,
            ),
            "points_count": (store.points_count()),
        }

        print(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            )
        )

    finally:
        store.client.close()


if __name__ == "__main__":
    main()
