"""Search indexed SEC filing chunks."""

import argparse

from fingraph.config import get_settings
from fingraph.retrieval.hybrid import (
    reciprocal_rank_fusion,
)
from fingraph.retrieval.models import (
    RetrievalFilter,
    RetrievalResult,
)
from fingraph.storage.qdrant import (
    QdrantChunkStore,
)


def print_results(
    title: str,
    results: list[RetrievalResult],
) -> None:
    """Print compact retrieval results."""

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    for result in results:
        snippet = result.text[:220].replace("\n", " ").strip()

        print(
            f"{result.rank:>2}. "
            f"{result.ticker:<6} "
            f"{str(result.fiscal_year):<6} "
            f"{result.section_name:<28} "
            f"{result.score:.6f}"
        )

        print(f"    {snippet}...")


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "query",
    )

    parser.add_argument(
        "--ticker",
        action="append",
        default=[],
    )

    parser.add_argument(
        "--year",
        action="append",
        type=int,
        default=[],
    )

    parser.add_argument(
        "--section",
        action="append",
        default=[],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=settings.api_evidence_limit,
    )

    args = parser.parse_args()

    if args.limit <= 0:
        raise SystemExit("--limit must be positive.")

    filters = RetrievalFilter(
        tickers=tuple(ticker.upper() for ticker in args.ticker),
        fiscal_years=tuple(args.year),
        section_names=tuple(args.section),
    )

    store = QdrantChunkStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
    )

    candidate_limit = max(
        args.limit * 4,
        20,
    )

    sparse = store.sparse_search(
        args.query,
        filters=filters,
        limit=candidate_limit,
    )

    dense = store.dense_search(
        args.query,
        filters=filters,
        limit=candidate_limit,
    )

    hybrid = reciprocal_rank_fusion(
        sparse,
        dense,
        limit=args.limit,
    )

    print_results(
        "SPARSE / BM25",
        sparse[: args.limit],
    )

    print_results(
        "DENSE",
        dense[: args.limit],
    )

    print_results(
        "HYBRID / RRF",
        hybrid,
    )


if __name__ == "__main__":
    main()
