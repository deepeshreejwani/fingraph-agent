"""Benchmark sparse, dense, and hybrid retrieval."""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from fingraph.config import get_settings
from fingraph.evaluation import (
    RelevanceSpec,
    ndcg_at_k,
    precision_at_k,
    recall_at_k_explicit,
    reciprocal_rank,
)
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

DEFAULT_DATASET = Path("evals/datasets/retrieval_bootstrap.json")

DEFAULT_REPORT = Path("evals/reports/retrieval_bootstrap_report.json")


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    """Compute percentile using linear interpolation."""

    if not values:
        return 0.0

    ordered = sorted(values)

    position = (len(ordered) - 1) * percentile_value

    lower = int(position)

    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    weight = position - lower

    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def mean_optional(
    values: list[float | None],
) -> float | None:
    """Average only defined metric values."""

    defined = [value for value in values if value is not None]

    if not defined:
        return None

    return round(
        statistics.mean(defined),
        3,
    )


def parse_filter(
    payload: dict[str, Any],
) -> RetrievalFilter:
    """Parse retrieval filters from one evaluation case."""

    return RetrievalFilter(
        tickers=tuple(
            payload.get(
                "tickers",
                [],
            )
        ),
        fiscal_years=tuple(
            payload.get(
                "fiscal_years",
                [],
            )
        ),
        section_names=tuple(
            payload.get(
                "section_names",
                [],
            )
        ),
    )


def parse_relevance(
    payload: dict[str, Any],
) -> RelevanceSpec:
    """Parse bootstrap metadata or explicit chunk-level relevance."""

    return RelevanceSpec(
        tickers=frozenset(
            payload.get(
                "tickers",
                [],
            )
        ),
        fiscal_years=frozenset(
            payload.get(
                "fiscal_years",
                [],
            )
        ),
        section_names=frozenset(
            payload.get(
                "section_names",
                [],
            )
        ),
        relevant_chunk_ids=frozenset(
            payload.get(
                "relevant_chunk_ids",
                [],
            )
        ),
    )


def relevance_type(
    relevance: RelevanceSpec,
) -> str:
    """Describe the relevance contract used by one case."""

    if relevance.relevant_chunk_ids:
        return "explicit_chunk_qrels"

    return "metadata_bootstrap"


def run_method(
    *,
    store: QdrantChunkStore,
    method: str,
    question: str,
    filters: RetrievalFilter,
    limit: int,
) -> tuple[
    list[RetrievalResult],
    float,
]:
    """Run one retrieval method and measure latency."""

    start = time.perf_counter()

    if method == "sparse":
        results = store.sparse_search(
            question,
            filters=filters,
            limit=limit,
        )

    elif method == "dense":
        results = store.dense_search(
            question,
            filters=filters,
            limit=limit,
        )

    elif method == "hybrid":
        candidate_limit = max(
            limit * 4,
            20,
        )

        sparse = store.sparse_search(
            question,
            filters=filters,
            limit=candidate_limit,
        )

        dense = store.dense_search(
            question,
            filters=filters,
            limit=candidate_limit,
        )

        results = reciprocal_rank_fusion(
            sparse,
            dense,
            limit=limit,
        )

    else:
        raise ValueError(f"Unknown retrieval method: {method}")

    latency_ms = (time.perf_counter() - start) * 1000

    return (
        results,
        latency_ms,
    )


def main() -> None:
    """Run sparse, dense, and hybrid retrieval evaluation."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if args.k <= 0:
        raise SystemExit("--k must be positive.")

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))

    settings = get_settings()

    store = QdrantChunkStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
    )

    methods = (
        "sparse",
        "dense",
        "hybrid",
    )

    per_method: dict[
        str,
        list[dict[str, Any]],
    ] = {method: [] for method in methods}

    for example in dataset:
        question = example["question"]

        filters = parse_filter(
            example.get(
                "filters",
                {},
            )
        )

        relevance = parse_relevance(example["relevance"])

        judgment_type = relevance_type(relevance)

        for method in methods:
            (
                results,
                latency_ms,
            ) = run_method(
                store=store,
                method=method,
                question=question,
                filters=filters,
                limit=args.k,
            )

            metrics = {
                "id": example["id"],
                "question": question,
                "judgment_type": (judgment_type),
                "precision_at_k": (
                    precision_at_k(
                        results,
                        relevance,
                        k=args.k,
                    )
                ),
                "mrr": (
                    reciprocal_rank(
                        results,
                        relevance,
                        k=args.k,
                    )
                ),
                "ndcg_at_k": (
                    ndcg_at_k(
                        results,
                        relevance,
                        k=args.k,
                    )
                ),
                "recall_at_k": (
                    recall_at_k_explicit(
                        results,
                        relevance,
                        k=args.k,
                    )
                ),
                "latency_ms": latency_ms,
                "top_results": [
                    {
                        "rank": (result.rank),
                        "ticker": (result.ticker),
                        "fiscal_year": (result.fiscal_year),
                        "section_name": (result.section_name),
                        "chunk_id": (result.chunk_id),
                        "score": (result.score),
                    }
                    for result in results
                ],
            }

            per_method[method].append(metrics)

    summary: dict[
        str,
        dict[str, Any],
    ] = {}

    for (
        method,
        rows,
    ) in per_method.items():
        latencies = [row["latency_ms"] for row in rows]

        explicit_rows = [row for row in rows if row["judgment_type"] == "explicit_chunk_qrels"]

        summary[method] = {
            "questions": len(rows),
            "explicit_qrels_questions": (len(explicit_rows)),
            "metadata_bootstrap_questions": (len(rows) - len(explicit_rows)),
            "precision_at_k": round(
                statistics.mean(row["precision_at_k"] for row in rows),
                3,
            ),
            "mrr": round(
                statistics.mean(row["mrr"] for row in rows),
                3,
            ),
            "ndcg_at_k": (mean_optional([row["ndcg_at_k"] for row in rows])),
            "recall_at_k": (mean_optional([row["recall_at_k"] for row in rows])),
            "latency_ms_mean": round(
                statistics.mean(latencies),
                3,
            ),
            "latency_ms_p50": round(
                percentile(
                    latencies,
                    0.50,
                ),
                3,
            ),
            "latency_ms_p95": round(
                percentile(
                    latencies,
                    0.95,
                ),
                3,
            ),
        }

    explicit_qrels_cases = sum(
        bool(parse_relevance(example["relevance"]).relevant_chunk_ids) for example in dataset
    )

    metadata_bootstrap_cases = len(dataset) - explicit_qrels_cases

    report = {
        "dataset": str(args.dataset),
        "k": args.k,
        "judgment_contract": {
            "explicit_chunk_qrels": (
                "Chunk IDs define the "
                "complete relevance set. "
                "P@K, MRR@K, NDCG@K, "
                "and Recall@K are valid."
            ),
            "metadata_bootstrap": (
                "Ticker/year/section "
                "metadata defines relevance "
                "for retrieved results only. "
                "P@K and MRR@K are reported; "
                "NDCG@K and Recall@K are "
                "undefined because the full "
                "relevant-document universe "
                "is unknown."
            ),
        },
        "dataset_judgment_counts": {
            "explicit_chunk_qrels": (explicit_qrels_cases),
            "metadata_bootstrap": (metadata_bootstrap_cases),
        },
        "summary": summary,
        "details": per_method,
    }

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.report.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()

    print(
        f"{'METHOD':<10}"
        f"{'P@K':>10}"
        f"{'MRR':>10}"
        f"{'NDCG':>10}"
        f"{'RECALL':>10}"
        f"{'MEAN MS':>12}"
        f"{'P95 MS':>12}"
    )

    print("-" * 74)

    for method in methods:
        values = summary[method]

        ndcg = "N/A" if values["ndcg_at_k"] is None else f"{values['ndcg_at_k']:.3f}"

        recall = "N/A" if values["recall_at_k"] is None else f"{values['recall_at_k']:.3f}"

        print(
            f"{method:<10}"
            f"{values['precision_at_k']:>10.3f}"
            f"{values['mrr']:>10.3f}"
            f"{ndcg:>10}"
            f"{recall:>10}"
            f"{values['latency_ms_mean']:>12.3f}"
            f"{values['latency_ms_p95']:>12.3f}"
        )

    print()

    if metadata_bootstrap_cases > 0:
        print("NDCG@K and Recall@K are reported only for cases with explicit chunk-level qrels.")

    print()

    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
