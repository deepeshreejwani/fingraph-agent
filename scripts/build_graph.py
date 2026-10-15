"""Build and audit the deterministic FinGraph Neo4j graph."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

from fingraph.config import (
    get_settings,
)
from fingraph.storage.neo4j.loader import (
    graph_integrity_report,
    prune_stale_graph,
)
from fingraph.storage.neo4j.scale_loader import (
    load_companies_scaled,
    load_facts_scaled,
    load_filings_and_sections_scaled,
)
from fingraph.storage.neo4j.store import (
    Neo4jStore,
)


def main() -> None:
    """Load canonical processed artifacts into Neo4j."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_000,
        help=("Maximum rows per Neo4j UNWIND write transaction."),
    )

    parser.add_argument(
        "--prune-stale",
        action="store_true",
        help=("Delete graph entities that are not present in the current processed SEC corpus."),
    )

    parser.add_argument(
        "--audit-only",
        action="store_true",
        help=("Do not load data; only compare the current graph with the processed SEC corpus."),
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero.")

    settings = get_settings()

    if not settings.neo4j_password:
        raise SystemExit("FINGRAPH_NEO4J_PASSWORD is required.")

    if args.audit_only and args.prune_stale:
        raise SystemExit("--audit-only and --prune-stale cannot be used together.")

    total_start = perf_counter()

    with Neo4jStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=(settings.neo4j_password),
        database=(settings.neo4j_database),
        read_timeout_seconds=(settings.neo4j_read_timeout_seconds),
        connection_timeout_seconds=(settings.neo4j_connection_timeout_seconds),
        max_transaction_retry_time_seconds=(settings.neo4j_max_transaction_retry_time_seconds),
    ) as store:
        store.verify_connectivity()

        print("Neo4j connectivity: OK")

        if args.audit_only:
            audit_start = perf_counter()

            report = graph_integrity_report(store)

            audit_seconds = perf_counter() - audit_start

            output = {
                "audit": report,
                "audit_seconds": round(
                    audit_seconds,
                    3,
                ),
            }

            print(
                json.dumps(
                    output,
                    indent=2,
                    sort_keys=True,
                )
            )

            if not report["passed"]:
                raise SystemExit("Neo4j integrity audit failed.")

            return

        schema_start = perf_counter()

        store.initialize_schema()

        schema_seconds = perf_counter() - schema_start

        companies_start = perf_counter()

        companies = load_companies_scaled(
            store,
            batch_size=(args.batch_size),
        )

        companies_seconds = perf_counter() - companies_start

        filings_start = perf_counter()

        (
            filings,
            sections,
        ) = load_filings_and_sections_scaled(
            store,
            batch_size=(args.batch_size),
        )

        filings_sections_seconds = perf_counter() - filings_start

        facts_start = perf_counter()

        (
            facts,
            periods,
        ) = load_facts_scaled(
            store,
            batch_size=(args.batch_size),
        )

        facts_periods_seconds = perf_counter() - facts_start

        loaded = {
            "companies": companies,
            "filings": filings,
            "sections": sections,
            "financial_facts": facts,
            "fiscal_periods": periods,
        }

        loaded_rows = sum(loaded.values())

        load_seconds = companies_seconds + filings_sections_seconds + facts_periods_seconds

        rows_per_second = loaded_rows / load_seconds if load_seconds > 0 else 0.0

        pruned: (
            dict[
                str,
                int,
            ]
            | None
        ) = None

        prune_seconds = 0.0

        if args.prune_stale:
            prune_start = perf_counter()

            pruned = prune_stale_graph(store)

            prune_seconds = perf_counter() - prune_start

        audit_start = perf_counter()

        audit = graph_integrity_report(store)

        audit_seconds = perf_counter() - audit_start

        elapsed_seconds = perf_counter() - total_start

        summary = {
            "loaded": loaded,
            "pruned": pruned,
            "audit": audit,
            "metrics": {
                "transaction_batch_size": (args.batch_size),
                "loaded_rows": (loaded_rows),
                "schema_seconds": round(
                    schema_seconds,
                    3,
                ),
                "companies_seconds": round(
                    companies_seconds,
                    3,
                ),
                "filings_sections_seconds": round(
                    filings_sections_seconds,
                    3,
                ),
                "facts_periods_seconds": round(
                    facts_periods_seconds,
                    3,
                ),
                "load_seconds": round(
                    load_seconds,
                    3,
                ),
                "rows_per_second": round(
                    rows_per_second,
                    3,
                ),
                "prune_seconds": round(
                    prune_seconds,
                    3,
                ),
                "audit_seconds": round(
                    audit_seconds,
                    3,
                ),
                "elapsed_seconds": round(
                    elapsed_seconds,
                    3,
                ),
            },
        }

        print(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            )
        )

        if not audit["passed"]:
            message = "Neo4j integrity audit failed."

            if not args.prune_stale and audit["count_mismatches"]:
                message += (
                    " If mismatches are caused by stale nodes, rerun explicitly with --prune-stale."
                )

            raise SystemExit(message)


if __name__ == "__main__":
    main()
