"""Query canonical financial metrics from Neo4j."""

import argparse
import json

from fingraph.config import (
    get_settings,
)
from fingraph.graph.metrics import (
    get_metric,
    query_metric,
)
from fingraph.storage.neo4j import (
    Neo4jStore,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ticker",
        required=True,
    )

    parser.add_argument(
        "--metric",
        required=True,
    )

    parser.add_argument(
        "--start-year",
        type=int,
    )

    parser.add_argument(
        "--end-year",
        type=int,
    )

    args = parser.parse_args()

    settings = get_settings()

    if not settings.neo4j_password:
        raise SystemExit("FINGRAPH_NEO4J_PASSWORD is required.")

    metric = get_metric(args.metric)

    with Neo4jStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    ) as store:
        rows = query_metric(
            store,
            ticker=args.ticker,
            metric=metric,
            start_year=args.start_year,
            end_year=args.end_year,
        )

    print(
        json.dumps(
            rows,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
