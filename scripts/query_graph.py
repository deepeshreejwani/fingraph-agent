"""Run deterministic graph queries."""

import argparse
import json

from fingraph.config import get_settings
from fingraph.storage.neo4j.store import (
    Neo4jStore,
)

COMPANY_QUERY = """
MATCH (company:Company)
RETURN
    company.ticker AS ticker,
    company.cik AS cik
ORDER BY ticker
"""


METRIC_QUERY = """
MATCH (company:Company)-[:REPORTS]->(
    fact:FinancialFact
)-[:FOR_PERIOD]->(
    period:FiscalPeriod
)

MATCH (filing:Filing)-[:REPORTED_FACT]->(
    fact
)

WHERE company.ticker = $ticker
AND toLower(fact.concept)
    CONTAINS toLower($concept)

RETURN
    company.ticker AS ticker,
    fact.concept AS concept,
    fact.value AS value,
    fact.unit AS unit,
    period.start_date AS start_date,
    period.end_date AS end_date,
    period.period_year AS period_year,
    fact.fiscal_year AS sec_fiscal_year,
    fact.fiscal_period AS sec_fiscal_period,
    fact.frame AS frame,
    filing.filing_date AS filed_date,
    filing.accession_number AS accession_number

ORDER BY
    period.end_date DESC,
    filing.filing_date DESC,
    fact.concept ASC

LIMIT $limit
"""


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ticker",
    )

    parser.add_argument(
        "--concept",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    settings = get_settings()

    if not settings.neo4j_password:
        raise SystemExit("FINGRAPH_NEO4J_PASSWORD is required.")

    with Neo4jStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=(settings.neo4j_password),
        database=(settings.neo4j_database),
    ) as store:
        if args.ticker and args.concept:
            rows = store.execute_read(
                METRIC_QUERY,
                parameters={
                    "ticker": (args.ticker.upper()),
                    "concept": (args.concept),
                    "limit": (args.limit),
                },
            )

        else:
            rows = store.execute_read(COMPANY_QUERY)

        print(
            json.dumps(
                rows,
                indent=2,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
