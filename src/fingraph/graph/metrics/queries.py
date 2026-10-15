"""Deterministic financial metric queries."""

from typing import Any

from fingraph.graph.metrics.catalog import (
    MetricDefinition,
)
from fingraph.storage.neo4j.store import (
    Neo4jStore,
)

_ANNUAL_FLOW_CONCEPTS = frozenset(
    {
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "NetIncomeLoss",
        "ProfitLoss",
        "OperatingIncomeLoss",
    }
)


LATEST_PERIOD_FACTS_QUERY = """
MATCH (company:Company)-[:REPORTS]->(
    fact:FinancialFact
)-[:FOR_PERIOD]->(
    period:FiscalPeriod
)

MATCH (filing:Filing)-[:REPORTED_FACT]->(
    fact
)

WHERE company.ticker = $ticker
AND fact.concept IN $concepts

AND (
    $unit IS NULL
    OR fact.unit = $unit
)

AND (
    $start_year IS NULL
    OR period.period_year >= $start_year
)

AND (
    $end_year IS NULL
    OR period.period_year <= $end_year
)

AND (
    NOT $require_annual_duration
    OR (
        fact.start_date IS NOT NULL
        AND duration.inDays(
            fact.start_date,
            fact.end_date
        ).days >= 300
        AND duration.inDays(
            fact.start_date,
            fact.end_date
        ).days <= 400
    )
)

WITH
    company,
    fact,
    period,
    filing

ORDER BY
    period.period_year ASC,
    filing.filing_date DESC,
    fact.concept ASC,
    fact.fact_id ASC

WITH
    company,
    period,
    collect({
        fact: fact,
        filing: filing
    })[0] AS latest

RETURN
    company.ticker AS ticker,
    period.period_year AS period_year,
    latest.fact.concept AS concept,
    latest.fact.value AS value,
    latest.fact.unit AS unit,
    latest.fact.fact_id AS fact_id,
    latest.fact.start_date AS start_date,
    latest.fact.end_date AS end_date,
    latest.filing.filing_date AS filed_date,
    latest.filing.accession_number AS accession_number,
    latest.fact.frame AS frame

ORDER BY period_year ASC
""".strip()


def query_metric(
    store: Neo4jStore,
    *,
    ticker: str,
    metric: MetricDefinition,
    start_year: int | None = None,
    end_year: int | None = None,
) -> list[dict[str, Any]]:
    """Return latest-filed, period-correct observations for one metric."""

    concepts = tuple(metric.concepts)

    require_annual_duration = bool(set(concepts).intersection(_ANNUAL_FLOW_CONCEPTS))

    return store.execute_read(
        LATEST_PERIOD_FACTS_QUERY,
        parameters={
            "ticker": ticker.upper(),
            "concepts": list(concepts),
            "unit": metric.unit,
            "start_year": start_year,
            "end_year": end_year,
            "require_annual_duration": require_annual_duration,
        },
    )
