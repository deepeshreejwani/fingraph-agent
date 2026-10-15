"""Regression tests for derived-margin Graph QA repair behavior."""

from typing import Any

from fingraph.graph.qa.models import CypherParameter, CypherPlan
from fingraph.graph.qa.service import GraphQAService


class FakeGenerator:
    """Return deterministic Cypher plans in sequence."""

    def __init__(self, plans: list[CypherPlan]) -> None:
        self.plans = plans
        self.calls: list[dict[str, Any]] = []

    def generate_cypher(
        self,
        *,
        question: str,
        schema: str,
        previous_cypher: str | None = None,
        error: str | None = None,
    ) -> CypherPlan:
        self.calls.append(
            {
                "question": question,
                "schema": schema,
                "previous_cypher": previous_cypher,
                "error": error,
            }
        )

        return self.plans[len(self.calls) - 1]


def _row(
    *,
    concept: str,
    value: float,
    fact_id: str,
) -> dict[str, Any]:
    """Return one complete annual MSFT financial row."""

    return {
        "ticker": "MSFT",
        "year": 2026,
        "metric_concept": concept,
        "value": value,
        "unit": "USD",
        "fact_id": fact_id,
        "start_date": "2025-07-01",
        "end_date": "2026-06-30",
        "accession_number": "0001193125-26-323660",
        "filing_date": "2026-07-29",
    }


class MarginStore:
    """Return controlled raw financial rows for margin queries."""

    def __init__(self) -> None:
        self.explained: list[str] = []
        self.executed: list[str] = []

    def explain_read(
        self,
        cypher: str,
        *,
        parameters: dict[str, Any],
    ) -> None:
        del parameters
        self.explained.append(cypher)

    def execute_read(
        self,
        cypher: str,
        *,
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del parameters
        self.executed.append(cypher)

        if "OperatingIncomeLoss" in cypher:
            return [
                _row(
                    concept="OperatingIncomeLoss",
                    value=155237000000.0,
                    fact_id="msft-operating-income-2026",
                ),
                _row(
                    concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                    value=331839000000.0,
                    fact_id="msft-revenue-2026",
                ),
            ]

        if "metric_concept, fact, filing" in cypher:
            return [
                _row(
                    concept="NetIncomeLoss",
                    value=133749000000.0,
                    fact_id="msft-net-income-2026",
                ),
                _row(
                    concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                    value=331839000000.0,
                    fact_id="msft-revenue-2026",
                ),
            ]

        return [
            _row(
                concept="NetIncomeLoss",
                value=133749000000.0,
                fact_id="msft-net-income-2026",
            )
        ]


def _scalar(
    name: str,
    value: str,
) -> CypherParameter:
    return CypherParameter(
        name=name,
        kind="scalar",
        scalar=value,
        values=[],
    )


def test_operating_margin_repairs_initial_empty_plan() -> None:
    repaired = CypherPlan(
        cypher="""
MATCH (company:Company {ticker: $ticker})
MATCH (company)-[:REPORTS]->(fact:FinancialFact)-[:FOR_PERIOD]->(period:FiscalPeriod)
MATCH (filing:Filing)-[:REPORTED_FACT]->(fact)
WHERE fact.concept IN [
    "OperatingIncomeLoss",
    "RevenueFromContractWithCustomerExcludingAssessedTax"
]
  AND period.period_year = toInteger($year)
WITH company, period, fact.concept AS metric_concept, fact, filing
ORDER BY period.period_year ASC, metric_concept ASC, filing.filing_date DESC
WITH company, period, metric_concept,
     collect({fact: fact, filing: filing})[0] AS latest
RETURN company.ticker AS ticker,
       period.period_year AS year,
       latest.fact.concept AS metric_concept,
       latest.fact.value AS value,
       latest.fact.unit AS unit,
       latest.fact.fact_id AS fact_id,
       latest.fact.start_date AS start_date,
       latest.fact.end_date AS end_date,
       latest.filing.accession_number AS accession_number,
       latest.filing.filing_date AS filing_date
""".strip(),
        parameters=[
            _scalar("ticker", "MSFT"),
            _scalar("year", "2026"),
        ],
    )

    generator = FakeGenerator(
        [
            CypherPlan(
                cypher="",
                parameters=[],
            ),
            repaired,
        ]
    )

    store = MarginStore()

    result = GraphQAService(
        store=store,  # type: ignore[arg-type]
        generator=generator,
        max_attempts=2,
    ).ask("What was Microsoft's operating margin in 2026?")

    assert result.attempts == 2
    assert len(generator.calls) == 2

    assert "supported derived financial metric" in str(generator.calls[1]["error"])

    assert {row["metric_concept"] for row in result.rows} == {
        "OperatingIncomeLoss",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
    }


def test_net_margin_repairs_collapsed_multi_metric_result() -> None:
    bad = CypherPlan(
        cypher="""
MATCH (company:Company {ticker: $ticker})
MATCH (company)-[:REPORTS]->(fact:FinancialFact)-[:FOR_PERIOD]->(period:FiscalPeriod)
MATCH (filing:Filing)-[:REPORTED_FACT]->(fact)
WHERE fact.concept IN [
    "NetIncomeLoss",
    "ProfitLoss",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet"
]
  AND period.period_year = toInteger($year)
WITH company, period, fact, filing
ORDER BY period.period_year ASC, filing.filing_date DESC
WITH company, period,
     collect({fact: fact, filing: filing})[0] AS latest
RETURN company.ticker AS ticker,
       period.period_year AS year,
       latest.fact.concept AS metric_concept,
       latest.fact.value AS value,
       latest.fact.unit AS unit,
       latest.fact.fact_id AS fact_id,
       latest.fact.start_date AS start_date,
       latest.fact.end_date AS end_date,
       latest.filing.accession_number AS accession_number,
       latest.filing.filing_date AS filing_date
""".strip(),
        parameters=[
            _scalar("ticker", "MSFT"),
            _scalar("year", "2026"),
        ],
    )

    repaired = CypherPlan(
        cypher="""
MATCH (company:Company {ticker: $ticker})
MATCH (company)-[:REPORTS]->(fact:FinancialFact)-[:FOR_PERIOD]->(period:FiscalPeriod)
MATCH (filing:Filing)-[:REPORTED_FACT]->(fact)
WHERE fact.concept IN [
    "NetIncomeLoss",
    "ProfitLoss",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet"
]
  AND period.period_year = toInteger($year)
WITH company, period, fact.concept AS metric_concept, fact, filing
ORDER BY period.period_year ASC, metric_concept ASC, filing.filing_date DESC
WITH company, period, metric_concept,
     collect({fact: fact, filing: filing})[0] AS latest
RETURN company.ticker AS ticker,
       period.period_year AS year,
       latest.fact.concept AS metric_concept,
       latest.fact.value AS value,
       latest.fact.unit AS unit,
       latest.fact.fact_id AS fact_id,
       latest.fact.start_date AS start_date,
       latest.fact.end_date AS end_date,
       latest.filing.accession_number AS accession_number,
       latest.filing.filing_date AS filing_date
""".strip(),
        parameters=[
            _scalar("ticker", "MSFT"),
            _scalar("year", "2026"),
        ],
    )

    generator = FakeGenerator(
        [
            bad,
            repaired,
        ]
    )

    store = MarginStore()

    result = GraphQAService(
        store=store,  # type: ignore[arg-type]
        generator=generator,
        max_attempts=2,
    ).ask("What was Microsoft's net margin in 2026?")

    assert result.attempts == 2
    assert len(generator.calls) == 2

    error = str(generator.calls[1]["error"])
    assert "missing required raw metric(s)" in error
    assert "revenue" in error

    assert {row["metric_concept"] for row in result.rows} == {
        "NetIncomeLoss",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
    }


def test_net_margin_rejects_plan_missing_revenue_before_execution() -> None:
    bad = CypherPlan(
        cypher="""
MATCH (company:Company {ticker: $ticker})
MATCH (company)-[:REPORTS]->(fact:FinancialFact)-[:FOR_PERIOD]->(period:FiscalPeriod)
MATCH (filing:Filing)-[:REPORTED_FACT]->(fact)
WHERE fact.concept IN [
    "NetIncomeLoss",
    "ProfitLoss"
]
  AND period.period_year = toInteger($year)
RETURN company.ticker AS ticker,
       period.period_year AS year,
       fact.concept AS metric_concept,
       fact.value AS value,
       fact.unit AS unit,
       fact.fact_id AS fact_id,
       fact.start_date AS start_date,
       fact.end_date AS end_date,
       filing.accession_number AS accession_number,
       filing.filing_date AS filing_date
""".strip(),
        parameters=[
            _scalar("ticker", "MSFT"),
            _scalar("year", "2026"),
        ],
    )

    repaired = CypherPlan(
        cypher="""
MATCH (company:Company {ticker: $ticker})
MATCH (company)-[:REPORTS]->(fact:FinancialFact)-[:FOR_PERIOD]->(period:FiscalPeriod)
MATCH (filing:Filing)-[:REPORTED_FACT]->(fact)
WHERE fact.concept IN [
    "NetIncomeLoss",
    "RevenueFromContractWithCustomerExcludingAssessedTax"
]
  AND period.period_year = toInteger($year)
WITH company, period, fact.concept AS metric_concept, fact, filing
ORDER BY period.period_year ASC, metric_concept ASC, filing.filing_date DESC
WITH company, period, metric_concept,
     collect({fact: fact, filing: filing})[0] AS latest
RETURN company.ticker AS ticker,
       period.period_year AS year,
       latest.fact.concept AS metric_concept,
       latest.fact.value AS value,
       latest.fact.unit AS unit,
       latest.fact.fact_id AS fact_id,
       latest.fact.start_date AS start_date,
       latest.fact.end_date AS end_date,
       latest.filing.accession_number AS accession_number,
       latest.filing.filing_date AS filing_date
""".strip(),
        parameters=[
            _scalar("ticker", "MSFT"),
            _scalar("year", "2026"),
        ],
    )

    generator = FakeGenerator(
        [
            bad,
            repaired,
        ]
    )

    store = MarginStore()

    result = GraphQAService(
        store=store,  # type: ignore[arg-type]
        generator=generator,
        max_attempts=2,
    ).ask("What was Microsoft's net margin in 2026?")

    assert result.attempts == 2
    assert len(generator.calls) == 2

    # The invalid first plan must fail before Neo4j execution.
    assert len(store.executed) == 1

    error = str(generator.calls[1]["error"])
    assert "missing required raw metric(s)" in error
    assert "revenue" in error
