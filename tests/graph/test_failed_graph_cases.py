"""Focused regression tests for failed graph evidence cases 093 and 130."""

from typing import Any, cast

from fingraph.graph.qa.models import CypherParameter, CypherPlan
from fingraph.graph.qa.service import GraphQAService
from fingraph.storage.neo4j import Neo4jStore


class FakeGenerator:
    """Return deterministic plans in sequence."""

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


def _revenue_row(
    *,
    ticker: str,
    value: float,
    fact_id: str,
    accession_number: str,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "year": 2024,
        "metric_concept": "RevenueFromContractWithCustomerExcludingAssessedTax",
        "value": value,
        "unit": "USD",
        "fact_id": fact_id,
        "start_date": "2023-07-01",
        "end_date": "2024-06-30",
        "accession_number": accession_number,
        "filing_date": "2025-02-07",
    }


class FakeStore:
    """Minimal Neo4j store double."""

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

        if "Liabilities" in cypher:
            return [
                {
                    "ticker": "AMZN",
                    "year": 2025,
                    "metric_concept": "Liabilities",
                    "value": 0.0,
                    "unit": "USD",
                    "fact_id": "amzn-liabilities-2025",
                    "start_date": None,
                    "end_date": "2025-12-31",
                    "accession_number": "amzn-2025-10k",
                    "filing_date": "2026-02-06",
                }
            ]

        if "company.ticker AS ticker" in cypher:
            return [
                _revenue_row(
                    ticker="AAPL",
                    value=391035000000.0,
                    fact_id="aapl-revenue-2024",
                    accession_number="aapl-2024-10k",
                ),
                _revenue_row(
                    ticker="AMZN",
                    value=637959000000.0,
                    fact_id="amzn-revenue-2024",
                    accession_number="amzn-2024-10k",
                ),
            ]

        return []


def _scalar(name: str, value: str) -> CypherParameter:
    return CypherParameter(
        name=name,
        kind="scalar",
        scalar=value,
        values=[],
    )


def test_liabilities_question_repairs_mixed_assets_query() -> None:
    bad = CypherPlan(
        cypher="""
MATCH (company:Company {ticker: $ticker})
MATCH (company)-[:REPORTS]->(fact:FinancialFact)-[:FOR_PERIOD]->(period:FiscalPeriod)
MATCH (filing:Filing)-[:REPORTED_FACT]->(fact)
WHERE fact.concept IN ["Assets", "Liabilities"]
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
            _scalar("ticker", "AMZN"),
            _scalar("year", "2025"),
        ],
    )

    repaired = CypherPlan(
        cypher="""
MATCH (company:Company {ticker: $ticker})
MATCH (company)-[:REPORTS]->(fact:FinancialFact)-[:FOR_PERIOD]->(period:FiscalPeriod)
MATCH (filing:Filing)-[:REPORTED_FACT]->(fact)
WHERE fact.concept IN ["Liabilities"]
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
            _scalar("ticker", "AMZN"),
            _scalar("year", "2025"),
        ],
    )

    generator = FakeGenerator([bad, repaired])
    store = FakeStore()

    result = GraphQAService(
        store=cast(Neo4jStore, store),
        generator=generator,
        max_attempts=2,
    ).ask("What were Amazon total liabilities in 2025?")

    assert result.attempts == 2
    assert len(generator.calls) == 2
    assert len(store.executed) == 1
    assert "Liabilities" in result.cypher
    assert '"Assets"' not in result.cypher
    assert "outside requested metric" in str(generator.calls[1]["error"])


def test_multi_company_financial_query_repairs_missing_ticker_rows() -> None:
    bad = CypherPlan(
        cypher="""
MATCH (company:Company)-[:REPORTS]->(fact:FinancialFact)-[:FOR_PERIOD]->(period:FiscalPeriod)
MATCH (filing:Filing)-[:REPORTED_FACT]->(fact)
WHERE company.ticker IN $tickers
  AND fact.concept IN ["RevenueFromContractWithCustomerExcludingAssessedTax"]
  AND period.period_year = toInteger($year)
RETURN period.period_year AS year,
       fact.concept AS metric_concept,
       fact.value AS value,
       fact.unit AS unit,
       fact.fact_id AS fact_id,
       fact.start_date AS start_date,
       fact.end_date AS end_date,
       filing.accession_number AS accession_number,
       filing.filing_date AS filing_date
ORDER BY year
""".strip(),
        parameters=[
            CypherParameter(
                name="tickers",
                kind="list",
                scalar="",
                values=["AAPL", "AMZN"],
            ),
            _scalar("year", "2024"),
        ],
    )

    repaired = CypherPlan(
        cypher="""
MATCH (company:Company)-[:REPORTS]->(fact:FinancialFact)-[:FOR_PERIOD]->(period:FiscalPeriod)
MATCH (filing:Filing)-[:REPORTED_FACT]->(fact)
WHERE company.ticker IN $tickers
  AND fact.concept IN ["RevenueFromContractWithCustomerExcludingAssessedTax"]
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
ORDER BY ticker
""".strip(),
        parameters=[
            CypherParameter(
                name="tickers",
                kind="list",
                scalar="",
                values=["AAPL", "AMZN"],
            ),
            _scalar("year", "2024"),
        ],
    )

    generator = FakeGenerator([bad, repaired])
    store = FakeStore()

    result = GraphQAService(
        store=cast(Neo4jStore, store),
        generator=generator,
        max_attempts=2,
    ).ask("Compare Apple and Amazon revenue in 2024.")

    assert result.attempts == 2
    assert {row["ticker"] for row in result.rows} == {"AAPL", "AMZN"}
    assert "Missing projection(s): ticker" in str(generator.calls[1]["error"])
