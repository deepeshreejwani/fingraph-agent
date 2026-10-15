"""Controlled schema contract for Text-to-Cypher."""

from fingraph.company_registry import (
    canonical_company_tickers,
    company_schema_lines,
)

CANONICAL_COMPANY_TICKERS: dict[
    str,
    tuple[
        str,
        ...,
    ],
] = canonical_company_tickers()


CANONICAL_FINANCIAL_METRICS: dict[
    str,
    tuple[
        str,
        ...,
    ],
] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "net_income": (
        "NetIncomeLoss",
        "ProfitLoss",
    ),
    "operating_income": ("OperatingIncomeLoss",),
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
    "cash": (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ),
}


DERIVED_FINANCIAL_METRICS: dict[
    str,
    tuple[
        str,
        ...,
    ],
] = {
    "operating_margin": (
        "operating_income",
        "revenue",
    ),
    "net_margin": (
        "net_income",
        "revenue",
    ),
}


GRAPH_SCHEMA = """
NODE Company
Properties:
- cik: string, unique
- ticker: string
- name: string

Available companies and canonical tickers:
__COMPANY_SCHEMA_LINES__

NODE Filing
Properties:
- accession_number: string, unique
- form: string
- filing_date: date
- report_date: date or null
- fiscal_year: integer or null
- source_url: string
- content_hash: string

NODE Section
Properties:
- section_id: string, unique
- section_name: string

NODE FinancialFact
Properties:
- fact_id: string, unique
- taxonomy: string
- concept: string
- value: float
- unit: string
- start_date: date or null
- end_date: date
- fiscal_year: integer or null
- fiscal_period: string or null
- accession_number: string
- filed_date: date or null
- frame: string or null

NODE FiscalPeriod
Properties:
- period_id: string, unique
- start_date: date or null
- end_date: date
- period_year: integer

RELATIONSHIPS:
(:Company)-[:FILED]->(:Filing)
(:Filing)-[:HAS_SECTION]->(:Section)
(:Company)-[:REPORTS]->(:FinancialFact)
(:Filing)-[:REPORTED_FACT]->(:FinancialFact)
(:FinancialFact)-[:FOR_PERIOD]->(:FiscalPeriod)

CANONICAL FINANCIAL METRICS:

revenue:
- RevenueFromContractWithCustomerExcludingAssessedTax
- Revenues
- SalesRevenueNet

net_income:
- NetIncomeLoss
- ProfitLoss

operating_income:
- OperatingIncomeLoss

total_assets:
- Assets

total_liabilities:
- Liabilities

cash:
- CashAndCashEquivalentsAtCarryingValue
- CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents

DERIVED FINANCIAL METRICS:

operating_margin:
- retrieve raw operating_income
- retrieve raw revenue
- do NOT calculate the margin in Cypher

net_margin:
- retrieve raw net_income
- retrieve raw revenue
- do NOT calculate the margin in Cypher

IMPORTANT RULES:

1. Never invent XBRL concept names.
2. Use only concepts listed above for these canonical metrics.
3. Company-name references must be converted to their canonical ticker.
4. Use FiscalPeriod.period_year for the actual economic year.
5. FinancialFact.fiscal_year is SEC filing-context metadata and is NOT
   necessarily the year represented by the fact.
6. Comparative facts may appear in multiple filings.
7. For one observation per economic period, prefer the observation from
   the latest Filing.filing_date.
8. Filing provenance should come from:
   (:Filing)-[:REPORTED_FACT]->(:FinancialFact)
9. Do not calculate ratios, growth rates, percentages, CAGR, margins, or
   other derived financial metrics in Cypher. Return raw values for
   deterministic Python calculation.
10. Every financial query MUST return direct raw-fact identity fields:
    company.ticker AS ticker
    FiscalPeriod.period_year AS year
    fact.concept AS metric_concept
    fact.value AS value
    fact.unit AS unit
    fact.fact_id AS fact_id
    fact.start_date AS start_date
    fact.end_date AS end_date
    filing.accession_number AS accession_number
11. For annual flow metrics (revenue, net income, operating income), return
    annual-duration facts, not quarterly observations embedded in a 10-K.
12. A derived metric that depends on multiple raw financial metrics MUST
    return every required raw metric.
13. When multiple raw metrics are requested for the same company and
    economic period, latest-filing selection MUST be performed separately
    for each fact.concept. Do not collapse different concepts into one
    company-period collection.

RECOMMENDED PATTERN FOR ONE FACT PER ECONOMIC PERIOD:

MATCH (company:Company {ticker: $ticker})
MATCH (company)-[:REPORTS]->(fact:FinancialFact)
MATCH (fact)-[:FOR_PERIOD]->(period:FiscalPeriod)
MATCH (filing:Filing)-[:REPORTED_FACT]->(fact)

WHERE fact.concept IN [...]
  AND period.period_year >= ...
  AND period.period_year <= ...

WITH company, period, fact, filing
ORDER BY period.period_year ASC, filing.filing_date DESC

WITH
    company,
    period,
    collect({
        fact: fact,
        filing: filing
    })[0] AS latest

RETURN
    company.ticker AS ticker,
    period.period_year AS year,
    latest.fact.concept AS metric_concept,
    latest.fact.value AS value,
    latest.fact.unit AS unit,
    latest.fact.fact_id AS fact_id,
    latest.fact.start_date AS start_date,
    latest.fact.end_date AS end_date,
    latest.filing.accession_number AS accession_number,
    latest.filing.filing_date AS filing_date

RECOMMENDED PATTERN FOR MULTIPLE RAW METRICS IN THE SAME PERIOD:

MATCH (company:Company {ticker: $ticker})
MATCH (company)-[:REPORTS]->(fact:FinancialFact)
MATCH (fact)-[:FOR_PERIOD]->(period:FiscalPeriod)
MATCH (filing:Filing)-[:REPORTED_FACT]->(fact)

WHERE fact.concept IN [...]
  AND period.period_year = toInteger($year)

WITH company, period, fact.concept AS metric_concept, fact, filing
ORDER BY period.period_year ASC, metric_concept ASC, filing.filing_date DESC

WITH
    company,
    period,
    metric_concept,
    collect({
        fact: fact,
        filing: filing
    })[0] AS latest

RETURN
    company.ticker AS ticker,
    period.period_year AS year,
    latest.fact.concept AS metric_concept,
    latest.fact.value AS value,
    latest.fact.unit AS unit,
    latest.fact.fact_id AS fact_id,
    latest.fact.start_date AS start_date,
    latest.fact.end_date AS end_date,
    latest.filing.accession_number AS accession_number,
    latest.filing.filing_date AS filing_date
""".strip().replace(
    "__COMPANY_SCHEMA_LINES__",
    company_schema_lines(),
)


ALLOWED_NODE_LABELS = frozenset(
    {
        "Company",
        "Filing",
        "Section",
        "FinancialFact",
        "FiscalPeriod",
    }
)


ALLOWED_NODE_PROPERTIES: dict[str, frozenset[str]] = {
    "Company": frozenset(
        {
            "cik",
            "ticker",
            "name",
        }
    ),
    "Filing": frozenset(
        {
            "accession_number",
            "form",
            "filing_date",
            "report_date",
            "fiscal_year",
            "source_url",
            "content_hash",
        }
    ),
    "Section": frozenset(
        {
            "section_id",
            "section_name",
        }
    ),
    "FinancialFact": frozenset(
        {
            "fact_id",
            "taxonomy",
            "concept",
            "value",
            "unit",
            "start_date",
            "end_date",
            "fiscal_year",
            "fiscal_period",
            "accession_number",
            "filed_date",
            "frame",
        }
    ),
    "FiscalPeriod": frozenset(
        {
            "period_id",
            "start_date",
            "end_date",
            "period_year",
        }
    ),
}


ALLOWED_RELATIONSHIP_TYPES = frozenset(
    {
        "FILED",
        "HAS_SECTION",
        "REPORTS",
        "REPORTED_FACT",
        "FOR_PERIOD",
    }
)
