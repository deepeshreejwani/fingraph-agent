"""Neo4j schema and constraint queries."""

CONSTRAINT_QUERIES = (
    """
    CREATE CONSTRAINT company_cik_unique IF NOT EXISTS
    FOR (company:Company)
    REQUIRE company.cik IS UNIQUE
    """,
    """
    CREATE CONSTRAINT filing_accession_unique IF NOT EXISTS
    FOR (filing:Filing)
    REQUIRE filing.accession_number IS UNIQUE
    """,
    """
    CREATE CONSTRAINT section_id_unique IF NOT EXISTS
    FOR (section:Section)
    REQUIRE section.section_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT fact_id_unique IF NOT EXISTS
    FOR (fact:FinancialFact)
    REQUIRE fact.fact_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT fiscal_period_id_unique IF NOT EXISTS
    FOR (period:FiscalPeriod)
    REQUIRE period.period_id IS UNIQUE
    """,
)

INDEX_QUERIES = (
    """
    CREATE INDEX company_ticker_index IF NOT EXISTS
    FOR (company:Company)
    ON (company.ticker)
    """,
    """
    CREATE INDEX filing_year_index IF NOT EXISTS
    FOR (filing:Filing)
    ON (filing.fiscal_year)
    """,
    """
    CREATE INDEX section_name_index IF NOT EXISTS
    FOR (section:Section)
    ON (section.section_name)
    """,
    """
    CREATE INDEX fact_concept_index IF NOT EXISTS
    FOR (fact:FinancialFact)
    ON (fact.concept)
    """,
    """
    CREATE INDEX fact_fiscal_year_index IF NOT EXISTS
    FOR (fact:FinancialFact)
    ON (fact.fiscal_year)
    """,
)
