"""Load canonical FinGraph data into Neo4j."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fingraph.ingestion.publication import (
    published_artifact_paths,
)
from fingraph.storage.neo4j.store import (
    Neo4jStore,
)
from fingraph.utils.ids import (
    stable_id,
)

PROCESSED_ROOT = Path("data/processed/sec")


COMPANY_QUERY = """
UNWIND $rows AS row

MERGE (company:Company {
    cik: row.cik
})

SET
    company.ticker = row.ticker,
    company.name = row.name,
    company.sic = row.sic,
    company.industry = row.industry
""".strip()


FILING_QUERY = """
UNWIND $rows AS row

MATCH (company:Company {
    cik: row.cik
})

MERGE (filing:Filing {
    accession_number: row.accession_number
})

SET
    filing.form = row.form,
    filing.filing_date = date(row.filing_date),

    filing.report_date = CASE
        WHEN row.report_date IS NULL
        THEN NULL
        ELSE date(row.report_date)
    END,

    filing.fiscal_year = row.fiscal_year,
    filing.source_url = row.source_url,
    filing.content_hash = row.content_hash

MERGE (company)-[:FILED]->(filing)
""".strip()


SECTION_QUERY = """
UNWIND $rows AS row

MATCH (filing:Filing {
    accession_number: row.filing_id
})

MERGE (section:Section {
    section_id: row.section_id
})

SET
    section.section_name = row.section_name

MERGE (filing)-[:HAS_SECTION]->(section)
""".strip()


PERIOD_QUERY = """
UNWIND $rows AS row

MERGE (period:FiscalPeriod {
    period_id: row.period_id
})

SET
    period.start_date = CASE
        WHEN row.start_date IS NULL
        THEN NULL
        ELSE date(row.start_date)
    END,

    period.end_date = date(
        row.end_date
    ),

    period.period_year =
        row.period_year
""".strip()


FACT_QUERY = """
UNWIND $rows AS row

MATCH (company:Company {
    cik: row.cik
})

MATCH (period:FiscalPeriod {
    period_id: row.period_id
})

MATCH (filing:Filing {
    accession_number: row.accession_number
})

MERGE (fact:FinancialFact {
    fact_id: row.fact_id
})

SET
    fact.taxonomy = row.taxonomy,
    fact.concept = row.concept,
    fact.value = row.value,
    fact.unit = row.unit,

    fact.start_date = CASE
        WHEN row.start_date IS NULL
        THEN NULL
        ELSE date(row.start_date)
    END,

    fact.end_date = date(
        row.end_date
    ),

    fact.fiscal_year =
        row.fiscal_year,

    fact.fiscal_period =
        row.fiscal_period,

    fact.accession_number =
        row.accession_number,

    fact.filed_date = CASE
        WHEN row.filed_date IS NULL
        THEN NULL
        ELSE date(row.filed_date)
    END,

    fact.frame = row.frame

MERGE (company)-[:REPORTS]->(fact)

MERGE (filing)-[:REPORTED_FACT]->(
    fact
)

MERGE (fact)-[:FOR_PERIOD]->(
    period
)
""".strip()


PRUNE_FACTS_QUERY = """
MATCH (node:FinancialFact)
WHERE NOT node.fact_id IN $ids
DETACH DELETE node
""".strip()


PRUNE_SECTIONS_QUERY = """
MATCH (node:Section)
WHERE NOT node.section_id IN $ids
DETACH DELETE node
""".strip()


PRUNE_FILINGS_QUERY = """
MATCH (node:Filing)
WHERE NOT node.accession_number IN $ids
DETACH DELETE node
""".strip()


PRUNE_PERIODS_QUERY = """
MATCH (node:FiscalPeriod)
WHERE NOT node.period_id IN $ids
DETACH DELETE node
""".strip()


PRUNE_COMPANIES_QUERY = """
MATCH (node:Company)
WHERE NOT node.cik IN $ids
DETACH DELETE node
""".strip()


COUNT_QUERIES = {
    "companies": ("MATCH (n:Company) RETURN count(n) AS count"),
    "filings": ("MATCH (n:Filing) RETURN count(n) AS count"),
    "sections": ("MATCH (n:Section) RETURN count(n) AS count"),
    "financial_facts": ("MATCH (n:FinancialFact) RETURN count(n) AS count"),
    "fiscal_periods": ("MATCH (n:FiscalPeriod) RETURN count(n) AS count"),
}


ORPHAN_QUERIES = {
    "orphan_filings": """
MATCH (filing:Filing)
WHERE NOT EXISTS {
    MATCH (:Company)-[:FILED]->(filing)
}
RETURN count(filing) AS count
""".strip(),
    "orphan_sections": """
MATCH (section:Section)
WHERE NOT EXISTS {
    MATCH (:Filing)-[:HAS_SECTION]->(section)
}
RETURN count(section) AS count
""".strip(),
    "orphan_facts_company": """
MATCH (fact:FinancialFact)
WHERE NOT EXISTS {
    MATCH (:Company)-[:REPORTS]->(fact)
}
RETURN count(fact) AS count
""".strip(),
    "orphan_facts_filing": """
MATCH (fact:FinancialFact)
WHERE NOT EXISTS {
    MATCH (:Filing)-[:REPORTED_FACT]->(fact)
}
RETURN count(fact) AS count
""".strip(),
    "orphan_facts_period": """
MATCH (fact:FinancialFact)
WHERE NOT EXISTS {
    MATCH (fact)-[:FOR_PERIOD]->(:FiscalPeriod)
}
RETURN count(fact) AS count
""".strip(),
    "orphan_periods": """
MATCH (period:FiscalPeriod)
WHERE NOT EXISTS {
    MATCH (:FinancialFact)-[:FOR_PERIOD]->(period)
}
RETURN count(period) AS count
""".strip(),
}


def _artifact_paths(
    filename: str,
) -> list[Path]:
    """Return published canonical artifact paths in deterministic ticker order."""

    return published_artifact_paths(
        PROCESSED_ROOT,
        filename,
    )


def _read_json(
    path: Path,
) -> Any:
    """Read one processed JSON artifact."""

    return json.loads(path.read_text(encoding="utf-8"))


def company_metadata_by_ticker() -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:
    """Load canonical company metadata artifacts."""

    grouped: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for path in _artifact_paths("company.json"):
        ticker = path.parent.name

        payload = _read_json(path)

        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(f"Invalid company metadata artifact: {path}")

        grouped[ticker] = payload

    return grouped


def chunks_by_ticker() -> dict[
    str,
    list[
        dict[
            str,
            Any,
        ]
    ],
]:
    """Load chunk metadata grouped by ticker."""

    grouped: dict[
        str,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = {}

    for path in _artifact_paths("chunks.json"):
        ticker = path.parent.name

        grouped[ticker] = _read_json(path)

    return grouped


def filings_by_ticker() -> dict[
    str,
    list[
        dict[
            str,
            Any,
        ]
    ],
]:
    """Load canonical filing metadata grouped by ticker."""

    grouped: dict[
        str,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = {}

    for path in _artifact_paths("filings.json"):
        ticker = path.parent.name

        grouped[ticker] = _read_json(path)

    return grouped


def facts_by_ticker() -> dict[
    str,
    list[
        dict[
            str,
            Any,
        ]
    ],
]:
    """Load financial facts grouped by ticker."""

    grouped: dict[
        str,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = {}

    for path in _artifact_paths("financial_facts.json"):
        ticker = path.parent.name

        grouped[ticker] = _read_json(path)

    return grouped


def company_rows() -> list[
    dict[
        str,
        Any,
    ]
]:
    """Build Company rows from canonical metadata with legacy fallback."""

    metadata = company_metadata_by_ticker()

    chunks = chunks_by_ticker()

    tickers = sorted(set(metadata) | set(chunks))

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for ticker in tickers:
        company = metadata.get(ticker)

        if company is not None:
            cik = company.get("cik")

            if cik is None:
                raise ValueError(f"Company metadata missing CIK: {ticker}")

            rows.append(
                {
                    "cik": str(cik),
                    "ticker": ticker,
                    "name": (company.get("name") or ticker),
                    "sic": (company.get("sic")),
                    "industry": (company.get("industry")),
                }
            )

            continue

        ticker_chunks = chunks.get(
            ticker,
            [],
        )

        if not ticker_chunks:
            continue

        first = ticker_chunks[0]

        rows.append(
            {
                "cik": first["cik"],
                "ticker": ticker,
                "name": ticker,
                "sic": None,
                "industry": None,
            }
        )

    return rows


def load_companies(
    store: Neo4jStore,
) -> int:
    """Load deterministic Company nodes."""

    rows = company_rows()

    store.execute_batch(
        COMPANY_QUERY,
        rows,
    )

    return len(rows)


def load_filings_and_sections(
    store: Neo4jStore,
) -> tuple[
    int,
    int,
]:
    """Load canonical Filing and Section graph entities."""

    filing_groups = filings_by_ticker()

    chunk_groups = chunks_by_ticker()

    filing_rows = [filing for ticker_filings in filing_groups.values() for filing in ticker_filings]

    sections: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for ticker_chunks in chunk_groups.values():
        for chunk in ticker_chunks:
            section_id = chunk["section_id"]

            sections.setdefault(
                section_id,
                {
                    "section_id": (section_id),
                    "filing_id": (chunk["filing_id"]),
                    "section_name": (chunk["section_name"]),
                },
            )

    store.execute_batch(
        FILING_QUERY,
        filing_rows,
    )

    store.execute_batch(
        SECTION_QUERY,
        list(sections.values()),
    )

    return (
        len(filing_rows),
        len(sections),
    )


def period_id_for_fact(
    fact: dict[
        str,
        Any,
    ],
) -> str:
    """Identify an actual economic reporting period."""

    return stable_id(
        "period",
        fact["cik"],
        fact.get("start_date"),
        fact["end_date"],
    )


def _fact_and_period_rows() -> tuple[
    list[
        dict[
            str,
            Any,
        ]
    ],
    dict[
        str,
        dict[
            str,
            Any,
        ],
    ],
]:
    """Materialize normalized fact and economic-period rows."""

    grouped = facts_by_ticker()

    periods: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    facts: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for ticker_facts in grouped.values():
        for fact in ticker_facts:
            period_id = period_id_for_fact(fact)

            end_date = str(fact["end_date"])

            periods.setdefault(
                period_id,
                {
                    "period_id": (period_id),
                    "start_date": (fact.get("start_date")),
                    "end_date": (end_date),
                    "period_year": (int(end_date[:4])),
                },
            )

            facts.append(
                {
                    **fact,
                    "period_id": (period_id),
                    "value": float(fact["value"]),
                }
            )

    return (
        facts,
        periods,
    )


def load_facts(
    store: Neo4jStore,
) -> tuple[
    int,
    int,
]:
    """Load normalized XBRL facts and economic periods."""

    (
        facts,
        periods,
    ) = _fact_and_period_rows()

    store.execute_batch(
        PERIOD_QUERY,
        list(periods.values()),
    )

    store.execute_batch(
        FACT_QUERY,
        facts,
    )

    return (
        len(facts),
        len(periods),
    )


def expected_graph_ids() -> dict[
    str,
    list[str],
]:
    """Return canonical IDs currently represented by published artifacts."""

    companies = company_rows()

    filings = filings_by_ticker()

    chunks = chunks_by_ticker()

    (
        facts,
        periods,
    ) = _fact_and_period_rows()

    return {
        "companies": sorted({str(row["cik"]) for row in companies}),
        "filings": sorted(
            {str(filing["accession_number"]) for group in filings.values() for filing in group}
        ),
        "sections": sorted(
            {str(chunk["section_id"]) for group in chunks.values() for chunk in group}
        ),
        "financial_facts": sorted({str(fact["fact_id"]) for fact in facts}),
        "fiscal_periods": sorted(periods.keys()),
    }


def expected_graph_counts() -> dict[
    str,
    int,
]:
    """Count canonical entities in published processed artifacts."""

    ids = expected_graph_ids()

    return {
        key: len(values)
        for (
            key,
            values,
        ) in ids.items()
    }


def prune_stale_graph(
    store: Neo4jStore,
) -> dict[
    str,
    int,
]:
    """Remove graph entities absent from the published canonical corpus."""

    ids = expected_graph_ids()

    required_non_empty = (
        "companies",
        "filings",
        "sections",
        "financial_facts",
        "fiscal_periods",
    )

    for key in required_non_empty:
        if not ids[key]:
            raise ValueError(f"Refusing graph prune because canonical {key} set is empty.")

    before = graph_counts(store)

    store.execute_write(
        PRUNE_FACTS_QUERY,
        parameters={"ids": ids["financial_facts"]},
    )

    store.execute_write(
        PRUNE_SECTIONS_QUERY,
        parameters={"ids": ids["sections"]},
    )

    store.execute_write(
        PRUNE_FILINGS_QUERY,
        parameters={"ids": ids["filings"]},
    )

    store.execute_write(
        PRUNE_PERIODS_QUERY,
        parameters={"ids": ids["fiscal_periods"]},
    )

    store.execute_write(
        PRUNE_COMPANIES_QUERY,
        parameters={"ids": ids["companies"]},
    )

    after = graph_counts(store)

    return {key: (before[key] - after[key]) for key in before}


def _single_count(
    store: Neo4jStore,
    query: str,
) -> int:
    """Execute one count query."""

    rows = store.execute_read(query)

    if len(rows) != 1 or "count" not in rows[0]:
        raise RuntimeError("Unexpected Neo4j count-query result.")

    return int(rows[0]["count"])


def graph_counts(
    store: Neo4jStore,
) -> dict[
    str,
    int,
]:
    """Read canonical graph node counts."""

    return {
        name: (
            _single_count(
                store,
                query,
            )
        )
        for (
            name,
            query,
        ) in COUNT_QUERIES.items()
    }


def graph_integrity_report(
    store: Neo4jStore,
) -> dict[
    str,
    Any,
]:
    """Return deterministic node-count and orphan checks."""

    counts = graph_counts(store)

    expected = expected_graph_counts()

    orphans = {
        name: (
            _single_count(
                store,
                query,
            )
        )
        for (
            name,
            query,
        ) in ORPHAN_QUERIES.items()
    }

    count_mismatches = {
        key: {
            "expected": (expected[key]),
            "actual": (counts[key]),
        }
        for key in expected
        if (counts[key] != expected[key])
    }

    orphan_total = sum(orphans.values())

    return {
        "expected_counts": (expected),
        "actual_counts": (counts),
        "count_mismatches": (count_mismatches),
        "orphans": orphans,
        "orphan_total": (orphan_total),
        "passed": (not count_mismatches and orphan_total == 0),
    }
