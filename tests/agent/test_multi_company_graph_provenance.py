"""Regression tests for multi-company graph provenance and citations."""

from types import SimpleNamespace
from typing import Any, cast

from fingraph.agent import Evidence
from fingraph.agent.tools import (
    GRAPH_PERIOD_PROVENANCE_QUERY,
    GRAPH_ROW_PROVENANCE_QUERY,
    _graph_filing_provenance,
)
from fingraph.answer.models import SynthesisPlan
from fingraph.answer.synthesis import build_answer_citations
from fingraph.graph.qa import GraphQAService


class FakeStore:
    """Minimal Neo4j-store double for provenance tests."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    def execute_read(
        self,
        query: str,
        *,
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "query": query,
                "parameters": parameters,
            }
        )

        return self.rows


def _as_graph_service(store: FakeStore) -> GraphQAService:
    """Cast the lightweight service double to the production service type."""

    return cast(
        GraphQAService,
        SimpleNamespace(store=store),
    )


def test_multi_company_row_provenance_uses_exact_ticker_and_filing_date() -> None:
    store = FakeStore(
        [
            {
                "ticker": "AMZN",
                "accession_number": "amzn-2025",
                "source_url": "https://sec.example/amzn",
                "filing_date": "2026-02-06",
            },
            {
                "ticker": "GOOGL",
                "accession_number": "googl-2025",
                "source_url": "https://sec.example/googl",
                "filing_date": "2026-02-05",
            },
            {
                "ticker": "MSFT",
                "accession_number": "msft-2025",
                "source_url": "https://sec.example/msft",
                "filing_date": "2026-07-29",
            },
        ]
    )

    service = _as_graph_service(store)

    filings = _graph_filing_provenance(
        service=service,
        tickers=("AMZN", "GOOGL", "MSFT"),
        rows=[
            {
                "ticker": "AMZN",
                "year": 2025,
                "operating_income": 1.0,
                "filing_date": "2026-02-06",
            },
            {
                "ticker": "GOOGL",
                "year": 2025,
                "operating_income": 2.0,
                "filing_date": "2026-02-05",
            },
            {
                "ticker": "MSFT",
                "year": 2025,
                "operating_income": 3.0,
                "filing_date": "2026-07-29",
            },
        ],
    )

    assert len(store.calls) == 1
    assert store.calls[0]["query"] == GRAPH_ROW_PROVENANCE_QUERY
    assert store.calls[0]["parameters"] == {
        "row_refs": [
            {"ticker": "AMZN", "filing_date": "2026-02-06"},
            {"ticker": "GOOGL", "filing_date": "2026-02-05"},
            {"ticker": "MSFT", "filing_date": "2026-07-29"},
        ]
    }

    assert [item["ticker"] for item in filings] == [
        "AMZN",
        "GOOGL",
        "MSFT",
    ]


def test_multi_company_period_provenance_is_batched_when_row_dates_absent() -> None:
    store = FakeStore(
        [
            {
                "ticker": "AMZN",
                "period_year": 2025,
                "accession_number": "amzn-2025",
                "source_url": "https://sec.example/amzn",
                "filing_date": "2026-02-06",
            },
            {
                "ticker": "GOOGL",
                "period_year": 2025,
                "accession_number": "googl-2025",
                "source_url": "https://sec.example/googl",
                "filing_date": "2026-02-05",
            },
            {
                "ticker": "MSFT",
                "period_year": 2025,
                "accession_number": "msft-2025",
                "source_url": "https://sec.example/msft",
                "filing_date": "2026-07-29",
            },
        ]
    )

    service = _as_graph_service(store)

    _graph_filing_provenance(
        service=service,
        tickers=("AMZN", "GOOGL", "MSFT"),
        rows=[
            {"ticker": "AMZN", "year": 2025},
            {"ticker": "GOOGL", "year": 2025},
            {"ticker": "MSFT", "year": 2025},
        ],
    )

    assert len(store.calls) == 1
    assert store.calls[0]["query"] == GRAPH_PERIOD_PROVENANCE_QUERY
    assert store.calls[0]["parameters"] == {
        "tickers": ["AMZN", "GOOGL", "MSFT"],
        "years": [2025],
    }


def test_multi_company_provenance_deduplicates_by_ticker_and_accession() -> None:
    store = FakeStore(
        [
            {
                "ticker": "MSFT",
                "period_year": 2025,
                "accession_number": "same-accession",
                "source_url": "https://sec.example/msft",
                "filing_date": "2025-07-30",
            },
            {
                "ticker": "MSFT",
                "period_year": 2025,
                "accession_number": "same-accession",
                "source_url": "https://sec.example/msft",
                "filing_date": "2025-07-30",
            },
            {
                "ticker": "AMZN",
                "period_year": 2025,
                "accession_number": "same-accession",
                "source_url": "https://sec.example/amzn",
                "filing_date": "2026-02-06",
            },
        ]
    )

    service = _as_graph_service(store)

    filings = _graph_filing_provenance(
        service=service,
        tickers=("AMZN", "MSFT"),
        rows=[
            {"ticker": "AMZN", "year": 2025},
            {"ticker": "MSFT", "year": 2025},
        ],
    )

    assert [(item["ticker"], item["accession_number"]) for item in filings] == [
        ("MSFT", "same-accession"),
        ("AMZN", "same-accession"),
    ]


def test_multi_company_graph_citations_use_filing_ticker() -> None:
    evidence = [
        Evidence(
            evidence_id="graph-multi",
            source_type="graph",
            ticker=None,
            filing_id=None,
            source_url=None,
            structured_payload={
                "rows": [
                    {
                        "ticker": "AMZN",
                        "year": 2025,
                        "operating_income": 1.0,
                    },
                    {
                        "ticker": "GOOGL",
                        "year": 2025,
                        "operating_income": 2.0,
                    },
                    {
                        "ticker": "MSFT",
                        "year": 2025,
                        "operating_income": 3.0,
                    },
                ]
            },
            provenance={
                "filings": [
                    {
                        "ticker": "AMZN",
                        "period_year": 2025,
                        "accession_number": "amzn-2025",
                        "source_url": "https://sec.example/amzn",
                    },
                    {
                        "ticker": "GOOGL",
                        "period_year": 2025,
                        "accession_number": "googl-2025",
                        "source_url": "https://sec.example/googl",
                    },
                    {
                        "ticker": "MSFT",
                        "period_year": 2025,
                        "accession_number": "msft-2025",
                        "source_url": "https://sec.example/msft",
                    },
                ]
            },
        )
    ]

    plan = SynthesisPlan(
        answer="Comparison.",
        cited_evidence_ids=["graph-multi"],
        insufficient_evidence=False,
    )

    citations = build_answer_citations(
        plan=plan,
        evidence=evidence,
    )

    assert [citation.ticker for citation in citations] == [
        "AMZN",
        "GOOGL",
        "MSFT",
    ]

    assert all(citation.filing_id for citation in citations)
    assert all(citation.source_url for citation in citations)


def test_single_company_graph_citation_falls_back_to_evidence_ticker() -> None:
    evidence = [
        Evidence(
            evidence_id="graph-single",
            source_type="graph",
            ticker="MSFT",
            filing_id="msft-2026",
            source_url="https://sec.example/msft",
            structured_payload={
                "rows": [
                    {
                        "year": 2026,
                        "revenue": 331839000000,
                    }
                ]
            },
            provenance={
                "filings": [
                    {
                        "accession_number": "msft-2026",
                        "source_url": "https://sec.example/msft",
                    }
                ]
            },
        )
    ]

    plan = SynthesisPlan(
        answer="Microsoft revenue.",
        cited_evidence_ids=["graph-single"],
        insufficient_evidence=False,
    )

    citations = build_answer_citations(
        plan=plan,
        evidence=evidence,
    )

    assert len(citations) == 1
    assert citations[0].ticker == "MSFT"
