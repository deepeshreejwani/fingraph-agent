"""Regression tests for graph evidence filing provenance."""

from types import SimpleNamespace
from typing import Any, cast

from fingraph.agent.tools import (
    GRAPH_FILING_PROVENANCE_QUERY,
    GRAPH_PERIOD_PROVENANCE_QUERY,
    GraphEvidenceTool,
)
from fingraph.graph.qa import GraphQAService


class FakeStore:
    """Minimal Neo4j store double for provenance tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute_read(
        self,
        query: str,
        *,
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        self.calls.append((query, parameters))

        if query == GRAPH_FILING_PROVENANCE_QUERY:
            return [
                {
                    "accession_number": "0001045810-26-000021",
                    "source_url": (
                        "https://www.sec.gov/"
                        "Archives/edgar/data/"
                        "1045810/"
                        "000104581026000021/"
                        "nvda-20260125.htm"
                    ),
                    "filing_date": "2026-02-25",
                }
            ]

        if query == GRAPH_PERIOD_PROVENANCE_QUERY:
            return [
                {
                    "accession_number": "0001045810-26-000021",
                    "source_url": (
                        "https://www.sec.gov/"
                        "Archives/edgar/data/"
                        "1045810/"
                        "000104581026000021/"
                        "nvda-20260125.htm"
                    ),
                    "filing_date": "2026-02-25",
                },
                {
                    # Simulate the same latest filing
                    # supporting multiple economic periods.
                    "accession_number": "0001045810-26-000021",
                    "source_url": (
                        "https://www.sec.gov/"
                        "Archives/edgar/data/"
                        "1045810/"
                        "000104581026000021/"
                        "nvda-20260125.htm"
                    ),
                    "filing_date": "2026-02-25",
                },
            ]

        raise AssertionError("Unexpected provenance query.")


class FakeGraphService:
    """Minimal Graph QA service double."""

    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
    ) -> None:
        self.store = FakeStore()
        self.rows = rows

    def ask(
        self,
        question: str,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            question=question,
            cypher="MATCH (...) RETURN ...",
            parameters={},
            rows=self.rows,
            attempts=1,
        )


def _as_graph_service(
    service: FakeGraphService,
) -> GraphQAService:
    """Cast a lightweight test double to the production service protocol."""

    return cast(GraphQAService, service)


def test_graph_provenance_uses_filing_date_when_present() -> None:
    service = FakeGraphService(
        rows=[
            {
                "year": 2026,
                "net_income": 120067000000.0,
                "filing_date": "2026-02-25",
            }
        ]
    )

    tool = GraphEvidenceTool(_as_graph_service(service))

    evidence = tool("What was NVIDIA net income in 2026?")[0]

    assert len(service.store.calls) == 1

    query, parameters = service.store.calls[0]

    assert query == GRAPH_FILING_PROVENANCE_QUERY

    assert parameters == {
        "tickers": ["NVDA"],
        "filing_dates": ["2026-02-25"],
    }

    assert evidence.filing_id == "0001045810-26-000021"

    assert evidence.source_url == (
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm"
    )


def test_graph_provenance_falls_back_to_economic_years() -> None:
    service = FakeGraphService(
        rows=[
            {
                "year": 2024,
                "net_income": 29760000000.0,
            },
            {
                "year": 2025,
                "net_income": 72880000000.0,
            },
        ]
    )

    tool = GraphEvidenceTool(_as_graph_service(service))

    evidence = tool("By what percentage did NVIDIA net income change from 2024 to 2025?")[0]

    assert len(service.store.calls) == 1

    query, parameters = service.store.calls[0]

    assert query == GRAPH_PERIOD_PROVENANCE_QUERY

    assert parameters == {
        "tickers": ["NVDA"],
        "years": [
            2024,
            2025,
        ],
    }

    assert evidence.filing_id == "0001045810-26-000021"

    assert evidence.source_url == (
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm"
    )

    filings = evidence.provenance["filings"]

    assert len(filings) == 1

    assert filings[0]["accession_number"] == "0001045810-26-000021"
