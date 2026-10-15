"""Focused multi-company vector retrieval tests."""

from fingraph.agent.tools import VectorEvidenceTool
from fingraph.retrieval.models import RetrievalFilter, RetrievalResult


def _result(
    *,
    ticker: str,
    suffix: str,
    rank: int,
    score: float,
    method: str,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{ticker}-{suffix}",
        filing_id=f"filing-{ticker}-{suffix}",
        section_id=f"section-{ticker}",
        cik=f"cik-{ticker}",
        ticker=ticker,
        fiscal_year=2025,
        section_name="Item 1A. Risk Factors",
        text=f"{ticker} evidence {suffix}",
        source_url=f"https://sec.example/{ticker}/{suffix}",
        score=score,
        method=method,
        rank=rank,
    )


class FakeStore:
    """Small deterministic Qdrant test double."""

    def __init__(self) -> None:
        self.seen_tickers: list[str] = []

    def sparse_search(
        self,
        query: str,
        *,
        filters: RetrievalFilter | None = None,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        assert query
        assert limit >= 1
        assert filters is not None
        assert len(filters.tickers) == 1

        ticker = filters.tickers[0]
        self.seen_tickers.append(ticker)

        return [
            _result(
                ticker=ticker,
                suffix="best",
                rank=1,
                score=1.0,
                method="sparse",
            ),
            _result(
                ticker=ticker,
                suffix="extra",
                rank=2,
                score=0.9,
                method="sparse",
            ),
        ]

    def dense_search(
        self,
        query: str,
        *,
        filters: RetrievalFilter | None = None,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        assert query
        assert limit >= 1
        assert filters is not None
        assert len(filters.tickers) == 1

        ticker = filters.tickers[0]
        self.seen_tickers.append(ticker)

        return [
            _result(
                ticker=ticker,
                suffix="best",
                rank=1,
                score=1.0,
                method="dense",
            ),
            _result(
                ticker=ticker,
                suffix="extra",
                rank=2,
                score=0.9,
                method="dense",
            ),
        ]


def test_multi_company_vector_guarantees_requested_ticker_coverage() -> None:
    store = FakeStore()

    tool = VectorEvidenceTool(
        store,  # type: ignore[arg-type]
        limit=5,
    )

    evidence = tool(
        "Compare AI infrastructure risks reported by Microsoft, Amazon, Meta and Alphabet."
    )

    assert len(evidence) == 5

    assert {item.ticker for item in evidence} >= {
        "MSFT",
        "AMZN",
        "META",
        "GOOGL",
    }

    assert {item.section_name for item in evidence} == {
        "Item 1A. Risk Factors",
    }

    assert set(store.seen_tickers) == {
        "MSFT",
        "AMZN",
        "META",
        "GOOGL",
    }


def test_multi_company_vector_preserves_global_evidence_budget() -> None:
    tool = VectorEvidenceTool(
        FakeStore(),  # type: ignore[arg-type]
        limit=3,
    )

    evidence = tool("Compare risks reported by Microsoft, Amazon, Meta and Alphabet.")

    assert len(evidence) == 3
    assert [item.rank for item in evidence] == [1, 2, 3]
