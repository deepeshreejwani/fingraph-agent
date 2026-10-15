"""Adapters exposing existing FinGraph systems as agent tools."""

from __future__ import annotations

import re
from typing import Any

from fingraph.agent.evidence import (
    Evidence,
)
from fingraph.company_registry import (
    extract_company_tickers,
    normalized_company_aliases,
)
from fingraph.graph.qa import (
    GraphQAService,
)
from fingraph.retrieval.hybrid import (
    reciprocal_rank_fusion,
)
from fingraph.retrieval.models import (
    RetrievalFilter,
    RetrievalResult,
)
from fingraph.storage.qdrant import (
    QdrantChunkStore,
)
from fingraph.utils.ids import (
    stable_id,
)

UNSUPPORTED_FINANCIAL_METRIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bebit\b", re.IGNORECASE),
    re.compile(r"\bebitda\b", re.IGNORECASE),
    re.compile(r"\bgross\s+profit\b", re.IGNORECASE),
    re.compile(r"\b(?:eps|earnings\s+per\s+share)\b", re.IGNORECASE),
    re.compile(r"\bfree\s+cash\s+flow\b", re.IGNORECASE),
    re.compile(r"\boperating\s+cash\s+flow\b", re.IGNORECASE),
)


_EXPLICIT_COMPANY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:describe|explain)\s+"
        r"(?P<company>[A-Z][A-Za-z0-9.&-]*(?:\s+[A-Z][A-Za-z0-9.&-]*){0,3})"
        r"(?:'s|’s)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+(?:market\s+)?risks?\s+did\s+"
        r"(?P<company>[A-Z][A-Za-z0-9.&-]*(?:\s+[A-Z][A-Za-z0-9.&-]*){0,3})"
        r"\s+describe\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+was\s+"
        r"(?P<company>[A-Z][A-Za-z0-9.&-]*(?:\s+[A-Z][A-Za-z0-9.&-]*){0,3})"
        r"(?:'s|’s)?\s+"
        r"(?:revenue|net\s+income|operating\s+income|total\s+assets?|"
        r"total\s+liabilit(?:y|ies)|cash|ebit|ebitda|gross\s+profit|"
        r"eps|earnings\s+per\s+share|free\s+cash\s+flow|"
        r"operating\s+cash\s+flow)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:risks?|business|operations?|revenue|income|assets?|"
        r"liabilit(?:y|ies)|cash)\s+(?:for|of|at|to)\s+"
        r"(?P<company>[A-Z][A-Za-z0-9.&-]*(?:\s+[A-Z][A-Za-z0-9.&-]*){0,3})"
        r"(?=\s+(?:in|during|for)\s+(?:19|20)\d{2}\b|[?.!,]|$)",
        re.IGNORECASE,
    ),
)


def _normalize_company_name(value: str) -> str:
    """Normalize a company mention for deterministic universe checks."""

    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _supported_company_names() -> set[str]:
    """Return normalized company aliases accepted by the configured universe."""

    return set(normalized_company_aliases())


def _extract_explicit_company_mentions(question: str) -> tuple[str, ...]:
    """Extract conservative explicit company mentions from supported query shapes."""

    mentions: set[str] = set()

    for pattern in _EXPLICIT_COMPANY_PATTERNS:
        for match in pattern.finditer(question):
            value = match.group("company").strip()

            if value:
                mentions.add(value)

    return tuple(sorted(mentions))


def _has_explicit_unsupported_company(question: str) -> bool:
    """Return whether a clearly named company is outside the configured universe."""

    supported = _supported_company_names()

    for mention in _extract_explicit_company_mentions(question):
        normalized = _normalize_company_name(mention)

        if not normalized:
            continue

        if normalized in supported:
            continue

        parts = tuple(part for part in re.split(r"\s+(?:and|&)\s+", normalized) if part)

        if len(parts) > 1 and all(part in supported for part in parts):
            continue

        return True

    return False


def _has_explicit_unsupported_financial_metric(question: str) -> bool:
    """Return whether the question requests a deliberately unsupported metric."""

    return any(
        pattern.search(question) is not None for pattern in UNSUPPORTED_FINANCIAL_METRIC_PATTERNS
    )


GRAPH_FILING_PROVENANCE_QUERY = """
MATCH (company:Company)-[:FILED]->(filing:Filing)
WHERE company.ticker IN $tickers
  AND toString(filing.filing_date) IN $filing_dates
RETURN
    company.ticker AS ticker,
    filing.accession_number AS accession_number,
    filing.source_url AS source_url,
    toString(filing.filing_date) AS filing_date
ORDER BY company.ticker ASC, filing.filing_date DESC
""".strip()


GRAPH_ROW_PROVENANCE_QUERY = """
UNWIND $row_refs AS row_ref
MATCH (company:Company {ticker: row_ref.ticker})-[:FILED]->(filing:Filing)
WHERE toString(filing.filing_date) = row_ref.filing_date
RETURN DISTINCT
    company.ticker AS ticker,
    filing.accession_number AS accession_number,
    filing.source_url AS source_url,
    toString(filing.filing_date) AS filing_date
ORDER BY ticker ASC, filing_date DESC
""".strip()


GRAPH_PERIOD_PROVENANCE_QUERY = """
MATCH (company:Company)-[:FILED]->(filing:Filing)
MATCH (filing)-[:REPORTED_FACT]->(fact:FinancialFact)
MATCH (fact)-[:FOR_PERIOD]->(period:FiscalPeriod)
WHERE company.ticker IN $tickers
  AND period.period_year IN [value IN $years | toInteger(value)]
WITH
    company,
    period.period_year AS period_year,
    filing
ORDER BY
    company.ticker ASC,
    period_year ASC,
    filing.filing_date DESC
WITH
    company,
    period_year,
    collect(filing)[0] AS filing
RETURN DISTINCT
    company.ticker AS ticker,
    period_year AS period_year,
    filing.accession_number AS accession_number,
    filing.source_url AS source_url,
    toString(filing.filing_date) AS filing_date
ORDER BY
    company.ticker ASC,
    period_year ASC,
    filing_date DESC
""".strip()


YEAR_PATTERN = re.compile(r"\b((?:19|20)\d{2})\b")


YEAR_RANGE_PATTERNS = (
    re.compile(
        r"\bfrom\s+"
        r"((?:19|20)\d{2})"
        r"\s+to\s+"
        r"((?:19|20)\d{2})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bbetween\s+"
        r"((?:19|20)\d{2})"
        r"\s+and\s+"
        r"((?:19|20)\d{2})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:19|20)\d{2})"
        r"\s*[-–—]\s*"
        r"((?:19|20)\d{2})\b",
        re.IGNORECASE,
    ),
)


MULTI_YEAR_TERMS = (
    "over the years",
    "across years",
    "across the years",
    "over time",
    "through time",
    "historically",
    "historical",
    "year over year",
    "year-over-year",
    "year on year",
    "year-on-year",
    "multi-year",
    "multiple years",
    "trend over",
    "trends over",
    "changed over time",
    "change over time",
    "evolved over time",
)


def _extract_tickers(
    question: str,
) -> tuple[str, ...]:
    """Infer canonical ticker filters from the question."""

    return extract_company_tickers(question)


def _extract_section_names(
    question: str,
) -> tuple[str, ...]:
    """Infer high-confidence filing-section filters."""

    normalized = question.strip().lower()

    if "market risk" in normalized:
        return ("Item 7A. Market Risk",)

    if any(
        term in normalized
        for term in (
            "risk factor",
            "risk factors",
            "risk",
            "risks",
        )
    ):
        return ("Item 1A. Risk Factors",)

    if any(
        term in normalized
        for term in (
            "md&a",
            "management discussion",
            "management said",
            "what drove",
            "drove",
            "drivers",
        )
    ):
        return ("Item 7. MD&A",)

    if any(
        term in normalized
        for term in (
            "business",
            "business model",
            "what does the company do",
        )
    ):
        return ("Item 1. Business",)

    return ()


def _extract_year_range(
    question: str,
) -> tuple[int, ...]:
    """Expand an explicit continuous year range."""

    for pattern in YEAR_RANGE_PATTERNS:
        match = pattern.search(question)

        if match is None:
            continue

        first = int(match.group(1))
        second = int(match.group(2))

        start = min(
            first,
            second,
        )

        end = max(
            first,
            second,
        )

        if end - start > 50:
            return ()

        return tuple(
            range(
                start,
                end + 1,
            )
        )

    return ()


def _extract_explicit_years(
    question: str,
) -> tuple[int, ...]:
    """Extract distinct explicit years."""

    return tuple(sorted({int(match) for match in YEAR_PATTERN.findall(question)}))


def _has_unbounded_multi_year_intent(
    question: str,
) -> bool:
    """Detect an explicit request for longitudinal narrative evidence."""

    normalized = question.strip().lower()

    return any(term in normalized for term in MULTI_YEAR_TERMS)


def _resolve_fiscal_years(
    *,
    question: str,
    tickers: tuple[str, ...],
    store: QdrantChunkStore,
) -> tuple[int, ...]:
    """
    Resolve standard temporal filtering.

    Priority:
    1. explicit continuous range;
    2. explicit year(s);
    3. unbounded longitudinal question;
    4. latest indexed filing for a single company.
    """

    range_years = _extract_year_range(question)

    if range_years:
        return range_years

    explicit_years = _extract_explicit_years(question)

    if explicit_years:
        return explicit_years

    if _has_unbounded_multi_year_intent(question):
        return ()

    if len(tickers) != 1:
        return ()

    latest = store.latest_fiscal_year(tickers[0])

    if latest is None:
        return ()

    return (latest,)


def _select_representative_years(
    *,
    years: tuple[int, ...],
    limit: int,
) -> tuple[int, ...]:
    """
    Choose deterministic chronological coverage within an evidence budget.

    The oldest and newest years are retained whenever at least two
    evidence slots are available.
    """

    if limit < 1:
        raise ValueError("limit must be positive.")

    ordered = tuple(sorted(set(years)))

    if len(ordered) <= limit:
        return ordered

    if limit == 1:
        return (ordered[-1],)

    last_index = len(ordered) - 1

    selected_indices: list[int] = []

    for position in range(limit):
        index = round((position * last_index) / (limit - 1))

        if index not in selected_indices:
            selected_indices.append(index)

    if len(selected_indices) < limit:
        for index in range(len(ordered)):
            if index in selected_indices:
                continue

            selected_indices.append(index)

            if len(selected_indices) == limit:
                break

    return tuple(ordered[index] for index in sorted(selected_indices))


def _diversify_results_by_year(
    *,
    results: list[RetrievalResult],
    target_years: tuple[int, ...],
    limit: int,
) -> list[RetrievalResult]:
    """Keep the strongest available result for each target year."""

    if limit < 1:
        raise ValueError("limit must be positive.")

    best_by_year: dict[
        int,
        RetrievalResult,
    ] = {}

    for result in results:
        year = result.fiscal_year

        if year is None or year not in target_years or year in best_by_year:
            continue

        best_by_year[year] = result

        if len(best_by_year) == len(target_years):
            break

    selected = [best_by_year[year] for year in target_years if year in best_by_year]

    if len(selected) < limit:
        selected_ids = {item.chunk_id for item in selected}

        selected_years = {item.fiscal_year for item in selected}

        for result in results:
            if result.chunk_id in selected_ids:
                continue

            if result.fiscal_year in selected_years:
                continue

            selected.append(result)
            selected_ids.add(result.chunk_id)
            selected_years.add(result.fiscal_year)

            if len(selected) == limit:
                break

    selected = selected[:limit]

    selected.sort(
        key=lambda item: (
            (item.fiscal_year if item.fiscal_year is not None else -1),
            item.rank,
        )
    )

    return [
        result.model_copy(
            update={
                "rank": rank,
            }
        )
        for rank, result in enumerate(
            selected,
            start=1,
        )
    ]


def _json_safe(
    value: Any,
) -> Any:
    """Convert external driver values into JSON-safe Python values."""

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        list,
    ):
        return [_json_safe(item) for item in value]

    if isinstance(
        value,
        tuple,
    ):
        return [_json_safe(item) for item in value]

    if isinstance(
        value,
        dict,
    ):
        return {str(key): _json_safe(item) for key, item in value.items()}

    return str(value)


def _filing_dates_from_rows(
    rows: list[dict[str, Any]],
) -> list[str]:
    """Extract filing dates already returned by Graph QA."""

    values: set[str] = set()

    for row in rows:
        raw = row.get("filing_date")

        if raw is not None:
            values.add(str(raw))

    return sorted(values)


def _row_provenance_refs(
    rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Extract exact ticker + filing-date provenance references from graph rows."""

    refs: set[tuple[str, str]] = set()

    for row in rows:
        raw_ticker = row.get("ticker")
        raw_filing_date = row.get("filing_date")

        if raw_ticker is None or raw_filing_date is None:
            continue

        refs.add(
            (
                str(raw_ticker),
                str(raw_filing_date),
            )
        )

    return [
        {
            "ticker": ticker,
            "filing_date": filing_date,
        }
        for ticker, filing_date in sorted(refs)
    ]


def _economic_years_from_rows(
    rows: list[dict[str, Any]],
) -> list[int]:
    """Extract economic years represented by structured graph rows."""

    year_keys = (
        "year",
        "fiscal_year",
        "economic_year",
        "period_year",
    )

    values: set[int] = set()

    for row in rows:
        for key in year_keys:
            raw = row.get(key)

            if raw is None:
                continue

            try:
                values.add(int(str(raw)))

            except ValueError:
                continue

            break

    return sorted(values)


def _deduplicate_filings(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate filing provenance by ticker and accession."""

    filings: list[dict[str, Any]] = []

    seen: set[
        tuple[
            str | None,
            str,
        ]
    ] = set()

    for row in rows:
        safe = _json_safe(row)

        raw_accession = safe.get("accession_number")

        if raw_accession is None:
            continue

        raw_ticker = safe.get("ticker")

        ticker = str(raw_ticker) if raw_ticker is not None else None

        accession = str(raw_accession)

        key = (
            ticker,
            accession,
        )

        if key in seen:
            continue

        seen.add(key)
        filings.append(safe)

    return filings


def _graph_filing_provenance(
    *,
    service: GraphQAService,
    tickers: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve trusted filing URLs for single- or multi-company graph results."""

    if not tickers:
        return []

    row_refs = _row_provenance_refs(rows)

    if row_refs:
        result = service.store.execute_read(
            GRAPH_ROW_PROVENANCE_QUERY,
            parameters={
                "row_refs": row_refs,
            },
        )

        return _deduplicate_filings(result)

    filing_dates = _filing_dates_from_rows(rows)

    if len(tickers) == 1 and filing_dates:
        result = service.store.execute_read(
            GRAPH_FILING_PROVENANCE_QUERY,
            parameters={
                "tickers": list(tickers),
                "filing_dates": filing_dates,
            },
        )

        return _deduplicate_filings(result)

    economic_years = _economic_years_from_rows(rows)

    if not economic_years:
        return []

    result = service.store.execute_read(
        GRAPH_PERIOD_PROVENANCE_QUERY,
        parameters={
            "tickers": list(tickers),
            "years": economic_years,
        },
    )

    return _deduplicate_filings(result)


def _diversify_results_by_ticker(
    *,
    per_ticker: dict[str, list[RetrievalResult]],
    target_tickers: tuple[str, ...],
    limit: int,
) -> list[RetrievalResult]:
    """Guarantee bounded company coverage before spending spare evidence slots."""

    if limit < 1:
        raise ValueError("limit must be positive.")

    selected: list[RetrievalResult] = []
    selected_ids: set[str] = set()

    for ticker in target_tickers:
        if len(selected) >= limit:
            break

        candidates = per_ticker.get(
            ticker,
            [],
        )

        if not candidates:
            continue

        item = candidates[0]

        if item.chunk_id in selected_ids:
            continue

        selected.append(item)
        selected_ids.add(item.chunk_id)

    remaining = [
        item
        for ticker in target_tickers
        for item in per_ticker.get(
            ticker,
            [],
        )
        if item.chunk_id not in selected_ids
    ]

    remaining.sort(
        key=lambda item: (
            -item.score,
            item.rank,
            item.ticker,
            item.chunk_id,
        )
    )

    for item in remaining:
        if len(selected) >= limit:
            break

        if item.chunk_id in selected_ids:
            continue

        selected.append(item)
        selected_ids.add(item.chunk_id)

    return [
        item.model_copy(
            update={
                "rank": rank,
            }
        )
        for rank, item in enumerate(
            selected,
            start=1,
        )
    ]


class VectorEvidenceTool:
    """Retrieve filing narrative evidence from Qdrant."""

    def __init__(
        self,
        store: QdrantChunkStore,
        *,
        limit: int = 5,
        candidate_multiplier: int = 4,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be positive.")

        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive.")

        self.store = store
        self.limit = limit
        self.candidate_multiplier = candidate_multiplier

    def _longitudinal_years(
        self,
        *,
        question: str,
        tickers: tuple[str, ...],
    ) -> tuple[int, ...]:
        """Choose representative indexed years for an unbounded trend query."""

        if not _has_unbounded_multi_year_intent(question):
            return ()

        if _extract_year_range(question) or _extract_explicit_years(question):
            return ()

        if len(tickers) != 1:
            return ()

        provider = getattr(
            self.store,
            "available_fiscal_years",
            None,
        )

        if not callable(provider):
            return ()

        available = tuple(provider(tickers[0]))

        return _select_representative_years(
            years=available,
            limit=self.limit,
        )

    def _multi_company_results(
        self,
        *,
        question: str,
        tickers: tuple[str, ...],
        fiscal_years: tuple[int, ...],
        section_names: tuple[str, ...],
    ) -> list[RetrievalResult]:
        """Retrieve and fuse independently per company before bounded balancing."""

        candidate_limit = max(
            self.limit * self.candidate_multiplier,
            20,
        )

        per_ticker: dict[
            str,
            list[RetrievalResult],
        ] = {}

        for ticker in tickers:
            filters = RetrievalFilter(
                tickers=(ticker,),
                fiscal_years=fiscal_years,
                section_names=section_names,
            )

            sparse = self.store.sparse_search(
                question,
                filters=filters,
                limit=candidate_limit,
            )

            dense = self.store.dense_search(
                question,
                filters=filters,
                limit=candidate_limit,
            )

            per_ticker[ticker] = reciprocal_rank_fusion(
                sparse,
                dense,
                limit=self.limit,
            )

        return _diversify_results_by_ticker(
            per_ticker=per_ticker,
            target_tickers=tickers,
            limit=self.limit,
        )

    def __call__(
        self,
        question: str,
    ) -> list[Evidence]:
        if _has_explicit_unsupported_company(question):
            return []

        tickers = _extract_tickers(question)
        section_names = _extract_section_names(question)

        fiscal_years = _resolve_fiscal_years(
            question=question,
            tickers=tickers,
            store=self.store,
        )

        longitudinal_years = self._longitudinal_years(
            question=question,
            tickers=tickers,
        )

        effective_years = longitudinal_years or fiscal_years

        if len(tickers) > 1:
            results = self._multi_company_results(
                question=question,
                tickers=tickers,
                fiscal_years=effective_years,
                section_names=section_names,
            )

        else:
            filters = (
                RetrievalFilter(
                    tickers=tickers,
                    fiscal_years=effective_years,
                    section_names=section_names,
                )
                if (tickers or effective_years or section_names)
                else None
            )

            if longitudinal_years:
                candidate_limit = max(
                    self.limit * self.candidate_multiplier * 5,
                    100,
                )

            else:
                candidate_limit = max(
                    self.limit * self.candidate_multiplier,
                    20,
                )

            sparse = self.store.sparse_search(
                question,
                filters=filters,
                limit=candidate_limit,
            )

            dense = self.store.dense_search(
                question,
                filters=filters,
                limit=candidate_limit,
            )

            fusion_limit = candidate_limit if longitudinal_years else self.limit

            fused = reciprocal_rank_fusion(
                sparse,
                dense,
                limit=fusion_limit,
            )

            if longitudinal_years:
                results = _diversify_results_by_year(
                    results=fused,
                    target_years=longitudinal_years,
                    limit=self.limit,
                )

            else:
                results = fused

        return [
            Evidence(
                evidence_id=stable_id(
                    "evidence",
                    "vector",
                    result.chunk_id,
                ),
                source_type="vector",
                ticker=result.ticker,
                filing_id=result.filing_id,
                section_name=result.section_name,
                text=result.text,
                source_url=result.source_url,
                score=round(
                    result.score,
                    3,
                ),
                rank=result.rank,
                provenance={
                    "chunk_id": result.chunk_id,
                    "section_id": result.section_id,
                    "fiscal_year": result.fiscal_year,
                    "retrieval_method": result.method,
                    "temporal_mode": ("longitudinal" if longitudinal_years else "filtered"),
                },
            )
            for result in results
        ]


class GraphEvidenceTool:
    """Retrieve structured evidence through safe Graph QA."""

    def __init__(
        self,
        service: GraphQAService,
    ) -> None:
        self.service = service

    def __call__(
        self,
        question: str,
    ) -> list[Evidence]:
        if _has_explicit_unsupported_company(
            question
        ) or _has_explicit_unsupported_financial_metric(question):
            return []

        result = self.service.ask(question)

        rows = [_json_safe(row) for row in result.rows]

        if not rows:
            return []

        tickers = _extract_tickers(question)

        ticker = tickers[0] if len(tickers) == 1 else None

        filings = _graph_filing_provenance(
            service=self.service,
            tickers=tickers,
            rows=rows,
        )

        filing_id = None
        source_url = None

        if filings:
            raw_filing_id = filings[0].get("accession_number")

            raw_source_url = filings[0].get("source_url")

            filing_id = str(raw_filing_id) if raw_filing_id is not None else None

            source_url = str(raw_source_url) if raw_source_url is not None else None

        return [
            Evidence(
                evidence_id=stable_id(
                    "evidence",
                    "graph",
                    result.question,
                    result.cypher,
                ),
                source_type="graph",
                ticker=ticker,
                filing_id=filing_id,
                structured_payload={
                    "rows": rows,
                },
                source_url=source_url,
                provenance={
                    "question": result.question,
                    "cypher": result.cypher,
                    "parameters": _json_safe(result.parameters),
                    "attempts": result.attempts,
                    "filings": filings,
                },
            )
        ]
