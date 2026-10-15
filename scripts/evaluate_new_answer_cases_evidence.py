"""Targeted evidence-only validation for answer_golden cases 81-160.

This intentionally skips answer synthesis and the semantic LLM judge. It validates
routing, evidence availability, temporal coverage, provenance, multi-company
coverage, and dependent graph->vector winner scoping before the expensive
160-case answer evaluation.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from fingraph.agent import (
    Evidence,
    GraphEvidenceTool,
    VectorEvidenceTool,
    build_agent_graph,
)
from fingraph.config import get_settings
from fingraph.evaluation.answers import AnswerGoldenCase
from fingraph.graph.qa import GraphQAService
from fingraph.llm import GroqCypherGenerator, LLMStateStore
from fingraph.storage.neo4j import Neo4jStore
from fingraph.storage.qdrant import QdrantChunkStore

DATASET_PATH = Path("evals/datasets/answer_golden.json")
REPORT_PATH = Path("evals/reports/answer_new_cases_evidence_report.json")

START_INDEX = 80
EXPECTED_NEW_CASES = 80

COMPANY_ALIASES: tuple[tuple[str, str], ...] = (
    ("apple", "AAPL"),
    ("aapl", "AAPL"),
    ("amazon", "AMZN"),
    ("amzn", "AMZN"),
    ("alphabet", "GOOGL"),
    ("google", "GOOGL"),
    ("googl", "GOOGL"),
    ("meta", "META"),
    ("facebook", "META"),
    ("microsoft", "MSFT"),
    ("msft", "MSFT"),
    ("nvidia", "NVDA"),
    ("nvda", "NVDA"),
)

YEAR_KEYS = (
    "year",
    "fiscal_year",
    "economic_year",
    "period_year",
)

NON_METRIC_KEYS = {
    "rank",
    "attempts",
    *YEAR_KEYS,
}

SECTION_EXPECTATIONS: tuple[tuple[str, str], ...] = (
    ("market risk", "Item 7A. Market Risk"),
    ("risk", "Item 1A. Risk Factors"),
    ("business", "Item 1. Business"),
)

DEPENDENT_MARKERS = (
    "that company",
    "that firm",
    "the higher-revenue company",
    "the lower-revenue company",
    "the selected company",
    "the winning company",
)


def _initial_state(question: str) -> dict[str, Any]:
    return {
        "question": question,
        "route": None,
        "route_reason": None,
        "vector_evidence": [],
        "graph_evidence": [],
        "evidence": [],
        "errors": [],
    }


def _extract_tickers(question: str) -> tuple[str, ...]:
    normalized = question.lower()
    found: list[str] = []

    for alias, ticker in COMPANY_ALIASES:
        if (
            re.search(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                normalized,
            )
            and ticker not in found
        ):
            found.append(ticker)

    return tuple(found)


def _expected_section(question: str) -> str | None:
    normalized = question.lower()

    if any(
        term in normalized
        for term in (
            "management",
            "drove",
            "drivers",
            "performance",
        )
    ):
        return "Item 7. MD&A"

    for marker, section in SECTION_EXPECTATIONS:
        if marker in normalized:
            return section

    return None


def _graph_rows(evidence: list[Evidence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for item in evidence:
        if item.source_type != "graph":
            continue

        payload = item.structured_payload or {}
        raw_rows = payload.get("rows", [])

        if isinstance(raw_rows, list):
            rows.extend(row for row in raw_rows if isinstance(row, dict))

    return rows


def _represented_years(
    *,
    graph_rows: list[dict[str, Any]],
    vector_evidence: list[Evidence],
) -> set[int]:
    years: set[int] = set()

    for item in vector_evidence:
        raw = item.provenance.get("fiscal_year")
        if raw is None:
            continue

        with contextlib.suppress(
            TypeError,
            ValueError,
        ):
            years.add(int(str(raw)))

    for row in graph_rows:
        for key in YEAR_KEYS:
            raw = row.get(key)
            if raw is None:
                continue

            with contextlib.suppress(
                TypeError,
                ValueError,
            ):
                years.add(int(str(raw)))

    return years


def _graph_tickers(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["ticker"]) for row in rows if row.get("ticker") is not None}


def _vector_tickers(evidence: list[Evidence]) -> set[str]:
    return {str(item.ticker) for item in evidence if item.ticker is not None}


def _graph_provenance_ok(
    evidence: list[Evidence],
    *,
    require_rows: bool,
) -> bool:
    graph_items = [item for item in evidence if item.source_type == "graph"]

    if require_rows and not graph_items:
        return False

    for item in graph_items:
        payload = item.structured_payload or {}
        rows = payload.get("rows", [])

        if require_rows and (not isinstance(rows, list) or not rows):
            return False

        if "cypher" not in item.provenance:
            return False

        if "parameters" not in item.provenance:
            return False

        if rows:
            filings = item.provenance.get("filings", [])
            if not isinstance(filings, list) or not filings:
                return False

            for filing in filings:
                if not isinstance(filing, dict):
                    return False

                if not filing.get("accession_number"):
                    return False

                if not filing.get("source_url"):
                    return False

    return True


def _vector_provenance_ok(evidence: list[Evidence]) -> bool:
    for item in evidence:
        if item.source_type != "vector":
            continue

        if not item.text:
            return False

        if not item.filing_id:
            return False

        if not item.source_url:
            return False

        if not item.provenance.get("chunk_id"):
            return False

    return True


def _metric_value(row: dict[str, Any]) -> Decimal | None:
    values: list[Decimal] = []

    for key, raw in row.items():
        if key.lower() in NON_METRIC_KEYS:
            continue

        if isinstance(raw, bool) or not isinstance(
            raw,
            (
                int,
                float,
                Decimal,
            ),
        ):
            continue

        try:
            values.append(Decimal(str(raw)))
        except InvalidOperation:
            continue

    return values[0] if len(values) == 1 else None


def _dependent_expected_tickers(
    *,
    question: str,
    rows: list[dict[str, Any]],
) -> set[str] | None:
    normalized = question.lower()

    if not any(marker in normalized for marker in DEPENDENT_MARKERS):
        return None

    comparable: dict[str, Decimal] = {}

    for row in rows:
        ticker = row.get("ticker")
        value = _metric_value(row)

        if ticker is None or value is None:
            continue

        comparable[str(ticker)] = value

    if not comparable:
        return set()

    if any(
        term in normalized
        for term in (
            "lower",
            "lowest",
            "smaller",
            "less than",
        )
    ):
        target = min(comparable.values())
    elif any(
        term in normalized
        for term in (
            "higher",
            "highest",
            "larger",
            "greater",
            "more than",
        )
    ):
        target = max(comparable.values())
    else:
        return set(comparable)

    return {ticker for ticker, value in comparable.items() if value == target}


def _usable_evidence_present(
    *,
    graph_rows: list[dict[str, Any]],
    vector_evidence: list[Evidence],
) -> bool:
    return bool(graph_rows or vector_evidence)


def _write_report(results: list[dict[str, Any]]) -> None:
    failure_counts = Counter(
        failure for result in results for failure in result["failure_categories"]
    )

    passed = sum(bool(result["passed"]) for result in results)

    report = {
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "accuracy": (
                round(
                    passed / len(results),
                    3,
                )
                if results
                else 0.0
            ),
            "failure_counts": dict(failure_counts),
        },
        "results": results,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = REPORT_PATH.with_suffix(".tmp")

    temp.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    temp.replace(REPORT_PATH)


def evaluate_case(
    *,
    agent: Any,
    case: AnswerGoldenCase,
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        result = agent.invoke(_initial_state(case.question))

        latency_ms = (time.perf_counter() - started) * 1000

        graph_evidence = list(
            result.get(
                "graph_evidence",
                [],
            )
        )

        vector_evidence = list(
            result.get(
                "vector_evidence",
                [],
            )
        )

        graph_rows = _graph_rows(graph_evidence)

        failures: list[str] = []

        actual_route = result.get("route")

        if actual_route != case.expected_route:
            failures.append("route_mismatch")

        usable = _usable_evidence_present(
            graph_rows=graph_rows,
            vector_evidence=vector_evidence,
        )

        if case.expected_insufficient:
            if usable:
                failures.append("unexpected_supported_evidence")

        else:
            if case.expected_route == "graph" and not graph_rows:
                failures.append("graph_evidence_missing")

            elif case.expected_route == "vector" and not vector_evidence:
                failures.append("vector_evidence_missing")

            elif case.expected_route == "hybrid":
                if not graph_rows:
                    failures.append("graph_evidence_missing")

                if not vector_evidence:
                    failures.append("vector_evidence_missing")

        if graph_evidence and not _graph_provenance_ok(
            graph_evidence,
            require_rows=not case.expected_insufficient,
        ):
            failures.append("graph_provenance_failure")

        if vector_evidence and not _vector_provenance_ok(vector_evidence):
            failures.append("vector_provenance_failure")

        represented_years = _represented_years(
            graph_rows=graph_rows,
            vector_evidence=vector_evidence,
        )

        if (
            not case.expected_insufficient
            and case.expected_fiscal_years
            and not set(case.expected_fiscal_years).issubset(represented_years)
        ):
            failures.append("temporal_coverage_failure")

        requested_tickers = set(_extract_tickers(case.question))

        graph_tickers = _graph_tickers(graph_rows)

        vector_tickers = _vector_tickers(vector_evidence)

        if not case.expected_insufficient and len(requested_tickers) > 1:
            if case.expected_route in {"graph", "hybrid"} and not requested_tickers.issubset(
                graph_tickers
            ):
                failures.append("graph_company_coverage_failure")

            if case.expected_route == "vector" and not requested_tickers.issubset(vector_tickers):
                failures.append("vector_company_coverage_failure")

            if (
                case.expected_route == "hybrid"
                and _dependent_expected_tickers(
                    question=case.question,
                    rows=graph_rows,
                )
                is None
                and not requested_tickers.issubset(vector_tickers)
            ):
                failures.append("vector_company_coverage_failure")

        dependent_expected = _dependent_expected_tickers(
            question=case.question,
            rows=graph_rows,
        )

        if (
            not case.expected_insufficient
            and dependent_expected is not None
            and vector_tickers != dependent_expected
        ):
            failures.append("dependent_vector_scope_failure")

        expected_section = _expected_section(case.question)

        if (
            not case.expected_insufficient
            and vector_evidence
            and expected_section is not None
            and any(item.section_name != expected_section for item in vector_evidence)
        ):
            failures.append("vector_section_failure")

        errors = list(
            result.get(
                "errors",
                [],
            )
        )

        if errors:
            failures.append("workflow_errors_present")

        return {
            "id": case.id,
            "question": case.question,
            "status": "completed",
            "passed": not failures,
            "expected_route": case.expected_route,
            "actual_route": actual_route,
            "expected_insufficient": case.expected_insufficient,
            "failure_categories": failures,
            "requested_tickers": sorted(requested_tickers),
            "graph_tickers": sorted(graph_tickers),
            "vector_tickers": sorted(vector_tickers),
            "represented_years": sorted(represented_years),
            "expected_years": list(case.expected_fiscal_years),
            "expected_section": expected_section,
            "dependent_expected_tickers": (
                sorted(dependent_expected) if dependent_expected is not None else None
            ),
            "graph_row_count": len(graph_rows),
            "vector_evidence_count": len(vector_evidence),
            "latency_ms": round(
                latency_ms,
                3,
            ),
            "errors": errors,
            "error": None,
        }

    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000

        return {
            "id": case.id,
            "question": case.question,
            "status": "failed",
            "passed": False,
            "expected_route": case.expected_route,
            "actual_route": None,
            "expected_insufficient": case.expected_insufficient,
            "failure_categories": [
                "execution_failure",
            ],
            "requested_tickers": sorted(_extract_tickers(case.question)),
            "graph_tickers": [],
            "vector_tickers": [],
            "represented_years": [],
            "expected_years": list(case.expected_fiscal_years),
            "expected_section": _expected_section(case.question),
            "dependent_expected_tickers": None,
            "graph_row_count": 0,
            "vector_evidence_count": 0,
            "latency_ms": round(
                latency_ms,
                3,
            ),
            "errors": [],
            "error": (f"{type(exc).__name__}: {exc}"),
        }


def main() -> None:
    raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    cases = [AnswerGoldenCase.model_validate(item) for item in raw]

    if len(cases) < START_INDEX + EXPECTED_NEW_CASES:
        raise SystemExit(
            f"Expected at least {START_INDEX + EXPECTED_NEW_CASES} cases, found {len(cases)}."
        )

    new_cases = cases[START_INDEX : START_INDEX + EXPECTED_NEW_CASES]

    print(f"Validating cases {START_INDEX + 1}-{START_INDEX + len(new_cases)}")

    print(f"Count: {len(new_cases)}")
    print("Semantic answer synthesis/judge: SKIPPED")
    print()

    settings = get_settings()

    if not settings.groq_api_key:
        raise SystemExit("FINGRAPH_GROQ_API_KEY is required.")

    if not settings.neo4j_password:
        raise SystemExit("FINGRAPH_NEO4J_PASSWORD is required.")

    qdrant_store = QdrantChunkStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
    )

    qdrant_store.health_check()

    vector_tool = VectorEvidenceTool(
        qdrant_store,
        limit=settings.api_evidence_limit,
    )

    state_store = LLMStateStore(settings.llm_state_db_path)

    generator = GroqCypherGenerator(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        state_store=state_store,
        reasoning_effort=settings.llm_reasoning_effort,
        infrastructure_max_attempts=(settings.llm_infrastructure_max_attempts),
        max_retry_wait_seconds=(settings.llm_max_retry_wait_seconds),
    )

    results: list[dict[str, Any]] = []

    with Neo4jStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        read_timeout_seconds=(settings.neo4j_read_timeout_seconds),
        connection_timeout_seconds=(settings.neo4j_connection_timeout_seconds),
        max_transaction_retry_time_seconds=(settings.neo4j_max_transaction_retry_time_seconds),
    ) as neo4j_store:
        graph_service = GraphQAService(
            store=neo4j_store,
            generator=generator,
            max_attempts=settings.graph_qa_max_attempts,
            max_rows=settings.graph_qa_max_rows,
        )

        graph_tool = GraphEvidenceTool(graph_service)

        agent = build_agent_graph(
            vector_tool=vector_tool,
            graph_tool=graph_tool,
        )

        for index, case in enumerate(
            new_cases,
            start=START_INDEX + 1,
        ):
            result = evaluate_case(
                agent=agent,
                case=case,
            )

            results.append(result)

            _write_report(results)

            outcome = "PASS" if result["passed"] else "FAIL"

            print(f"{index:03d} {case.id} -> {outcome}")

            if not result["passed"]:
                print(
                    "    failures:",
                    result["failure_categories"],
                )

                if result["error"]:
                    print(
                        "    error:",
                        result["error"],
                    )

                if "execution_failure" in result["failure_categories"]:
                    print(
                        "    stopping early -> execution failure encountered; report already saved."
                    )
                    break

    passed = sum(bool(result["passed"]) for result in results)

    failure_counts = Counter(
        failure for result in results for failure in result["failure_categories"]
    )

    summary = {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "accuracy": (
            round(
                passed / len(results),
                3,
            )
            if results
            else 0.0
        ),
        "failure_counts": dict(failure_counts),
        "report": str(REPORT_PATH),
    }

    _write_report(results)

    print()

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
