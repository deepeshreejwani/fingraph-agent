"""Bootstrap evaluation for safe graph QA."""

import json
import time
from pathlib import Path
from typing import Any, cast

from fingraph.config import get_settings
from fingraph.graph.qa import (
    GraphQAError,
    GraphQAService,
)
from fingraph.llm import (
    GroqCypherGenerator,
    LLMDeferredError,
    LLMPermanentError,
    LLMStateStore,
)
from fingraph.storage.neo4j import (
    Neo4jStore,
)

DATASET_PATH = Path("evals/datasets/graph_qa_bootstrap.json")
REPORT_PATH = Path("evals/reports/graph_qa_bootstrap_report.json")


def load_dataset() -> list[dict[str, Any]]:
    return cast(
        list[dict[str, Any]],
        json.loads(DATASET_PATH.read_text(encoding="utf-8")),
    )


def _numeric_values_by_year(
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Extract year -> financial value independent of LLM column aliases."""

    result: dict[str, float] = {}

    for row in rows:
        year: int | None = None
        value: float | None = None

        for key, raw_value in row.items():
            normalized_key = key.lower()

            if "year" in normalized_key and isinstance(raw_value, int):
                year = raw_value

            if normalized_key not in {
                "year",
                "economic_year",
                "fiscal_year",
                "period_year",
            } and isinstance(
                raw_value,
                (int, float),
            ):
                value = float(raw_value)

        if year is not None and value is not None:
            result[str(year)] = value

    return result


def _ticker_values(
    rows: list[dict[str, Any]],
) -> set[str]:
    """Extract ticker-like values independent of result column alias."""

    tickers: set[str] = set()

    for row in rows:
        for key, raw_value in row.items():
            if "ticker" in key.lower() and isinstance(
                raw_value,
                str,
            ):
                tickers.add(raw_value.upper())

    return tickers


def _evaluate_expected_values(
    *,
    rows: list[dict[str, Any]],
    expected: dict[str, float],
) -> bool:
    actual = _numeric_values_by_year(rows)

    if set(actual) != set(expected):
        return False

    return all(actual[year] == float(expected_value) for year, expected_value in expected.items())


def _evaluate_expected_tickers(
    *,
    rows: list[dict[str, Any]],
    expected: list[str],
) -> bool:
    actual = _ticker_values(rows)
    expected_set = {ticker.upper() for ticker in expected}

    return actual == expected_set


def evaluate_case(
    *,
    service: GraphQAService,
    case: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        result = service.ask(case["question"])

        latency_ms = (time.perf_counter() - started) * 1000.0

        passed_non_empty = bool(result.rows) if case.get("expected_non_empty") else True

        passed_values: bool | None = None
        passed_tickers: bool | None = None

        if "expected_values" in case:
            passed_values = _evaluate_expected_values(
                rows=result.rows,
                expected=case["expected_values"],
            )

        if "expected_tickers" in case:
            passed_tickers = _evaluate_expected_tickers(
                rows=result.rows,
                expected=case["expected_tickers"],
            )

        checks = [passed_non_empty]

        if passed_values is not None:
            checks.append(passed_values)

        if passed_tickers is not None:
            checks.append(passed_tickers)

        passed = all(checks)

        return {
            "id": case["id"],
            "status": "completed",
            "passed": passed,
            "passed_non_empty": passed_non_empty,
            "passed_values": passed_values,
            "passed_tickers": passed_tickers,
            "attempts": result.attempts,
            "row_count": len(result.rows),
            "latency_ms": round(
                latency_ms,
                3,
            ),
            "cypher": result.cypher,
            "parameters": result.parameters,
            "rows": result.rows,
            "error": None,
        }

    except LLMDeferredError as exc:
        return {
            "id": case["id"],
            "status": "deferred",
            "passed_non_empty": False,
            "passed": False,
            "passed_values": None,
            "passed_tickers": None,
            "attempts": None,
            "row_count": 0,
            "latency_ms": None,
            "cypher": None,
            "parameters": None,
            "rows": [],
            "error": str(exc),
        }

    except (
        LLMPermanentError,
        GraphQAError,
    ) as exc:
        return {
            "id": case["id"],
            "status": "failed",
            "passed_non_empty": False,
            "passed": False,
            "passed_values": None,
            "passed_tickers": None,
            "attempts": None,
            "row_count": 0,
            "latency_ms": None,
            "cypher": None,
            "parameters": None,
            "rows": [],
            "error": str(exc),
        }


def main() -> None:
    settings = get_settings()

    if not settings.groq_api_key:
        raise SystemExit("FINGRAPH_GROQ_API_KEY is required.")

    if not settings.neo4j_password:
        raise SystemExit("FINGRAPH_NEO4J_PASSWORD is required.")

    dataset = load_dataset()

    state_store = LLMStateStore(settings.llm_state_db_path)

    generator = GroqCypherGenerator(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        state_store=state_store,
        reasoning_effort=settings.llm_reasoning_effort,
        infrastructure_max_attempts=settings.llm_infrastructure_max_attempts,
        max_retry_wait_seconds=settings.llm_max_retry_wait_seconds,
    )

    results: list[dict[str, Any]] = []

    with Neo4jStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    ) as store:
        service = GraphQAService(
            store=store,
            generator=generator,
            max_attempts=settings.graph_qa_max_attempts,
            max_rows=settings.graph_qa_max_rows,
        )

        for case in dataset:
            result = evaluate_case(
                service=service,
                case=case,
            )

            results.append(result)

            print(
                case["id"],
                "->",
                result["status"],
            )

    completed = [result for result in results if result["status"] == "completed"]

    passed = [result for result in completed if result["passed"]]

    deferred = [result for result in results if result["status"] == "deferred"]

    failed = [result for result in results if result["status"] == "failed"]

    summary = {
        "total": len(results),
        "completed": len(completed),
        "passed": len(passed),
        "deferred": len(deferred),
        "failed": len(failed),
        "completion_rate": round(
            len(completed) / len(results) if results else 0.0,
            3,
        ),
        "correctness_rate": round(
            len(passed) / len(results) if results else 0.0,
            3,
        ),
    }

    payload = {
        "summary": summary,
        "results": results,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
