"""End-to-end evidence evaluation for the FinGraph agent."""

import json
import time
from pathlib import Path
from typing import Any

from fingraph.agent import (
    Evidence,
    GraphEvidenceTool,
    VectorEvidenceTool,
    build_agent_graph,
)
from fingraph.config import (
    get_settings,
)
from fingraph.evaluation import (
    classify_failures,
)
from fingraph.graph.qa import (
    GraphQAService,
)
from fingraph.llm import (
    GroqCypherGenerator,
    LLMStateStore,
)
from fingraph.storage.neo4j import (
    Neo4jStore,
)
from fingraph.storage.qdrant import (
    QdrantChunkStore,
)

DATASET_PATH = Path("evals/datasets/agent_evidence_bootstrap.json")

REPORT_PATH = Path("evals/reports/agent_evidence_bootstrap_report.json")


def _initial_state(
    question: str,
) -> dict[str, Any]:
    """Create clean state for one agent invocation."""

    return {
        "question": question,
        "route": None,
        "route_reason": None,
        "vector_evidence": [],
        "graph_evidence": [],
        "evidence": [],
        "errors": [],
    }


def _graph_rows(
    evidence: list[Evidence],
) -> list[dict[str, Any]]:
    """Collect structured result rows from graph evidence."""

    rows: list[dict[str, Any]] = []

    for item in evidence:
        if item.source_type != "graph":
            continue

        payload = item.structured_payload or {}

        raw_rows = payload.get(
            "rows",
            [],
        )

        if not isinstance(
            raw_rows,
            list,
        ):
            continue

        rows.extend(
            row
            for row in raw_rows
            if isinstance(
                row,
                dict,
            )
        )

    return rows


def _extract_year_values(
    rows: list[dict[str, Any]],
) -> dict[
    str,
    float,
]:
    """Extract economic year -> numeric value independent of aliases."""

    result: dict[
        str,
        float,
    ] = {}

    year_keys = {
        "year",
        "fiscal_year",
        "economic_year",
        "period_year",
    }

    ignored_numeric_keys = {
        "rank",
        "attempts",
    }

    for row in rows:
        year: int | None = None

        value: float | None = None

        for (
            key,
            raw_value,
        ) in row.items():
            normalized = key.lower()

            if (
                normalized in year_keys
                and isinstance(
                    raw_value,
                    int,
                )
                and not isinstance(
                    raw_value,
                    bool,
                )
            ):
                year = raw_value

                continue

            if normalized in year_keys or normalized in ignored_numeric_keys:
                continue

            if isinstance(
                raw_value,
                (
                    int,
                    float,
                ),
            ) and not isinstance(
                raw_value,
                bool,
            ):
                value = float(raw_value)

        if year is not None and value is not None:
            result[str(year)] = value

    return result


def _check_expected_values(
    *,
    evidence: list[Evidence],
    expected: dict[
        str,
        float,
    ],
) -> bool:
    """Check expected graph values by year."""

    actual = _extract_year_values(_graph_rows(evidence))

    for (
        year,
        expected_value,
    ) in expected.items():
        actual_value = actual.get(year)

        if actual_value is None:
            return False

        if actual_value != float(expected_value):
            return False

    return True


def _check_vector_metadata(
    *,
    evidence: list[Evidence],
    ticker: str | None,
    section: str | None,
    fiscal_years: (set[int] | None),
) -> bool:
    """Check vector evidence provenance constraints."""

    vector_items = [item for item in evidence if (item.source_type == "vector")]

    if not vector_items:
        return False

    for item in vector_items:
        if ticker is not None and item.ticker != ticker:
            return False

        if section is not None and item.section_name != section:
            return False

        if fiscal_years is not None:
            raw_year = item.provenance.get("fiscal_year")

            if raw_year is None:
                return False

            try:
                year = int(str(raw_year))

            except (
                TypeError,
                ValueError,
            ):
                return False

            if year not in fiscal_years:
                return False

    return True


def _check_grounding(
    *,
    evidence: list[Evidence],
) -> bool:
    """Verify evidence contains usable provenance."""

    if not evidence:
        return False

    for item in evidence:
        if item.source_type == "vector":
            if not item.text:
                return False

            if not item.source_url:
                return False

            if not item.filing_id:
                return False

            if "chunk_id" not in item.provenance:
                return False

        elif item.source_type == "graph":
            payload = item.structured_payload or {}

            rows = payload.get("rows")

            if not isinstance(
                rows,
                list,
            ):
                return False

            if not rows:
                return False

            if "cypher" not in item.provenance:
                return False

            if "parameters" not in item.provenance:
                return False

        else:
            return False

    return True


def _serialize_evidence(
    evidence: list[Evidence],
) -> list[dict[str, Any]]:
    """Serialize evidence for JSON report output."""

    return [item.model_dump(mode="python") for item in evidence]


def _build_summary(
    results: list[dict[str, Any]],
) -> dict[
    str,
    Any,
]:
    """Build current evaluation summary."""

    passed_count = sum(
        bool(
            result.get(
                "passed",
                False,
            )
        )
        for result in results
    )

    failed_count = len(results) - passed_count

    completed_count = sum((result.get("status") == "completed") for result in results)

    failure_counts: dict[
        str,
        int,
    ] = {}

    for result in results:
        for category in result.get(
            "failure_categories",
            [],
        ):
            category_name = str(category)

            failure_counts[category_name] = (
                failure_counts.get(
                    category_name,
                    0,
                )
                + 1
            )

    return {
        "total": (len(results)),
        "completed": (completed_count),
        "passed": (passed_count),
        "failed": (failed_count),
        "accuracy": round(
            (passed_count / len(results) if results else 0.0),
            3,
        ),
        "failure_counts": (failure_counts),
    }


def _write_report(
    results: list[dict[str, Any]],
) -> None:
    """Persist evaluation progress after every completed case."""

    report = {
        "summary": (_build_summary(results)),
        "results": results,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = REPORT_PATH.with_suffix(".tmp")

    temp_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    temp_path.replace(REPORT_PATH)


def evaluate_case(
    *,
    agent: Any,
    case: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    """Evaluate one end-to-end agent case."""

    started = time.perf_counter()

    try:
        result = agent.invoke(_initial_state(case["question"]))

        latency_ms = (time.perf_counter() - started) * 1000

        evidence = list(result["evidence"])

        graph_evidence = list(result["graph_evidence"])

        vector_evidence = list(result["vector_evidence"])

        route_passed = result["route"] == case["expected_route"]

        graph_non_empty = bool(_graph_rows(graph_evidence))

        vector_non_empty = bool(vector_evidence)

        graph_passed = (
            graph_non_empty
            if case.get(
                "expected_graph_non_empty",
                False,
            )
            else True
        )

        vector_passed = (
            vector_non_empty
            if case.get(
                "expected_vector_non_empty",
                False,
            )
            else True
        )

        values_passed = True

        if "expected_values" in case:
            values_passed = _check_expected_values(
                evidence=evidence,
                expected=(case["expected_values"]),
            )

        metadata_passed = True

        if (
            "expected_ticker" in case
            or "expected_section" in case
            or "expected_fiscal_years" in case
        ):
            expected_years = case.get("expected_fiscal_years")

            fiscal_years = (
                {int(year) for year in expected_years}
                if isinstance(
                    expected_years,
                    list,
                )
                else None
            )

            metadata_passed = _check_vector_metadata(
                evidence=evidence,
                ticker=(case.get("expected_ticker")),
                section=(case.get("expected_section")),
                fiscal_years=(fiscal_years),
            )

        grounding_passed = _check_grounding(evidence=evidence)

        passed = all(
            (
                route_passed,
                graph_passed,
                vector_passed,
                values_passed,
                metadata_passed,
                grounding_passed,
            )
        )

        failures = classify_failures(
            route_passed=(route_passed),
            graph_passed=(graph_passed),
            vector_passed=(vector_passed),
            values_passed=(values_passed),
            metadata_passed=(metadata_passed),
            grounding_passed=(grounding_passed),
        )

        return {
            "id": (case["id"]),
            "question": (case["question"]),
            "status": ("completed"),
            "passed": passed,
            "expected_route": (case["expected_route"]),
            "actual_route": (result["route"]),
            "route_passed": (route_passed),
            "graph_passed": (graph_passed),
            "vector_passed": (vector_passed),
            "values_passed": (values_passed),
            "metadata_passed": (metadata_passed),
            "grounding_passed": (grounding_passed),
            "failure_categories": (failures),
            "latency_ms": round(
                latency_ms,
                3,
            ),
            "graph_evidence_count": (len(graph_evidence)),
            "vector_evidence_count": (len(vector_evidence)),
            "evidence_count": (len(evidence)),
            "evidence": (_serialize_evidence(evidence)),
            "error": None,
        }

    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000

        return {
            "id": (case["id"]),
            "question": (case["question"]),
            "status": "failed",
            "passed": False,
            "expected_route": (case["expected_route"]),
            "actual_route": None,
            "route_passed": False,
            "graph_passed": False,
            "vector_passed": False,
            "values_passed": False,
            "metadata_passed": False,
            "grounding_passed": False,
            "failure_categories": [
                "execution_failure",
            ],
            "latency_ms": round(
                latency_ms,
                3,
            ),
            "graph_evidence_count": 0,
            "vector_evidence_count": 0,
            "evidence_count": 0,
            "evidence": [],
            "error": (f"{type(exc).__name__}: {exc}"),
        }


def main() -> None:
    """Run the end-to-end bootstrap evidence evaluation."""

    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    settings = get_settings()

    if not settings.groq_api_key:
        raise SystemExit("FINGRAPH_GROQ_API_KEY is required.")

    if not settings.neo4j_password:
        raise SystemExit("FINGRAPH_NEO4J_PASSWORD is required.")

    qdrant_store = QdrantChunkStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=(settings.qdrant_collection),
    )

    qdrant_store.health_check()

    vector_tool = VectorEvidenceTool(
        qdrant_store,
        limit=(settings.api_evidence_limit),
    )

    state_store = LLMStateStore(settings.llm_state_db_path)

    generator = GroqCypherGenerator(
        api_key=(settings.groq_api_key),
        model=(settings.llm_model),
        state_store=(state_store),
        reasoning_effort=(settings.llm_reasoning_effort),
        infrastructure_max_attempts=(settings.llm_infrastructure_max_attempts),
        max_retry_wait_seconds=(settings.llm_max_retry_wait_seconds),
    )

    results: list[
        dict[
            str,
            Any,
        ]
    ] = []

    with Neo4jStore(
        uri=(settings.neo4j_uri),
        user=(settings.neo4j_user),
        password=(settings.neo4j_password),
        database=(settings.neo4j_database),
        read_timeout_seconds=(settings.neo4j_read_timeout_seconds),
        connection_timeout_seconds=(settings.neo4j_connection_timeout_seconds),
        max_transaction_retry_time_seconds=(settings.neo4j_max_transaction_retry_time_seconds),
    ) as neo4j_store:
        graph_service = GraphQAService(
            store=(neo4j_store),
            generator=(generator),
            max_attempts=(settings.graph_qa_max_attempts),
            max_rows=(settings.graph_qa_max_rows),
        )

        graph_tool = GraphEvidenceTool(graph_service)

        agent = build_agent_graph(
            vector_tool=(vector_tool),
            graph_tool=(graph_tool),
        )

        for case in cases:
            result = evaluate_case(
                agent=agent,
                case=case,
            )

            results.append(result)

            _write_report(results)

            outcome = "PASS" if result["passed"] else "FAIL"

            print(f"{case['id']} -> {outcome}")

            if not result["passed"]:
                print(
                    "  failures:",
                    result["failure_categories"],
                )

                if result["error"] is not None:
                    print(
                        "  error:",
                        result["error"],
                    )

    summary = _build_summary(results)

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
