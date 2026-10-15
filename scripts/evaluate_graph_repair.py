"""Benchmark baseline Text-to-Cypher against bounded self-healing."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from neo4j.exceptions import Neo4jError

from fingraph.config import get_settings
from fingraph.graph.qa.models import (
    CypherParameter,
    CypherPlan,
)
from fingraph.graph.qa.service import (
    GraphQAError,
    GraphQAService,
)
from fingraph.graph.qa.validator import (
    CypherValidationError,
    validate_cypher,
)
from fingraph.llm.base import CypherGenerator
from fingraph.llm.groq_client import (
    GroqCypherGenerator,
    LLMDeferredError,
    LLMPermanentError,
)
from fingraph.llm.state import LLMStateStore
from fingraph.llm.usage import LLMUsage, sum_usage
from fingraph.storage.neo4j import Neo4jStore

DEFAULT_DATASET = Path("evals/datasets/graph_repair_cases.json")

DEFAULT_REPORT = Path("evals/reports/graph_repair_report.json")

DEFAULT_STATE_DB = Path("data/state/graph_repair_benchmark.db")


FaultInjector = Callable[
    [CypherPlan],
    CypherPlan,
]


def _round3(
    value: float,
) -> float:
    """Round displayed benchmark values to three decimals."""

    return round(
        value,
        3,
    )


def _atomic_json_write(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Persist benchmark output atomically."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(f"{path.suffix}.tmp")

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def _replace_parameters(
    plan: CypherPlan,
    *,
    transform: Callable[
        [CypherParameter],
        CypherParameter,
    ],
) -> CypherPlan:
    """Return a plan with deterministically transformed parameters."""

    return plan.model_copy(
        update={"parameters": [transform(parameter) for parameter in plan.parameters]}
    )


def _inject_invalid_relationship(
    plan: CypherPlan,
) -> CypherPlan:
    """Replace the first relationship type with an invalid one."""

    corrupted, replacements = re.subn(
        r"\[:[A-Za-z_][A-Za-z0-9_]*",
        "[:BROKEN_RELATIONSHIP",
        plan.cypher,
        count=1,
    )

    if replacements != 1:
        raise ValueError("Could not inject invalid relationship.")

    return plan.model_copy(
        update={
            "cypher": corrupted,
        }
    )


def _inject_invalid_property(
    plan: CypherPlan,
) -> CypherPlan:
    """Replace one known property with a nonexistent property."""

    property_patterns = (
        r"\.period_year\b",
        r"\.concept\b",
        r"\.value\b",
        r"\.filing_date\b",
    )

    for pattern in property_patterns:
        corrupted, replacements = re.subn(
            pattern,
            ".definitely_not_a_property",
            plan.cypher,
            count=1,
        )

        if replacements == 1:
            return plan.model_copy(
                update={
                    "cypher": corrupted,
                }
            )

    raise ValueError("Could not inject invalid property.")


def _inject_syntax_corruption(
    plan: CypherPlan,
) -> CypherPlan:
    """Append deterministic invalid Cypher syntax."""

    return plan.model_copy(update={"cypher": (f"{plan.cypher.rstrip()}\nTHIS IS NOT VALID CYPHER")})


def _inject_wrong_entity_alias(
    plan: CypherPlan,
) -> CypherPlan:
    """Replace the ticker parameter with a nonexistent entity."""

    changed = False

    def transform(
        parameter: CypherParameter,
    ) -> CypherParameter:
        nonlocal changed

        if not changed and parameter.name.lower() == "ticker" and parameter.kind == "scalar":
            changed = True

            return parameter.model_copy(update={"scalar": ("INVALID_TICKER")})

        return parameter

    corrupted = _replace_parameters(
        plan,
        transform=transform,
    )

    if not changed:
        raise ValueError("Could not find ticker parameter for wrong-entity fault.")

    return corrupted


def _inject_unsafe_query(
    plan: CypherPlan,
) -> CypherPlan:
    """Append a forbidden mutation clause."""

    return plan.model_copy(update={"cypher": (f"{plan.cypher.rstrip()}\nDELETE c")})


def _inject_empty_result(
    plan: CypherPlan,
) -> CypherPlan:
    """Replace year filters with an impossible historical period."""

    changed = False

    def transform(
        parameter: CypherParameter,
    ) -> CypherParameter:
        nonlocal changed

        if "year" not in parameter.name.lower():
            return parameter

        changed = True

        if parameter.kind == "scalar":
            return parameter.model_copy(
                update={
                    "scalar": "1900",
                }
            )

        return parameter.model_copy(
            update={
                "values": ["1900" for _ in parameter.values] or ["1900"],
            }
        )

    corrupted = _replace_parameters(
        plan,
        transform=transform,
    )

    if not changed:
        raise ValueError("Could not find a year parameter for empty-result fault.")

    return corrupted


FAULT_INJECTORS: dict[
    str,
    FaultInjector,
] = {
    "invalid_relationship": (_inject_invalid_relationship),
    "invalid_property": (_inject_invalid_property),
    "syntax_corruption": (_inject_syntax_corruption),
    "wrong_entity_alias": (_inject_wrong_entity_alias),
    "unsafe_query": (_inject_unsafe_query),
    "empty_result": (_inject_empty_result),
}


class InjectedFirstPlanGenerator:
    """
    Return one deterministic corrupted plan, then delegate repairs.

    The first call simulates a faulty initial Text-to-Cypher generation.
    Subsequent calls use the real Groq generator and the existing repair
    contract with previous_cypher + error.
    """

    def __init__(
        self,
        *,
        first_plan: CypherPlan,
        repair_generator: CypherGenerator,
    ) -> None:
        self._first_plan = first_plan
        self._repair_generator = repair_generator
        self._first_call = True
        self.repair_usages: list[LLMUsage] = []

    def generate_cypher(
        self,
        *,
        question: str,
        schema: str,
        previous_cypher: str | None = None,
        error: str | None = None,
    ) -> CypherPlan:
        """Return injected first plan then use real repair generation."""

        if self._first_call:
            self._first_call = False

            return self._first_plan

        plan = self._repair_generator.generate_cypher(
            question=question,
            schema=schema,
            previous_cypher=(previous_cypher),
            error=error,
        )

        usage = getattr(
            self._repair_generator,
            "last_usage",
            None,
        )

        if isinstance(usage, LLMUsage):
            self.repair_usages.append(usage)

        return plan


def _execute_baseline(
    *,
    store: Neo4jStore,
    plan: CypherPlan,
    max_rows: int,
) -> dict[str, Any]:
    """Execute exactly one corrupted attempt without repair."""

    started = time.perf_counter()

    try:
        validated = validate_cypher(
            plan,
            max_rows=max_rows,
        )

        store.explain_read(
            validated.cypher,
            parameters=(validated.parameters),
        )

        rows = store.execute_read(
            validated.cypher,
            parameters=(validated.parameters),
        )

        success = bool(rows)

        failure_type = None if success else "empty_result"

        error = None

        if not success:
            error = "Query executed successfully but returned zero rows."

    except CypherValidationError as exc:
        success = False
        rows = []
        failure_type = "validation_error"
        error = str(exc)

    except Neo4jError as exc:
        success = False
        rows = []
        failure_type = "neo4j_error"
        error = str(exc)

    except Exception as exc:
        success = False
        rows = []
        failure_type = "unexpected_error"
        error = f"{type(exc).__name__}: {exc}"

    latency_ms = (time.perf_counter() - started) * 1000

    return {
        "success": success,
        "row_count": len(rows),
        "failure_type": (failure_type),
        "error": error,
        "latency_ms": (_round3(latency_ms)),
    }


def _execute_self_healing(
    *,
    store: Neo4jStore,
    repair_generator: CypherGenerator,
    corrupted_plan: CypherPlan,
    question: str,
    max_attempts: int,
    max_rows: int,
) -> dict[str, Any]:
    """Execute the same corrupted first plan through GraphQA repair."""

    generator = InjectedFirstPlanGenerator(
        first_plan=corrupted_plan,
        repair_generator=(repair_generator),
    )

    service = GraphQAService(
        store=store,
        generator=generator,
        max_attempts=max_attempts,
        max_rows=max_rows,
    )

    started = time.perf_counter()

    try:
        result = service.ask(question)

        success = bool(result.cypher.strip() and result.rows)

        failure_type = None if success else "empty_or_unsupported"

        error = None

        if not success:
            error = "Self-healing completed without non-empty graph rows."

        attempts = result.attempts

        row_count = len(result.rows)

        final_cypher = result.cypher

    except GraphQAError as exc:
        success = False
        failure_type = "graph_qa_error"
        error = str(exc)
        attempts = max_attempts
        row_count = 0
        final_cypher = None

    except LLMDeferredError as exc:
        success = False
        failure_type = "deferred"
        error = str(exc)
        attempts = 0
        row_count = 0
        final_cypher = None

    except LLMPermanentError as exc:
        success = False
        failure_type = "llm_permanent_error"
        error = str(exc)
        attempts = 0
        row_count = 0
        final_cypher = None

    except Exception as exc:
        success = False
        failure_type = "unexpected_error"
        error = f"{type(exc).__name__}: {exc}"
        attempts = 0
        row_count = 0
        final_cypher = None

    latency_ms = (time.perf_counter() - started) * 1000

    usage = sum_usage(
        generator.repair_usages,
        model=getattr(repair_generator, "model", "unknown"),
    )

    return {
        "success": success,
        "row_count": row_count,
        "failure_type": (failure_type),
        "error": error,
        "attempts": attempts,
        "repair_attempts": max(
            attempts - 1,
            0,
        ),
        "final_cypher": (final_cypher),
        "latency_ms": (_round3(latency_ms)),
        "repair_usage": {
            "input_tokens": usage.input_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "cost_usd": (
                _round3(usage.total_cost_usd) if usage.total_cost_usd is not None else None
            ),
            "cost_usd_precise": usage.total_cost_usd,
        },
    }


def _load_cases(
    path: Path,
) -> list[dict[str, str]]:
    """Load and validate benchmark case definitions."""

    raw = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(
        raw,
        list,
    ):
        raise ValueError("Graph repair dataset must contain a JSON list.")

    cases: list[dict[str, str]] = []

    seen_ids: set[str] = set()

    for index, item in enumerate(raw):
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(f"Case {index} must be an object.")

        case_id = item.get("id")
        question = item.get("question")
        fault_type = item.get("fault_type")

        if (
            not isinstance(
                case_id,
                str,
            )
            or not case_id
        ):
            raise ValueError(f"Case {index} has invalid id.")

        if case_id in seen_ids:
            raise ValueError(f"Duplicate case id: {case_id}")

        if (
            not isinstance(
                question,
                str,
            )
            or not question
        ):
            raise ValueError(f"Case {case_id} has invalid question.")

        if (
            not isinstance(
                fault_type,
                str,
            )
            or fault_type not in FAULT_INJECTORS
        ):
            raise ValueError(f"Case {case_id} has unsupported fault_type: {fault_type}")

        seen_ids.add(case_id)

        cases.append(
            {
                "id": case_id,
                "question": question,
                "fault_type": (fault_type),
            }
        )

    return cases


def _mean(
    values: list[float],
) -> float | None:
    """Return rounded mean or None for an empty list."""

    if not values:
        return None

    return _round3(sum(values) / len(values))


def _build_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate self-healing benchmark metrics."""

    total = len(results)

    baseline_successes = sum(bool(item["baseline"]["success"]) for item in results)

    final_successes = sum(bool(item["self_healing"]["success"]) for item in results)

    baseline_failures = total - baseline_successes

    recoveries = sum(
        (not bool(item["baseline"]["success"]) and bool(item["self_healing"]["success"]))
        for item in results
    )

    repair_attempts = [float(item["self_healing"]["repair_attempts"]) for item in results]

    recovered_attempts = [
        float(item["self_healing"]["repair_attempts"])
        for item in results
        if (not item["baseline"]["success"] and item["self_healing"]["success"])
    ]

    added_latencies = [float(item["added_latency_ms"]) for item in results]

    return {
        "total_cases": total,
        "faulted_first_pass_success_rate": (_round3(baseline_successes / total) if total else None),
        "baseline_failures": (baseline_failures),
        "recovered_cases": (recoveries),
        "recovery_rate": (_round3(recoveries / baseline_failures) if baseline_failures else None),
        "final_success_rate": (_round3(final_successes / total) if total else None),
        "average_repair_attempts": (_mean(repair_attempts)),
        "average_repair_attempts_recovered": (_mean(recovered_attempts)),
        "mean_added_latency_ms": (_mean(added_latencies)),
        "token_cost": {
            "status": "instrumented",
            "added_input_tokens": sum(
                int(item["self_healing"]["repair_usage"]["input_tokens"]) for item in results
            ),
            "added_cached_input_tokens": sum(
                int(item["self_healing"]["repair_usage"]["cached_input_tokens"]) for item in results
            ),
            "added_output_tokens": sum(
                int(item["self_healing"]["repair_usage"]["output_tokens"]) for item in results
            ),
            "added_total_tokens": sum(
                int(item["self_healing"]["repair_usage"]["total_tokens"]) for item in results
            ),
            "added_cost_usd": _round3(
                sum(
                    float(item["self_healing"]["repair_usage"]["cost_usd_precise"] or 0.0)
                    for item in results
                )
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=("Benchmark controlled Text-to-Cypher repair."))

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
    )

    parser.add_argument(
        "--state-db",
        type=Path,
        default=DEFAULT_STATE_DB,
    )

    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=65.0,
    )

    return parser.parse_args()


def main() -> None:
    """Run controlled baseline-versus-self-healing benchmark."""

    args = _parse_args()

    if args.cooldown_seconds < 0:
        raise ValueError("cooldown-seconds cannot be negative.")

    settings = get_settings()

    if not settings.groq_api_key:
        raise RuntimeError("FINGRAPH_GROQ_API_KEY is required.")

    if not settings.neo4j_password:
        raise RuntimeError("FINGRAPH_NEO4J_PASSWORD is required.")

    cases = _load_cases(args.dataset)

    args.state_db.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_store = LLMStateStore(str(args.state_db))

    generator = GroqCypherGenerator(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        state_store=state_store,
        reasoning_effort=(settings.llm_reasoning_effort),
        infrastructure_max_attempts=(settings.llm_infrastructure_max_attempts),
        max_retry_wait_seconds=(settings.llm_max_retry_wait_seconds),
    )

    store = Neo4jStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=(settings.neo4j_password),
        database=(settings.neo4j_database),
        read_timeout_seconds=(settings.neo4j_read_timeout_seconds),
        connection_timeout_seconds=(settings.neo4j_connection_timeout_seconds),
        max_transaction_retry_time_seconds=(settings.neo4j_max_transaction_retry_time_seconds),
    )

    results: list[dict[str, Any]] = []

    print("Graph self-healing benchmark")
    print(f"  cases: {len(cases)}")
    print("  baseline: one injected faulted attempt")
    print("  self_healing: same fault + bounded repair")
    print(f"  cooldown_seconds: {args.cooldown_seconds:.3f}")
    print()

    try:
        store.health_check()

        for index, case in enumerate(cases):
            case_id = case["id"]
            question = case["question"]
            fault_type = case["fault_type"]

            print(f"{case_id} [{fault_type}]")

            generation_started = time.perf_counter()

            clean_plan = generator.generate_cypher(
                question=question,
                schema=(
                    __import__(
                        "fingraph.graph.qa.schema",
                        fromlist=["GRAPH_SCHEMA"],
                    ).GRAPH_SCHEMA
                ),
            )

            initial_generation_ms = (time.perf_counter() - generation_started) * 1000

            if not clean_plan.cypher.strip():
                raise RuntimeError(f"{case_id}: clean generation returned unsupported result.")

            injector = FAULT_INJECTORS[fault_type]

            corrupted_plan = injector(clean_plan)

            baseline = _execute_baseline(
                store=store,
                plan=(corrupted_plan),
                max_rows=(settings.graph_qa_max_rows),
            )

            self_healing = _execute_self_healing(
                store=store,
                repair_generator=(generator),
                corrupted_plan=(corrupted_plan),
                question=question,
                max_attempts=(settings.graph_qa_max_attempts),
                max_rows=(settings.graph_qa_max_rows),
            )

            baseline_total_ms = initial_generation_ms + float(baseline["latency_ms"])

            self_healing_total_ms = initial_generation_ms + float(self_healing["latency_ms"])

            added_latency_ms = self_healing_total_ms - baseline_total_ms

            result = {
                "id": case_id,
                "question": question,
                "fault_type": (fault_type),
                "initial_generation_ms": (_round3(initial_generation_ms)),
                "baseline": {
                    **baseline,
                    "total_latency_ms": (_round3(baseline_total_ms)),
                },
                "self_healing": {
                    **self_healing,
                    "total_latency_ms": (_round3(self_healing_total_ms)),
                },
                "recovered": (not baseline["success"] and self_healing["success"]),
                "added_latency_ms": (_round3(added_latency_ms)),
            }

            results.append(result)

            print(
                "  baseline="
                f"{'PASS' if baseline['success'] else 'FAIL'}"
                " | self_healing="
                f"{'PASS' if self_healing['success'] else 'FAIL'}"
                " | repairs="
                f"{self_healing['repair_attempts']}"
                " | added_latency_ms="
                f"{result['added_latency_ms']:.3f}"
            )

            partial_report = {
                "benchmark": ("graph_self_healing"),
                "status": ("partial" if len(results) < len(cases) else "complete"),
                "summary": (_build_summary(results)),
                "cases": results,
            }

            _atomic_json_write(
                args.report,
                partial_report,
            )

            if index < len(cases) - 1 and args.cooldown_seconds > 0:
                print(f"  cooldown -> {args.cooldown_seconds:.3f}s")

                time.sleep(args.cooldown_seconds)

        summary = _build_summary(results)

        report = {
            "benchmark": ("graph_self_healing"),
            "status": "complete",
            "methodology": {
                "baseline": ("One deterministic fault-injected Cypher attempt with no repair."),
                "self_healing": (
                    "The same injected first "
                    "plan executed through "
                    "GraphQAService with its "
                    "real bounded repair path."
                ),
                "max_attempts": (settings.graph_qa_max_attempts),
                "max_rows": (settings.graph_qa_max_rows),
                "state_db": str(args.state_db),
                "token_cost_note": (
                    "Repair token counts come from Groq completion usage. "
                    "Cost uses the pinned GPT-OSS 120B pricing snapshot in "
                    "fingraph.llm.usage, including cached-input pricing."
                ),
            },
            "summary": summary,
            "cases": results,
        }

        _atomic_json_write(
            args.report,
            report,
        )

        print()
        print("Summary")
        print(
            f"  faulted_first_pass_success_rate: {summary['faulted_first_pass_success_rate']:.3f}"
        )
        print(f"  recovery_rate: {summary['recovery_rate']:.3f}")
        print(f"  final_success_rate: {summary['final_success_rate']:.3f}")
        print(f"  average_repair_attempts: {summary['average_repair_attempts']:.3f}")
        print(f"  mean_added_latency_ms: {summary['mean_added_latency_ms']:.3f}")
        token_cost = summary["token_cost"]
        print(
            "  added_tokens: "
            f"input={token_cost['added_input_tokens']} "
            f"cached_input={token_cost['added_cached_input_tokens']} "
            f"output={token_cost['added_output_tokens']} "
            f"total={token_cost['added_total_tokens']}"
        )
        print(f"  added_cost_usd: {float(token_cost['added_cost_usd']):.3f}")
        print()
        print(f"Report: {args.report}")

    finally:
        store.close()


if __name__ == "__main__":
    main()
