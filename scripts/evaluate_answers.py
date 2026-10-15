"""End-to-end resumable answer evaluation."""

from __future__ import annotations

import argparse
import hashlib
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
from fingraph.answer import (
    GroqAnswerSynthesizer,
)
from fingraph.config import (
    get_settings,
)
from fingraph.evaluation.answers import (
    ANSWER_EVALUATOR_VERSION,
    AnswerGoldenCase,
    GroqAnswerJudge,
    case_fingerprint,
    dataset_fingerprint,
    evaluate_answer_contract,
    evaluation_fingerprint,
    load_report,
    reusable_result,
    write_report_atomic,
)
from fingraph.evaluation.quota import (
    classify_groq_deferred_error,
)
from fingraph.graph.qa import (
    GraphQAService,
)
from fingraph.llm import (
    GroqCypherGenerator,
    LLMStateStore,
)
from fingraph.llm.groq_client import (
    LLMDeferredError,
    LLMQualityError,
)
from fingraph.storage.neo4j import (
    Neo4jStore,
)
from fingraph.storage.qdrant import (
    QdrantChunkStore,
)

DATASET_PATH = Path("evals/datasets/answer_golden.json")

REPORT_PATH = Path("evals/reports/answer_golden_report.json")


def _non_negative_float(
    value: str,
) -> float:
    """Parse a non-negative CLI float."""

    try:
        parsed = float(value)

    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number.") from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0.")

    return parsed


def _parse_args() -> argparse.Namespace:
    """Parse evaluator CLI arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=("Ignore prior evaluation results and rerun every case."),
    )

    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help=("Reuse passing completed cases but rerun quality failures."),
    )

    parser.add_argument(
        "--cooldown-seconds",
        type=_non_negative_float,
        default=0.0,
        metavar="N",
        help=(
            "Wait N seconds after each "
            "newly attempted case before "
            "continuing. Skipped cases do "
            "not trigger a cooldown. "
            "Default: 0."
        ),
    )

    return parser.parse_args()


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


def _sha256_file(
    path: Path,
) -> str:
    """Hash one file by content."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _tree_fingerprint(
    root: Path,
    *,
    suffixes: frozenset[str] | None = None,
) -> str:
    """Hash a directory tree deterministically by relative path and content."""

    digest = hashlib.sha256()

    if not root.exists():
        digest.update(f"missing:{root.as_posix()}".encode())
        return digest.hexdigest()

    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and (suffixes is None or path.suffix.lower() in suffixes)
    ]

    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()

        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\0")

    return digest.hexdigest()


def _evaluation_configuration(
    settings: Any,
) -> dict[str, Any]:
    """Build the non-secret identity of the evaluated system."""

    return {
        "evaluator_version": ANSWER_EVALUATOR_VERSION,
        "llm": {
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "reasoning_effort": settings.llm_reasoning_effort,
            "infrastructure_max_attempts": (settings.llm_infrastructure_max_attempts),
            "max_retry_wait_seconds": (settings.llm_max_retry_wait_seconds),
            "synthesis_input_token_budget": (settings.llm_synthesis_input_token_budget),
        },
        "retrieval": {
            "qdrant_url": settings.qdrant_url,
            "qdrant_collection": settings.qdrant_collection,
            "evidence_limit": settings.api_evidence_limit,
        },
        "graph": {
            "neo4j_uri": settings.neo4j_uri,
            "neo4j_database": settings.neo4j_database,
            "read_timeout_seconds": settings.neo4j_read_timeout_seconds,
            "connection_timeout_seconds": settings.neo4j_connection_timeout_seconds,
            "max_transaction_retry_time_seconds": (
                settings.neo4j_max_transaction_retry_time_seconds
            ),
            "max_attempts": settings.graph_qa_max_attempts,
            "max_rows": settings.graph_qa_max_rows,
        },
        "serving_code_hash": _tree_fingerprint(
            Path("src/fingraph"),
            suffixes=frozenset({".py"}),
        ),
        "processed_corpus_hash": _tree_fingerprint(
            Path("data/processed"),
        ),
    }


def _build_summary(
    *,
    cases: list[AnswerGoldenCase],
    results: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    """Build current answer-evaluation summary."""

    ordered = [results[case.id] for case in cases if case.id in results]

    completed = [item for item in ordered if item.get("status") == "completed"]

    passed = [
        item
        for item in completed
        if bool(
            item.get(
                "passed",
                False,
            )
        )
    ]

    deferred = sum(item.get("status") == "deferred" for item in ordered)

    llm_quality_failed = sum(
        item.get("status") == "failed"
        and "quality_failure"
        in item.get(
            "failure_categories",
            [],
        )
        for item in ordered
    )

    execution_failed = sum(
        item.get("status") == "failed"
        and "quality_failure"
        not in item.get(
            "failure_categories",
            [],
        )
        for item in ordered
    )

    failure_counts: dict[
        str,
        int,
    ] = {}

    for item in completed:
        for category in item.get(
            "failure_categories",
            [],
        ):
            name = str(category)

            failure_counts[name] = (
                failure_counts.get(
                    name,
                    0,
                )
                + 1
            )

    return {
        "total": len(cases),
        "completed": len(completed),
        "passed": len(passed),
        "quality_failed": (len(completed) - len(passed)),
        "llm_quality_failed": llm_quality_failed,
        "deferred": deferred,
        "execution_failed": execution_failed,
        "accuracy": round(
            (len(passed) / len(completed) if completed else 0.0),
            3,
        ),
        "failure_counts": failure_counts,
    }


def _write_current_report(
    *,
    cases: list[AnswerGoldenCase],
    results: dict[
        str,
        dict[str, Any],
    ],
    evaluation_hash: str,
    evaluation_configuration: dict[str, Any],
) -> None:
    """Persist current evaluation progress."""

    ordered = [results[case.id] for case in cases if case.id in results]

    report = {
        "evaluator_version": (ANSWER_EVALUATOR_VERSION),
        "dataset_hash": dataset_fingerprint(cases),
        "evaluation_hash": evaluation_hash,
        "evaluation_configuration": (evaluation_configuration),
        "summary": _build_summary(
            cases=cases,
            results=results,
        ),
        "results": ordered,
    }

    write_report_atomic(
        path=REPORT_PATH,
        report=report,
    )


def _load_previous_results(
    *,
    cases: list[AnswerGoldenCase],
    fresh: bool,
    evaluation_hash: str,
) -> dict[
    str,
    dict[str, Any],
]:
    """Load reusable prior case results."""

    if fresh:
        return {}

    report = load_report(REPORT_PATH)

    if report is None:
        return {}

    if report.get("evaluation_hash") != evaluation_hash:
        return {}

    raw_results = report.get(
        "results",
        [],
    )

    if not isinstance(
        raw_results,
        list,
    ):
        return {}

    known_ids = {case.id for case in cases}

    return {
        str(result["id"]): result
        for result in raw_results
        if (
            isinstance(
                result,
                dict,
            )
            and result.get("id") in known_ids
        )
    }


def _should_skip_case(
    *,
    previous: dict[str, Any] | None,
    case: AnswerGoldenCase,
    fresh: bool,
    rerun_failed: bool,
    evaluation_hash: str,
) -> bool:
    """Return whether a prior result is safe to reuse for this case."""

    if fresh or previous is None:
        return False

    return reusable_result(
        previous=previous,
        case=case,
        evaluation_hash=evaluation_hash,
        rerun_failed=rerun_failed,
    )


def _serialize_evidence(
    evidence: list[Evidence],
) -> list[dict[str, Any]]:
    """Serialize canonical evidence for the report."""

    return [item.model_dump(mode="json") for item in evidence]


def _apply_cooldown(
    seconds: float,
) -> None:
    """Pause between attempted evaluation cases."""

    if seconds <= 0:
        return

    display_seconds = round(
        seconds,
        3,
    )

    print(f"  cooldown -> {display_seconds:.3f}s")

    time.sleep(seconds)


def _build_qdrant_store(settings: Any) -> QdrantChunkStore:
    """Build the evaluator's Qdrant store with serving-equivalent authentication."""

    return QdrantChunkStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
    )


def _build_neo4j_store(settings: Any) -> Neo4jStore:
    """Build the evaluator's Neo4j store with serving-equivalent timeouts."""

    return Neo4jStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        read_timeout_seconds=settings.neo4j_read_timeout_seconds,
        connection_timeout_seconds=settings.neo4j_connection_timeout_seconds,
        max_transaction_retry_time_seconds=(settings.neo4j_max_transaction_retry_time_seconds),
    )


def main() -> None:
    """Run resumable answer-level evaluation."""

    args = _parse_args()

    raw_cases = json.loads(
        DATASET_PATH.read_text(
            encoding="utf-8",
        )
    )

    cases = [AnswerGoldenCase.model_validate(raw) for raw in raw_cases]

    settings = get_settings()

    evaluation_configuration = _evaluation_configuration(settings)

    evaluation_hash = evaluation_fingerprint(evaluation_configuration)

    if not settings.groq_api_key:
        raise SystemExit("FINGRAPH_GROQ_API_KEY is required.")

    if not settings.neo4j_password:
        raise SystemExit("FINGRAPH_NEO4J_PASSWORD is required.")

    results = _load_previous_results(
        cases=cases,
        fresh=args.fresh,
        evaluation_hash=(evaluation_hash),
    )

    qdrant_store = _build_qdrant_store(settings)

    qdrant_store.health_check()

    vector_tool = VectorEvidenceTool(
        qdrant_store,
        limit=(settings.api_evidence_limit),
    )

    state_store = LLMStateStore(settings.llm_state_db_path)

    generator = GroqCypherGenerator(
        api_key=(settings.groq_api_key),
        model=settings.llm_model,
        state_store=state_store,
        reasoning_effort=(settings.llm_reasoning_effort),
        infrastructure_max_attempts=(settings.llm_infrastructure_max_attempts),
        max_retry_wait_seconds=(settings.llm_max_retry_wait_seconds),
    )

    synthesizer = GroqAnswerSynthesizer(
        api_key=(settings.groq_api_key),
        model=(settings.llm_model),
        state_store=state_store,
        reasoning_effort=(settings.llm_reasoning_effort),
        infrastructure_max_attempts=(settings.llm_infrastructure_max_attempts),
        max_retry_wait_seconds=(settings.llm_max_retry_wait_seconds),
    )

    judge = GroqAnswerJudge(
        api_key=(settings.groq_api_key),
        model=settings.llm_model,
        state_store=state_store,
        reasoning_effort=(settings.llm_reasoning_effort),
        infrastructure_max_attempts=(settings.llm_infrastructure_max_attempts),
        max_retry_wait_seconds=(settings.llm_max_retry_wait_seconds),
    )

    print("Answer evaluation")

    print(
        "  total_cases:",
        len(cases),
    )

    print(
        "  rerun_failed:",
        args.rerun_failed,
    )

    print(
        "  cooldown_seconds:",
        (f"{args.cooldown_seconds:.3f}"),
    )

    print(
        "  evaluation_hash:",
        evaluation_hash,
    )

    print()

    with _build_neo4j_store(settings) as neo4j_store:
        graph_service = GraphQAService(
            store=neo4j_store,
            generator=generator,
            max_attempts=(settings.graph_qa_max_attempts),
            max_rows=(settings.graph_qa_max_rows),
        )

        graph_tool = GraphEvidenceTool(graph_service)

        agent = build_agent_graph(
            vector_tool=vector_tool,
            graph_tool=graph_tool,
        )

        for case in cases:
            previous = results.get(case.id)

            if _should_skip_case(
                previous=previous,
                case=case,
                fresh=args.fresh,
                rerun_failed=(args.rerun_failed),
                evaluation_hash=(evaluation_hash),
            ):
                state = "PASS" if (previous and previous.get("passed")) else "FAIL"

                print(f"{case.id} -> SKIP ({state})")

                continue

            started = time.perf_counter()

            stop_after_case = False

            stop_for_daily_quota = False

            quota_retry_after_seconds: float | None = None

            try:
                agent_result = agent.invoke(_initial_state(case.question))

                route = str(agent_result["route"])

                evidence = list(
                    agent_result.get(
                        "evidence",
                        [],
                    )
                )

                (
                    plan,
                    calculations,
                    citations,
                ) = synthesizer.synthesize(
                    question=(case.question),
                    evidence=evidence,
                )

                contract = evaluate_answer_contract(
                    case=case,
                    route=route,
                    plan=plan,
                    citations=(citations),
                    calculations=(calculations),
                    evidence=evidence,
                )

                if case.semantic_judge:
                    judgment = judge.evaluate(
                        case=case,
                        answer=(plan.answer),
                        insufficient_evidence=(plan.insufficient_evidence),
                        citations=(citations),
                        calculations=(calculations),
                        evidence=(evidence),
                    )

                else:
                    from fingraph.evaluation.answers import (
                        AnswerJudgeResult,
                    )

                    judgment = AnswerJudgeResult(
                        correct=True,
                        grounded=True,
                        complete=True,
                        temporally_correct=True,
                        unsupported_claims=[],
                        reason=("Semantic judge disabled for case."),
                    )

                semantic_passed = all(
                    (
                        judgment.correct,
                        judgment.grounded,
                        judgment.complete,
                        judgment.temporally_correct,
                        not (judgment.unsupported_claims),
                    )
                )

                failure_categories = list(contract.failure_categories)

                if not judgment.correct:
                    failure_categories.append("answer_correctness_failure")

                if not judgment.grounded:
                    failure_categories.append("answer_grounding_failure")

                if not judgment.complete:
                    failure_categories.append("answer_completeness_failure")

                if not (judgment.temporally_correct):
                    failure_categories.append("answer_temporal_failure")

                if judgment.unsupported_claims:
                    failure_categories.append("unsupported_claim")

                passed = contract.passed and semantic_passed

                latency_ms = round(
                    (time.perf_counter() - started) * 1000,
                    3,
                )

                results[case.id] = {
                    "id": case.id,
                    "case_hash": (case_fingerprint(case)),
                    "evaluator_version": (ANSWER_EVALUATOR_VERSION),
                    "evaluation_hash": (evaluation_hash),
                    "question": (case.question),
                    "status": ("completed"),
                    "passed": passed,
                    "route": route,
                    "answer": (plan.answer),
                    "insufficient_evidence": (plan.insufficient_evidence),
                    "contract": (contract.model_dump(mode="json")),
                    "judgment": (judgment.model_dump(mode="json")),
                    "failure_categories": (failure_categories),
                    "calculations": [item.model_dump(mode="json") for item in calculations],
                    "citations": [item.model_dump(mode="json") for item in citations],
                    "evidence": (_serialize_evidence(evidence)),
                    "latency_ms": (latency_ms),
                    "error": None,
                }

                outcome = "PASS" if passed else "FAIL"

                print(f"{case.id} -> {outcome}")

                if not passed:
                    print(
                        "  failures:",
                        failure_categories,
                    )

                    print(
                        "  judge:",
                        judgment.reason,
                    )

            except LLMDeferredError as exc:
                stop_after_case = True

                quota_signal = classify_groq_deferred_error(exc)

                if quota_signal is not None and quota_signal.scope == "daily_tokens":
                    stop_for_daily_quota = True

                    quota_retry_after_seconds = quota_signal.retry_after_seconds

                latency_ms = round(
                    (time.perf_counter() - started) * 1000,
                    3,
                )

                results[case.id] = {
                    "id": case.id,
                    "case_hash": (case_fingerprint(case)),
                    "evaluator_version": (ANSWER_EVALUATOR_VERSION),
                    "evaluation_hash": (evaluation_hash),
                    "question": (case.question),
                    "status": "deferred",
                    "passed": False,
                    "failure_categories": ["provider_deferred"],
                    "latency_ms": (latency_ms),
                    "error": (f"{type(exc).__name__}: {exc}"),
                }

                print(f"{case.id} -> DEFERRED")

                if stop_for_daily_quota:
                    print(
                        "  provider quota:",
                        ("Groq daily token limit detected."),
                    )

                    if quota_retry_after_seconds is not None:
                        print(
                            ("  retry_after_seconds:"),
                            (f"{quota_retry_after_seconds:.3f}"),
                        )

            except LLMQualityError as exc:
                latency_ms = round(
                    (time.perf_counter() - started) * 1000,
                    3,
                )

                results[case.id] = {
                    "id": case.id,
                    "case_hash": (case_fingerprint(case)),
                    "evaluator_version": (ANSWER_EVALUATOR_VERSION),
                    "evaluation_hash": (evaluation_hash),
                    "question": (case.question),
                    "status": "failed",
                    "passed": False,
                    "failure_categories": ["quality_failure"],
                    "latency_ms": (latency_ms),
                    "error": (f"{type(exc).__name__}: {exc}"),
                }

                print(f"{case.id} -> QUALITY FAILURE")

                print(
                    "  error:",
                    results[case.id]["error"],
                )

            except Exception as exc:
                stop_after_case = True

                latency_ms = round(
                    (time.perf_counter() - started) * 1000,
                    3,
                )

                results[case.id] = {
                    "id": case.id,
                    "case_hash": (case_fingerprint(case)),
                    "evaluator_version": (ANSWER_EVALUATOR_VERSION),
                    "evaluation_hash": (evaluation_hash),
                    "question": (case.question),
                    "status": "failed",
                    "passed": False,
                    "failure_categories": ["execution_failure"],
                    "latency_ms": (latency_ms),
                    "error": (f"{type(exc).__name__}: {exc}"),
                }

                print(f"{case.id} -> ERROR")

                print(
                    "  error:",
                    results[case.id]["error"],
                )

            _write_current_report(
                cases=cases,
                results=results,
                evaluation_hash=(evaluation_hash),
                evaluation_configuration=(evaluation_configuration),
            )

            if stop_after_case:
                if stop_for_daily_quota:
                    print(
                        "  stopping "
                        "evaluation early -> "
                        "provider quota/error "
                        "saved; rerun later "
                        "to resume."
                    )

                else:
                    print(
                        "  stopping "
                        "evaluation early -> "
                        "execution/provider "
                        "error saved; rerun "
                        "with --rerun-failed "
                        "to resume."
                    )

                break

            _apply_cooldown(args.cooldown_seconds)

    _write_current_report(
        cases=cases,
        results=results,
        evaluation_hash=(evaluation_hash),
        evaluation_configuration=(evaluation_configuration),
    )

    summary = _build_summary(
        cases=cases,
        results=results,
    )

    print()

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
