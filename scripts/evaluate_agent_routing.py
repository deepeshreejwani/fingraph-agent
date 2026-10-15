"""Evaluate deterministic FinGraph routing."""

import json
import time
from pathlib import Path

from fingraph.agent.router import (
    route_question,
)

DATASET_PATH = Path("evals/datasets/agent_routing_bootstrap.json")

REPORT_PATH = Path("evals/reports/agent_routing_bootstrap_report.json")


def main() -> None:
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    results: list[dict[str, object]] = []

    for case in cases:
        started = time.perf_counter()

        decision = route_question(case["question"])

        latency_ms = (time.perf_counter() - started) * 1000

        passed = decision.route == case["expected_route"]

        results.append(
            {
                "id": case["id"],
                "question": (case["question"]),
                "expected_route": (case["expected_route"]),
                "actual_route": (decision.route),
                "reason": (decision.reason),
                "passed": passed,
                "latency_ms": round(
                    latency_ms,
                    3,
                ),
            }
        )

        print(f"{case['id']} -> {decision.route} ({'PASS' if passed else 'FAIL'})")

    passed_count = sum(bool(result["passed"]) for result in results)

    summary = {
        "total": len(results),
        "passed": (passed_count),
        "failed": (len(results) - passed_count),
        "accuracy": round(
            (passed_count / len(results) if results else 0.0),
            3,
        ),
    }

    report = {
        "summary": summary,
        "results": results,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
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
