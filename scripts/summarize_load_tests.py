"""Aggregate formal FinGraph Locust benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROUTES = (
    "vector",
    "graph",
    "hybrid",
)


def _round3(
    value: float,
) -> float:
    """Round benchmark metrics to three decimals."""

    return round(
        float(value),
        3,
    )


def _read_rows(
    path: Path,
) -> list[dict[str, str]]:
    """Read one Locust stats CSV."""

    if not path.exists():
        raise FileNotFoundError(f"Missing Locust stats file: {path}")

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def _find_row(
    rows: list[dict[str, str]],
    *,
    request_type: str,
    name: str,
) -> dict[str, str]:
    """Find one exact Locust metric row."""

    for row in rows:
        if row.get("Type") == request_type and row.get("Name") == name:
            return row

    raise ValueError(f"Could not find Locust metric type={request_type!r}, name={name!r}.")


def _number(
    row: dict[str, str],
    key: str,
) -> float:
    """Read one numeric Locust field."""

    raw = (
        row.get(
            key,
            "",
        )
        or "0"
    )

    return float(raw)


def _count(
    row: dict[str, str],
    key: str,
) -> int:
    """Read one integer Locust field."""

    return int(
        float(
            row.get(
                key,
                "0",
            )
            or "0"
        )
    )


def _metric_summary(
    row: dict[str, str],
) -> dict[str, Any]:
    """Convert one Locust row into the final report schema."""

    requests = _count(
        row,
        "Request Count",
    )

    failures = _count(
        row,
        "Failure Count",
    )

    error_rate = failures / requests if requests else 0.0

    return {
        "requests": requests,
        "failures": failures,
        "error_rate": (_round3(error_rate)),
        "requests_per_second": (
            _round3(
                _number(
                    row,
                    "Requests/s",
                )
            )
        ),
        "latency_ms_mean": (
            _round3(
                _number(
                    row,
                    "Average Response Time",
                )
            )
        ),
        "latency_ms_p50": (
            _round3(
                _number(
                    row,
                    "50%",
                )
            )
        ),
        "latency_ms_p95": (
            _round3(
                _number(
                    row,
                    "95%",
                )
            )
        ),
        "latency_ms_p99": (
            _round3(
                _number(
                    row,
                    "99%",
                )
            )
        ),
    }


def _route_report(
    *,
    input_dir: Path,
    route: str,
) -> dict[str, Any]:
    """Load sync, SSE completion, and TTFT metrics for one route."""

    route_label = route.upper()

    sync_rows = _read_rows(input_dir / f"{route}_sync_stats.csv")

    stream_rows = _read_rows(input_dir / f"{route}_stream_stats.csv")

    sync_row = _find_row(
        sync_rows,
        request_type="POST",
        name=(f"{route_label} /ask"),
    )

    stream_open_row = _find_row(
        stream_rows,
        request_type="POST",
        name=(f"{route_label} /ask/stream-open"),
    )

    stream_completed_row = _find_row(
        stream_rows,
        request_type="STREAM",
        name=(f"{route_label} completed"),
    )

    ttft_row = _find_row(
        stream_rows,
        request_type="TTFT",
        name=(f"{route_label} first-answer"),
    )

    return {
        "sync": (_metric_summary(sync_row)),
        "stream_completed": (_metric_summary(stream_completed_row)),
        "ttft": (_metric_summary(ttft_row)),
        "diagnostics": {
            "stream_open": (_metric_summary(stream_open_row)),
        },
    }


def main() -> None:
    """Aggregate route-specific Locust outputs."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("reports/load"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/reports/api_load_report.json"),
    )

    args = parser.parse_args()

    routes = {
        route: _route_report(
            input_dir=(args.input_dir),
            route=route,
        )
        for route in ROUTES
    }

    report = {
        "benchmark": ("fingraph_api_locust"),
        "benchmark_mode": ("warm_resumable_production_path"),
        "routes": routes,
        "metric_semantics": {
            "sync": (
                "End-to-end client latency for "
                "POST /ask through receipt of the "
                "validated JSON response."
            ),
            "stream_completed": (
                "End-to-end client latency from "
                "POST /ask/stream start through "
                "receipt of the validated SSE "
                "completed event."
            ),
            "ttft": (
                "Elapsed client time from "
                "POST /ask/stream start until "
                "validated answer text first "
                "becomes available to the client."
            ),
            "stream_open": (
                "HTTP streaming-response opening "
                "latency only. Diagnostic metric; "
                "not end-to-end stream latency."
            ),
        },
        "ttft_semantics": {
            "definition": (
                "Elapsed client time from "
                "request start until validated "
                "answer text first becomes "
                "available in the SSE "
                "completed event."
            ),
            "buffered_structured_output": True,
            "expected_relationship": (
                "TTFT and stream-completed "
                "latency are expected to be "
                "nearly equal because FinGraph "
                "exposes answer text only after "
                "strict structured synthesis "
                "and validation complete."
            ),
        },
        "cache_semantics": {
            "warm_resumable": True,
            "note": (
                "Formal concurrent load results "
                "represent the production "
                "resumable/cache-enabled path. "
                "They must not be presented as "
                "cold LLM-provider latency."
            ),
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = args.output.with_suffix(args.output.suffix + ".tmp")

    temporary.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(args.output)

    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )

    print()

    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
