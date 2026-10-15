"""Concurrent API load and latency checkpoint for FinGraph."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_REPORT = Path("reports/api_load_test.json")


@dataclass(frozen=True)
class RequestResult:
    """One API request result."""

    request_id: str
    status_code: int | None
    latency_ms: float
    success: bool
    deferred: bool
    error: str | None


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    """Compute percentile using linear interpolation."""

    if not values:
        return 0.0

    ordered = sorted(values)

    position = (len(ordered) - 1) * percentile_value

    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)

    weight = position - lower

    return ordered[lower] * (1 - weight) + ordered[upper] * weight


async def run_request(
    *,
    client: httpx.AsyncClient,
    url: str,
    method: str,
    request_id: str,
    json_payload: dict[str, Any] | None,
) -> RequestResult:
    """Execute one request and capture latency/result."""

    start = time.perf_counter()

    try:
        response = await client.request(
            method,
            url,
            headers={
                "X-Request-ID": request_id,
            },
            json=json_payload,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        deferred = False

        if response.status_code == 503:
            try:
                payload = response.json()

                deferred = payload.get("error_code") == "llm_temporarily_unavailable"

            except (ValueError, AttributeError):
                deferred = False

        success = 200 <= response.status_code < 300

        return RequestResult(
            request_id=request_id,
            status_code=response.status_code,
            latency_ms=latency_ms,
            success=success,
            deferred=deferred,
            error=None,
        )

    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000

        return RequestResult(
            request_id=request_id,
            status_code=None,
            latency_ms=latency_ms,
            success=False,
            deferred=False,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_batch(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    endpoint: str,
    method: str,
    total_requests: int,
    concurrency: int,
    json_payload: dict[str, Any] | None,
) -> list[RequestResult]:
    """Execute bounded concurrent requests."""

    semaphore = asyncio.Semaphore(concurrency)

    async def worker(index: int) -> RequestResult:
        request_id = (
            f"load-{endpoint.strip('/').replace('/', '-') or 'root'}-c{concurrency}-{index:03d}"
        )

        async with semaphore:
            return await run_request(
                client=client,
                url=f"{base_url.rstrip('/')}{endpoint}",
                method=method,
                request_id=request_id,
                json_payload=json_payload,
            )

    return await asyncio.gather(*(worker(index) for index in range(total_requests)))


def summarize(
    results: list[RequestResult],
) -> dict[str, Any]:
    """Build deterministic benchmark summary."""

    latencies = [result.latency_ms for result in results]

    success_count = sum(result.success for result in results)

    deferred_count = sum(result.deferred for result in results)

    error_count = sum(1 for result in results if result.error is not None)

    status_counts: dict[str, int] = {}

    for result in results:
        key = str(result.status_code) if result.status_code is not None else "exception"

        status_counts[key] = status_counts.get(key, 0) + 1

    total = len(results)

    return {
        "requests": total,
        "successes": success_count,
        "success_rate": round(
            success_count / total if total else 0.0,
            3,
        ),
        "deferred": deferred_count,
        "exceptions": error_count,
        "latency_ms_mean": round(
            statistics.mean(latencies) if latencies else 0.0,
            3,
        ),
        "latency_ms_p50": round(
            percentile(latencies, 0.50),
            3,
        ),
        "latency_ms_p95": round(
            percentile(latencies, 0.95),
            3,
        ),
        "latency_ms_p99": round(
            percentile(latencies, 0.99),
            3,
        ),
        "status_counts": status_counts,
    }


async def run_benchmark(
    *,
    base_url: str,
    endpoint: str,
    method: str,
    total_requests: int,
    concurrency_levels: list[int],
    timeout_seconds: float,
    json_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run benchmark across requested concurrency levels."""

    timeout = httpx.Timeout(timeout_seconds)

    limits = httpx.Limits(
        max_connections=max(concurrency_levels) * 2,
        max_keepalive_connections=max(concurrency_levels),
    )

    report: dict[str, Any] = {
        "base_url": base_url,
        "endpoint": endpoint,
        "method": method,
        "total_requests_per_level": total_requests,
        "concurrency_levels": concurrency_levels,
        "results": {},
    }

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
    ) as client:
        for concurrency in concurrency_levels:
            batch_start = time.perf_counter()

            results = await run_batch(
                client=client,
                base_url=base_url,
                endpoint=endpoint,
                method=method,
                total_requests=total_requests,
                concurrency=concurrency,
                json_payload=json_payload,
            )

            wall_ms = (time.perf_counter() - batch_start) * 1000

            summary = summarize(results)

            summary["wall_time_ms"] = round(
                wall_ms,
                3,
            )

            summary["throughput_rps"] = round(
                (total_requests / (wall_ms / 1000) if wall_ms > 0 else 0.0),
                3,
            )

            report["results"][str(concurrency)] = summary

    return report


def parse_concurrency(
    raw: str,
) -> list[int]:
    """Parse comma-separated positive concurrency levels."""

    values = [int(part.strip()) for part in raw.split(",") if part.strip()]

    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Concurrency must contain positive integers.")

    return values


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
    )

    parser.add_argument(
        "--endpoint",
        default="/health",
    )

    parser.add_argument(
        "--method",
        choices=("GET", "POST"),
        default="GET",
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--concurrency",
        type=parse_concurrency,
        default=parse_concurrency("1,2,4"),
    )

    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--question",
        default=None,
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
    )

    args = parser.parse_args()

    if args.requests <= 0:
        parser.error("--requests must be positive")

    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    json_payload: dict[str, Any] | None = None

    if args.method == "POST":
        if not args.question:
            parser.error("--question is required for POST benchmarks")

        json_payload = {
            "question": args.question,
        }

    report = asyncio.run(
        run_benchmark(
            base_url=args.base_url,
            endpoint=args.endpoint,
            method=args.method,
            total_requests=args.requests,
            concurrency_levels=args.concurrency,
            timeout_seconds=args.timeout_seconds,
            json_payload=json_payload,
        )
    )

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.report.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"{'CONC':>6}"
        f"{'SUCCESS':>10}"
        f"{'DEFERRED':>10}"
        f"{'MEAN MS':>12}"
        f"{'P50 MS':>12}"
        f"{'P95 MS':>12}"
        f"{'P99 MS':>12}"
        f"{'RPS':>10}"
    )

    print("-" * 84)

    for concurrency in args.concurrency:
        values = report["results"][str(concurrency)]

        print(
            f"{concurrency:>6}"
            f"{values['success_rate']:>10.3f}"
            f"{values['deferred']:>10}"
            f"{values['latency_ms_mean']:>12.3f}"
            f"{values['latency_ms_p50']:>12.3f}"
            f"{values['latency_ms_p95']:>12.3f}"
            f"{values['latency_ms_p99']:>12.3f}"
            f"{values['throughput_rps']:>10.3f}"
        )

    print()
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
