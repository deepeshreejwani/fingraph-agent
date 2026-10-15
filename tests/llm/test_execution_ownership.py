"""Concurrency regressions for durable LLM execution ownership."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fingraph.agent.evidence import Evidence
from fingraph.answer.models import SynthesisPlan
from fingraph.answer.synthesis import GroqAnswerSynthesizer
from fingraph.graph.qa.models import CypherPlan
from fingraph.llm.groq_client import (
    GroqCypherGenerator,
    LLMDeferredError,
)
from fingraph.llm.state import LLMStateStore


class BlockingCompletionProvider:
    """Return one deterministic completion while holding the first caller."""

    def __init__(
        self,
        content: str,
        *,
        delay_seconds: float = 0.150,
    ) -> None:
        self.content = content
        self.delay_seconds = delay_seconds
        self.calls = 0
        self._lock = threading.Lock()

    def create(
        self,
        **_: Any,
    ) -> SimpleNamespace:
        with self._lock:
            self.calls += 1

        time.sleep(self.delay_seconds)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.content,
                    )
                )
            ],
            usage=None,
        )


def _fake_client(
    provider: BlockingCompletionProvider,
) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=provider,
        )
    )


def _register_test_request(
    store: LLMStateStore,
    request_id: str = "request-1",
) -> None:
    store.register(
        request_id=request_id,
        task_type="test",
        provider="groq",
        model="openai/gpt-oss-120b",
        prompt_version="test-v1",
        input_hash="hash",
        request_payload={"x": 1},
    )


def test_state_store_allows_only_one_active_owner(
    tmp_path: Path,
) -> None:
    store = LLMStateStore(tmp_path / "state.db")

    _register_test_request(store)

    barrier = threading.Barrier(2)

    def claim(
        owner_id: str,
    ) -> bool:
        barrier.wait()

        return store.claim_execution(
            request_id="request-1",
            owner_id=owner_id,
            lease_seconds=30.0,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            claim,
            "owner-a",
        )

        second = executor.submit(
            claim,
            "owner-b",
        )

        results = [
            first.result(),
            second.result(),
        ]

    assert sorted(results) == [
        False,
        True,
    ]

    record = store.get("request-1")

    assert record is not None
    assert record.status == "running"

    assert record.execution_owner in {
        "owner-a",
        "owner-b",
    }


def test_stale_owner_cannot_overwrite_new_owner(
    tmp_path: Path,
) -> None:
    store = LLMStateStore(tmp_path / "state.db")

    _register_test_request(store)

    assert store.claim_execution(
        request_id="request-1",
        owner_id="owner-a",
        lease_seconds=0.010,
    )

    time.sleep(0.030)

    assert store.claim_execution(
        request_id="request-1",
        owner_id="owner-b",
        lease_seconds=30.0,
    )

    assert not (
        store.mark_succeeded(
            request_id="request-1",
            response_json='{"owner":"a"}',
            owner_id="owner-a",
        )
    )

    assert store.mark_succeeded(
        request_id="request-1",
        response_json='{"owner":"b"}',
        owner_id="owner-b",
    )

    record = store.get("request-1")

    assert record is not None

    assert record.status == "succeeded"

    assert record.response_json == '{"owner":"b"}'

    assert record.execution_owner is None

    assert record.lease_expires_at is None


def test_expected_status_reset_cannot_clear_active_claim(
    tmp_path: Path,
) -> None:
    store = LLMStateStore(tmp_path / "state.db")

    _register_test_request(store)

    assert store.mark_failed(
        request_id="request-1",
        error="retryable",
    )

    assert store.mark_pending(
        request_id="request-1",
        error="retryable",
        expected_status="failed",
    )

    assert store.claim_execution(
        request_id="request-1",
        owner_id="owner-b",
        lease_seconds=30.0,
    )

    assert not (
        store.mark_pending(
            request_id="request-1",
            error="stale reset",
            expected_status="failed",
        )
    )

    record = store.get("request-1")

    assert record is not None

    assert record.status == "running"

    assert record.execution_owner == "owner-b"


def test_concurrent_identical_cypher_requests_call_provider_once(
    tmp_path: Path,
) -> None:
    store = LLMStateStore(tmp_path / "state.db")

    plan = CypherPlan(
        cypher="MATCH (c:Company) RETURN c.ticker AS ticker LIMIT 1",
        parameters=[],
    )

    provider = BlockingCompletionProvider(plan.model_dump_json())

    generators = [
        GroqCypherGenerator(
            api_key="test-key",
            model="openai/gpt-oss-120b",
            state_store=store,
            infrastructure_max_attempts=1,
        )
        for _ in range(2)
    ]

    for generator in generators:
        generator.client = _fake_client(provider)  # type: ignore[assignment]

    barrier = threading.Barrier(2)

    def execute(
        generator: GroqCypherGenerator,
    ) -> object:
        barrier.wait()

        try:
            return generator.generate_cypher(
                question="Which company uses ticker MSFT?",
                schema="NODE Company",
            )

        except LLMDeferredError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                execute,
                generator,
            )
            for generator in generators
        ]

        results = [future.result() for future in futures]

    assert provider.calls == 1

    assert (
        sum(
            isinstance(
                result,
                CypherPlan,
            )
            for result in results
        )
        == 1
    )

    assert (
        sum(
            isinstance(
                result,
                LLMDeferredError,
            )
            for result in results
        )
        == 1
    )


def test_concurrent_identical_synthesis_requests_call_provider_once(
    tmp_path: Path,
) -> None:
    store = LLMStateStore(tmp_path / "state.db")

    evidence = [
        Evidence(
            evidence_id="evidence-msft",
            source_type="vector",
            ticker="MSFT",
            filing_id="msft-filing",
            text="Microsoft described infrastructure risks.",
            source_url="https://sec.example/msft",
        )
    ]

    plan = SynthesisPlan(
        answer="Microsoft described infrastructure risks.",
        cited_evidence_ids=["evidence-msft"],
        insufficient_evidence=False,
    )

    provider = BlockingCompletionProvider(plan.model_dump_json())

    synthesizers = [
        GroqAnswerSynthesizer(
            api_key="test-key",
            model="openai/gpt-oss-120b",
            state_store=store,
            infrastructure_max_attempts=1,
        )
        for _ in range(2)
    ]

    for synthesizer in synthesizers:
        synthesizer.client = _fake_client(provider)  # type: ignore[assignment]

    barrier = threading.Barrier(2)

    def execute(
        synthesizer: GroqAnswerSynthesizer,
    ) -> object:
        barrier.wait()

        try:
            return synthesizer.synthesize(
                question="What risks did Microsoft describe?",
                evidence=evidence,
            )

        except LLMDeferredError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                execute,
                synthesizer,
            )
            for synthesizer in synthesizers
        ]

        results = [future.result() for future in futures]

    assert provider.calls == 1

    assert (
        sum(
            isinstance(
                result,
                tuple,
            )
            for result in results
        )
        == 1
    )

    assert (
        sum(
            isinstance(
                result,
                LLMDeferredError,
            )
            for result in results
        )
        == 1
    )


def test_execution_lease_must_be_positive(
    tmp_path: Path,
) -> None:
    store = LLMStateStore(tmp_path / "state.db")

    _register_test_request(store)

    with pytest.raises(
        ValueError,
        match="lease_seconds must be positive",
    ):
        store.claim_execution(
            request_id="request-1",
            owner_id="owner",
            lease_seconds=0.0,
        )
