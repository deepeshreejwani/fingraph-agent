"""Focused tests for LLM token/cost accounting and persistence."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from fingraph.llm.state import LLMStateStore
from fingraph.llm.usage import extract_groq_usage, usage_from_counts


def test_gpt_oss_120b_cost_with_cached_input() -> None:
    usage = usage_from_counts(
        model="openai/gpt-oss-120b",
        input_tokens=1000,
        cached_input_tokens=200,
        output_tokens=500,
        total_tokens=1500,
    )

    assert usage.uncached_input_tokens == 800
    assert usage.input_cost_usd == 0.00012
    assert usage.cached_input_cost_usd == 0.000015
    assert usage.output_cost_usd == 0.0003
    assert usage.total_cost_usd == pytest.approx(0.000435)


def test_extract_groq_usage() -> None:
    completion = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
            prompt_tokens_details=SimpleNamespace(cached_tokens=20),
        )
    )

    usage = extract_groq_usage(
        completion=completion,
        model="openai/gpt-oss-120b",
    )

    assert usage is not None
    assert usage.input_tokens == 120
    assert usage.cached_input_tokens == 20
    assert usage.output_tokens == 30
    assert usage.total_tokens == 150


def test_state_store_persists_usage(tmp_path: Path) -> None:
    store = LLMStateStore(tmp_path / "state.db")

    usage = usage_from_counts(
        model="openai/gpt-oss-120b",
        input_tokens=100,
        cached_input_tokens=10,
        output_tokens=20,
        total_tokens=120,
    )

    store.register(
        request_id="request-1",
        task_type="test",
        provider="groq",
        model="openai/gpt-oss-120b",
        prompt_version="test-v1",
        input_hash="hash",
        request_payload={"x": 1},
    )

    store.mark_succeeded(
        request_id="request-1",
        response_json='{"ok":true}',
        usage=usage,
    )

    record = store.get("request-1")

    assert record is not None

    restored = record.usage()

    assert restored is not None
    assert restored.input_tokens == 100
    assert restored.cached_input_tokens == 10
    assert restored.output_tokens == 20
    assert restored.total_tokens == 120
    assert restored.total_cost_usd == usage.total_cost_usd
