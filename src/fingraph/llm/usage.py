"""LLM token-usage and cost accounting."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

_MILLION = Decimal("1000000")

# Pricing snapshot used for reproducible benchmark cost estimates.
# Groq GPT-OSS 120B pricing as of 2026-10-10:
#   input: $0.15 / 1M tokens
#   cached input: $0.075 / 1M tokens
#   output: $0.60 / 1M tokens
_MODEL_PRICING_PER_MILLION: dict[str, tuple[Decimal, Decimal, Decimal]] = {
    "openai/gpt-oss-120b": (
        Decimal("0.15"),
        Decimal("0.075"),
        Decimal("0.60"),
    ),
}


@dataclass(frozen=True)
class LLMUsage:
    """One provider completion's token usage and estimated USD cost."""

    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost_usd: float | None
    cached_input_cost_usd: float | None
    output_cost_usd: float | None
    total_cost_usd: float | None

    @property
    def uncached_input_tokens(self) -> int:
        """Return input tokens that were not served from prompt cache."""

        return max(self.input_tokens - self.cached_input_tokens, 0)

    def langfuse_usage_details(self) -> dict[str, int]:
        """Return mutually exclusive Langfuse usage buckets."""

        return {
            "input": self.uncached_input_tokens,
            "cache_read_input_tokens": self.cached_input_tokens,
            "output": self.output_tokens,
            "total": self.total_tokens,
        }

    def langfuse_cost_details(self) -> dict[str, float] | None:
        """Return Langfuse cost buckets when model pricing is known."""

        if self.total_cost_usd is None:
            return None

        return {
            "input": self.input_cost_usd or 0.0,
            "cache_read_input_tokens": self.cached_input_cost_usd or 0.0,
            "output": self.output_cost_usd or 0.0,
            "total": self.total_cost_usd,
        }


def _cost(
    *,
    tokens: int,
    usd_per_million: Decimal,
) -> float:
    """Calculate USD cost from token count and per-million pricing."""

    value = (Decimal(tokens) * usd_per_million) / _MILLION

    return float(value)


def usage_from_counts(
    *,
    model: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    total_tokens: int | None = None,
) -> LLMUsage:
    """Build normalized usage and deterministic cost from provider counts."""

    normalized_input = max(int(input_tokens), 0)
    normalized_cached = min(
        max(int(cached_input_tokens), 0),
        normalized_input,
    )
    normalized_output = max(int(output_tokens), 0)
    normalized_total = (
        max(int(total_tokens), 0)
        if total_tokens is not None
        else normalized_input + normalized_output
    )

    pricing = _MODEL_PRICING_PER_MILLION.get(model)

    if pricing is None:
        return LLMUsage(
            model=model,
            input_tokens=normalized_input,
            cached_input_tokens=normalized_cached,
            output_tokens=normalized_output,
            total_tokens=normalized_total,
            input_cost_usd=None,
            cached_input_cost_usd=None,
            output_cost_usd=None,
            total_cost_usd=None,
        )

    input_price, cached_input_price, output_price = pricing
    uncached_input = normalized_input - normalized_cached

    input_cost = _cost(
        tokens=uncached_input,
        usd_per_million=input_price,
    )
    cached_input_cost = _cost(
        tokens=normalized_cached,
        usd_per_million=cached_input_price,
    )
    output_cost = _cost(
        tokens=normalized_output,
        usd_per_million=output_price,
    )
    total_cost = input_cost + cached_input_cost + output_cost

    return LLMUsage(
        model=model,
        input_tokens=normalized_input,
        cached_input_tokens=normalized_cached,
        output_tokens=normalized_output,
        total_tokens=normalized_total,
        input_cost_usd=input_cost,
        cached_input_cost_usd=cached_input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=total_cost,
    )


def extract_groq_usage(
    *,
    completion: Any,
    model: str,
) -> LLMUsage | None:
    """Extract Groq/OpenAI-style usage from a chat completion."""

    usage = getattr(completion, "usage", None)

    if usage is None:
        return None

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    if prompt_tokens is None or completion_tokens is None:
        return None

    cached_tokens = 0
    details = getattr(usage, "prompt_tokens_details", None)

    if details is not None:
        raw_cached = getattr(details, "cached_tokens", None)

        if raw_cached is not None:
            cached_tokens = int(raw_cached)

    return usage_from_counts(
        model=model,
        input_tokens=int(prompt_tokens),
        cached_input_tokens=cached_tokens,
        output_tokens=int(completion_tokens),
        total_tokens=(int(total_tokens) if total_tokens is not None else None),
    )


def sum_usage(
    usages: list[LLMUsage],
    *,
    model: str,
) -> LLMUsage:
    """Aggregate usage records for one model."""

    input_tokens = sum(item.input_tokens for item in usages)
    cached_input_tokens = sum(item.cached_input_tokens for item in usages)
    output_tokens = sum(item.output_tokens for item in usages)
    total_tokens = sum(item.total_tokens for item in usages)

    return usage_from_counts(
        model=model,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
