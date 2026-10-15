"""Deterministic synthesis-context budgeting."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from fingraph.agent.evidence import Evidence

_TOKEN_ESTIMATE_BYTES_PER_TOKEN = 3
_DEFAULT_RESERVE_TOKENS = 512


class SynthesisContextBudgetError(RuntimeError):
    """Raised when non-truncatable synthesis input already exceeds the budget."""


@dataclass(frozen=True)
class BoundedSynthesisContext:
    """One deterministic bounded synthesis input."""

    evidence_payload: list[dict[str, Any]]
    estimated_input_tokens: int
    input_token_budget: int
    reserve_tokens: int
    truncated: bool
    truncated_evidence_ids: tuple[str, ...]
    original_text_chars: int
    retained_text_chars: int

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe budget metadata for persistence and observability."""

        return {
            "estimated_input_tokens": self.estimated_input_tokens,
            "input_token_budget": self.input_token_budget,
            "reserve_tokens": self.reserve_tokens,
            "truncated": self.truncated,
            "truncated_evidence_ids": list(self.truncated_evidence_ids),
            "original_text_chars": self.original_text_chars,
            "retained_text_chars": self.retained_text_chars,
        }


def _canonical_json(
    value: Any,
) -> str:
    """Serialize deterministically for token estimation."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def estimate_input_tokens(
    text: str,
) -> int:
    """Conservatively estimate tokens from UTF-8 byte length.

    This is intentionally an estimate rather than a provider-tokenizer claim.
    Exact provider usage is captured after successful Groq calls.
    """

    if not text:
        return 0

    return math.ceil(len(text.encode("utf-8")) / _TOKEN_ESTIMATE_BYTES_PER_TOKEN)


def _truncate_text(
    text: str,
    *,
    max_chars: int,
) -> str:
    """Truncate text deterministically while marking visible truncation."""

    if len(text) <= max_chars:
        return text

    if max_chars <= 0:
        return ""

    if max_chars <= 3:
        return text[:max_chars]

    return text[: max_chars - 3] + "..."


def _evidence_payload(
    evidence: list[Evidence],
    *,
    text_char_cap: int | None,
) -> list[dict[str, Any]]:
    """Serialize evidence while preserving IDs, structure, and provenance."""

    payload: list[dict[str, Any]] = []

    for item in evidence:
        text = item.text

        if text is not None and text_char_cap is not None:
            text = _truncate_text(
                text,
                max_chars=text_char_cap,
            )

        payload.append(
            {
                "evidence_id": item.evidence_id,
                "source_type": item.source_type,
                "ticker": item.ticker,
                "filing_id": item.filing_id,
                "section_name": item.section_name,
                "text": text,
                "structured_payload": item.structured_payload,
                "source_url": item.source_url,
                "provenance": item.provenance,
            }
        )

    return payload


def _estimate_complete_input(
    *,
    question: str,
    evidence_payload: list[dict[str, Any]],
    calculations_payload: list[dict[str, Any]],
    response_schema: dict[str, Any],
    system_instructions: str,
) -> int:
    """Estimate the complete provider-side input for grounded synthesis."""

    prompt_payload = {
        "question": question,
        "evidence": evidence_payload,
        "deterministic_calculations": calculations_payload,
    }

    return (
        estimate_input_tokens(system_instructions)
        + estimate_input_tokens(_canonical_json(prompt_payload))
        + estimate_input_tokens(_canonical_json(response_schema))
    )


def build_bounded_synthesis_context(
    *,
    question: str,
    evidence: list[Evidence],
    calculations_payload: list[dict[str, Any]],
    response_schema: dict[str, Any],
    system_instructions: str,
    input_token_budget: int,
    reserve_tokens: int = _DEFAULT_RESERVE_TOKENS,
) -> BoundedSynthesisContext:
    """Bound narrative evidence while preserving structured evidence/provenance."""

    if input_token_budget <= 0:
        raise ValueError("input_token_budget must be positive.")

    if reserve_tokens < 0:
        raise ValueError("reserve_tokens must be non-negative.")

    if reserve_tokens >= input_token_budget:
        raise ValueError("reserve_tokens must be smaller than input_token_budget.")

    effective_budget = input_token_budget - reserve_tokens

    full_payload = _evidence_payload(
        evidence,
        text_char_cap=None,
    )

    full_estimate = _estimate_complete_input(
        question=question,
        evidence_payload=full_payload,
        calculations_payload=calculations_payload,
        response_schema=response_schema,
        system_instructions=system_instructions,
    )

    original_text_chars = sum(len(item.text) for item in evidence if item.text is not None)

    if full_estimate <= effective_budget:
        return BoundedSynthesisContext(
            evidence_payload=full_payload,
            estimated_input_tokens=full_estimate,
            input_token_budget=input_token_budget,
            reserve_tokens=reserve_tokens,
            truncated=False,
            truncated_evidence_ids=(),
            original_text_chars=original_text_chars,
            retained_text_chars=original_text_chars,
        )

    minimal_payload = _evidence_payload(
        evidence,
        text_char_cap=0,
    )

    minimal_estimate = _estimate_complete_input(
        question=question,
        evidence_payload=minimal_payload,
        calculations_payload=calculations_payload,
        response_schema=response_schema,
        system_instructions=system_instructions,
    )

    if minimal_estimate > effective_budget:
        raise SynthesisContextBudgetError(
            "Synthesis input exceeds the configured "
            "context budget even after removing "
            "narrative evidence text. Structured "
            "evidence, provenance, question, "
            "calculation payload, and response schema "
            "are preserved and cannot be safely "
            "truncated."
        )

    max_original_chars = max(
        (len(item.text) for item in evidence if item.text is not None),
        default=0,
    )

    low = 0
    high = max_original_chars

    while low < high:
        candidate = (low + high + 1) // 2

        candidate_payload = _evidence_payload(
            evidence,
            text_char_cap=candidate,
        )

        candidate_estimate = _estimate_complete_input(
            question=question,
            evidence_payload=candidate_payload,
            calculations_payload=(calculations_payload),
            response_schema=response_schema,
            system_instructions=(system_instructions),
        )

        if candidate_estimate <= effective_budget:
            low = candidate

        else:
            high = candidate - 1

    bounded_payload = _evidence_payload(
        evidence,
        text_char_cap=low,
    )

    bounded_estimate = _estimate_complete_input(
        question=question,
        evidence_payload=bounded_payload,
        calculations_payload=calculations_payload,
        response_schema=response_schema,
        system_instructions=system_instructions,
    )

    truncated_ids = tuple(
        item.evidence_id for item in evidence if (item.text is not None and len(item.text) > low)
    )

    retained_text_chars = sum(
        len(payload_item["text"])
        for payload_item in bounded_payload
        if payload_item["text"] is not None
    )

    return BoundedSynthesisContext(
        evidence_payload=bounded_payload,
        estimated_input_tokens=bounded_estimate,
        input_token_budget=input_token_budget,
        reserve_tokens=reserve_tokens,
        truncated=bool(truncated_ids),
        truncated_evidence_ids=(truncated_ids),
        original_text_chars=(original_text_chars),
        retained_text_chars=(retained_text_chars),
    )
