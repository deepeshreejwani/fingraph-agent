"""Regression test for the Text-to-Cypher WITH-scope contract."""

import re

from fingraph.graph.qa.schema import GRAPH_SCHEMA
from fingraph.llm.groq_client import SYSTEM_INSTRUCTIONS


def _normalize_whitespace(value: str) -> str:
    """Collapse formatting whitespace for stable prompt-contract assertions."""

    return re.sub(r"\s+", " ", value).strip()


def test_schema_recommended_pattern_preserves_company_scope() -> None:
    normalized_schema = _normalize_whitespace(GRAPH_SCHEMA)

    assert "WITH company, period, fact, filing" in normalized_schema
    assert (
        "WITH company, period, collect({ fact: fact, filing: filing })[0] AS latest"
        in normalized_schema
    )
    assert "company.ticker AS ticker" in normalized_schema
    assert "period.period_year AS year" in normalized_schema
    assert "latest.filing.filing_date AS filing_date" in normalized_schema


def test_system_prompt_explicitly_requires_with_scope_preservation() -> None:
    normalized_prompt = _normalize_whitespace(SYSTEM_INSTRUCTIONS)

    assert "Cypher WITH changes variable scope" in normalized_prompt
    assert "preserve the company variable through every WITH" in normalized_prompt
    assert "Never reference a variable after a WITH" in normalized_prompt
