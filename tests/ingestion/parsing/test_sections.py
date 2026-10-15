"""Regression tests for deterministic 10-K section extraction."""

from fingraph.domain.models import Section
from fingraph.ingestion.parsing.sections import (
    extract_sections,
)


def body(
    label: str,
) -> str:
    """Return enough deterministic text to exceed the TOC filter."""

    return f"{label} " + "material filing narrative " * 100


def large_body(
    label: str,
    *,
    minimum_chars: int,
) -> str:
    """Return deterministic text at least as large as requested."""

    unit = f"{label} material filing narrative "
    repeats = (minimum_chars // len(unit)) + 1

    return unit * repeats


def section_map(
    sections: list[Section],
) -> dict[str, str]:
    """Map extracted canonical section names to text."""

    return {section.section_name: section.text for section in sections}


def test_extracts_headings_with_trailing_colons() -> None:
    text = "\n".join(
        [
            "Item 1. Business:",
            body("business"),
            "Item 1A. Risk Factors:",
            body("risk"),
            "Item 1B. Unresolved Staff Comments:",
            "None",
            "Item 7. Management's Discussion and Analysis "
            "of Financial Condition and Results of Operations:",
            body("mda"),
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk:",
            body("market-risk"),
            "Item 8. Financial Statements and Supplementary Data:",
            body("financial-statements"),
        ]
    )

    sections = extract_sections(
        text,
        filing_id="test-filing-colons",
    )

    assert [section.section_name for section in sections] == [
        "Item 1. Business",
        "Item 1A. Risk Factors",
        "Item 7. MD&A",
        "Item 7A. Market Risk",
    ]


def test_extracts_title_on_following_line_with_colon() -> None:
    text = "\n".join(
        [
            "Item 1.",
            "Business:",
            body("business"),
            "Item 1A.",
            "Risk Factors:",
            body("risk"),
            "Item 1B.",
            "Unresolved Staff Comments:",
            "None",
            "Item 7.",
            "Management's Discussion and Analysis "
            "of Financial Condition and Results of Operations:",
            body("mda"),
            "Item 7A.",
            "Quantitative and Qualitative Disclosures About Market Risk:",
            body("market-risk"),
            "Item 8.",
            "Financial Statements and Supplementary Data:",
            body("financial-statements"),
        ]
    )

    sections = extract_sections(
        text,
        filing_id="test-filing-multiline",
    )

    assert len(sections) == 4

    assert [section.section_name for section in sections] == [
        "Item 1. Business",
        "Item 1A. Risk Factors",
        "Item 7. MD&A",
        "Item 7A. Market Risk",
    ]


def test_short_table_of_contents_candidate_is_not_selected() -> None:
    table_of_contents = "\n".join(
        [
            "Table of Contents",
            "Item 1. Business",
            "1",
            "Item 1A. Risk Factors",
            "3",
            "Item 7. Management's Discussion and Analysis "
            "of Financial Condition and Results of Operations",
            "11",
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
            "11",
            "Item 8. Financial Statements and Supplementary Data",
            "12",
        ]
    )

    actual_body = "\n".join(
        [
            "Item 1. Business:",
            body("real-business"),
            "Item 1A. Risk Factors:",
            body("real-risk"),
            "Item 1B. Unresolved Staff Comments:",
            "None",
            "Item 7. Management's Discussion and Analysis "
            "of Financial Condition and Results of Operations:",
            body("real-mda"),
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk:",
            body("real-market-risk"),
            "Item 8. Financial Statements and Supplementary Data:",
            body("financial-statements"),
        ]
    )

    sections = extract_sections(
        table_of_contents + "\n\n" + actual_body,
        filing_id="test-filing-toc",
    )

    assert len(sections) == 4

    assert all("real-" in section.text for section in sections)


def test_semantic_heading_fallback_extracts_custom_layout() -> None:
    table_of_contents = "\n".join(
        [
            "Page",
            "Our Business",
            "6",
            "Management's Discussion and Analysis",
            "18",
            "Quantitative and Qualitative Disclosures About Market Risk",
            "33",
            "Critical Accounting Estimates",
            "34",
            "Risk Factors",
            "37",
            "Other Key Information",
            "52",
        ]
    )

    actual_body = "\n".join(
        [
            "Our Business",
            body("fallback-business"),
            "Management's Discussion and Analysis",
            body("fallback-mda"),
            "Quantitative and Qualitative Disclosures About Market Risk",
            body("fallback-market-risk"),
            "Critical Accounting Estimates",
            body("critical-accounting"),
            "Risk Factors",
            body("fallback-risk"),
            "Properties",
            body("properties"),
        ]
    )

    sections = extract_sections(
        table_of_contents + "\n\n" + actual_body,
        filing_id="test-filing-semantic-layout",
    )

    extracted = section_map(sections)

    assert set(extracted) == {
        "Item 1. Business",
        "Item 1A. Risk Factors",
        "Item 7. MD&A",
        "Item 7A. Market Risk",
    }

    assert "fallback-business" in extracted["Item 1. Business"]

    assert "fallback-risk" in extracted["Item 1A. Risk Factors"]

    assert "fallback-mda" in extracted["Item 7. MD&A"]

    assert "fallback-market-risk" in extracted["Item 7A. Market Risk"]


def test_primary_item_extraction_precedes_semantic_fallback() -> None:
    canonical = "\n".join(
        [
            "Item 1. Business:",
            body("canonical-business"),
            "Item 1A. Risk Factors:",
            body("canonical-risk"),
            "Item 1B. Unresolved Staff Comments:",
            "None",
            "Item 7. Management's Discussion and Analysis "
            "of Financial Condition and Results of Operations:",
            body("canonical-mda"),
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk:",
            body("canonical-market-risk"),
            "Item 8. Financial Statements and Supplementary Data:",
            body("financial-statements"),
        ]
    )

    semantic_fallback = "\n".join(
        [
            "Our Business",
            body("fallback-business"),
            "Management's Discussion and Analysis",
            body("fallback-mda"),
            "Quantitative and Qualitative Disclosures About Market Risk",
            body("fallback-market-risk"),
            "Critical Accounting Estimates",
            body("critical-accounting"),
            "Risk Factors",
            body("fallback-risk"),
            "Properties",
            body("properties"),
        ]
    )

    sections = extract_sections(
        canonical + "\n\n" + semantic_fallback,
        filing_id="test-filing-primary-precedence",
    )

    extracted = section_map(sections)

    assert "canonical-business" in extracted["Item 1. Business"]

    assert "fallback-business" not in extracted["Item 1. Business"]

    assert "canonical-risk" in extracted["Item 1A. Risk Factors"]

    assert "fallback-risk" not in extracted["Item 1A. Risk Factors"]

    assert "canonical-mda" in extracted["Item 7. MD&A"]

    assert "fallback-mda" not in extracted["Item 7. MD&A"]

    assert "canonical-market-risk" in extracted["Item 7A. Market Risk"]

    assert "fallback-market-risk" not in extracted["Item 7A. Market Risk"]


def test_market_risk_accepts_critical_accounting_policies_boundary() -> None:
    text = "\n".join(
        [
            "Quantitative and Qualitative Disclosures About Market Risk",
            body("market-risk"),
            "Critical Accounting Policies and Estimates",
            body("critical-accounting"),
            "Risk Factors",
            body("risk"),
            "Properties",
        ]
    )

    sections = extract_sections(
        text,
        filing_id="test-market-risk-policy-boundary",
    )

    extracted = section_map(sections)

    assert "market-risk" in extracted["Item 7A. Market Risk"]

    assert "critical-accounting" not in extracted["Item 7A. Market Risk"]


def test_oversized_primary_market_risk_is_rejected_for_valid_fallback() -> None:
    oversized_primary = "\n".join(
        [
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk:",
            large_body(
                "wrong-primary-content",
                minimum_chars=90_000,
            ),
            "Item 8. Financial Statements and Supplementary Data:",
        ]
    )

    semantic_fallback = "\n".join(
        [
            "Quantitative and Qualitative Disclosures About Market Risk",
            body("correct-fallback-market-risk"),
            "Critical Accounting Estimates",
            body("critical-accounting"),
            "Risk Factors",
            body("risk"),
            "Properties",
        ]
    )

    sections = extract_sections(
        oversized_primary + "\n\n" + semantic_fallback,
        filing_id="test-market-risk-oversized-primary",
    )

    extracted = section_map(sections)

    assert "correct-fallback-market-risk" in extracted["Item 7A. Market Risk"]

    assert "wrong-primary-content" not in extracted["Item 7A. Market Risk"]


def test_oversized_market_risk_without_valid_fallback_is_omitted() -> None:
    text = "\n".join(
        [
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk:",
            large_body(
                "oversized-market-risk",
                minimum_chars=90_000,
            ),
            "Item 8. Financial Statements and Supplementary Data:",
        ]
    )

    sections = extract_sections(
        text,
        filing_id="test-market-risk-oversized-only",
    )

    extracted = section_map(sections)

    assert "Item 7A. Market Risk" not in extracted


def test_risk_factors_ignores_other_key_information_page_heading() -> None:
    text = "\n".join(
        [
            "Risk Factors",
            body("risk-part-one"),
            "Other Key Information",
            body("risk-part-two"),
            "Other Key Information",
            body("risk-part-three"),
            "Properties",
            body("properties"),
        ]
    )

    sections = extract_sections(
        text,
        filing_id="test-risk-factors-running-heading",
    )

    extracted = section_map(sections)

    risk_text = extracted["Item 1A. Risk Factors"]

    assert "risk-part-one" in risk_text
    assert "risk-part-two" in risk_text
    assert "risk-part-three" in risk_text
    assert "properties" not in risk_text
