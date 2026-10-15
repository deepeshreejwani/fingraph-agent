"""Regression coverage for evidence usability and company citation grounding."""

from fingraph.agent import Evidence
from fingraph.answer.grounding import (
    evaluate_company_citation_grounding,
    is_usable_evidence,
)


def _vector_evidence(
    *,
    evidence_id: str,
    ticker: str,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type="vector",
        ticker=ticker,
        filing_id=(f"{ticker.lower()}-filing"),
        text=(f"{ticker} filing evidence."),
        source_url=(f"https://sec.example/{ticker.lower()}"),
    )


def test_empty_graph_wrapper_is_not_usable_evidence() -> None:
    item = Evidence(
        evidence_id="graph-empty",
        source_type="graph",
        structured_payload={
            "rows": [],
        },
    )

    assert not is_usable_evidence(item)


def test_nonempty_graph_rows_are_usable_evidence() -> None:
    item = Evidence(
        evidence_id="graph-row",
        source_type="graph",
        structured_payload={
            "rows": [
                {
                    "ticker": "MSFT",
                    "year": 2025,
                    "value": 1,
                }
            ]
        },
    )

    assert is_usable_evidence(item)


def test_multi_company_claim_requires_every_company_to_be_covered() -> None:
    apple = _vector_evidence(
        evidence_id="evidence-apple",
        ticker="AAPL",
    )

    result = evaluate_company_citation_grounding(
        answer=("Apple and Microsoft both increased revenue. evidence-apple"),
        cited_evidence_ids=["evidence-apple"],
        evidence=[apple],
    )

    assert not result.passed

    assert any("missing_company_evidence:MSFT" in reason for reason in result.failure_reasons)


def test_multi_company_claim_passes_with_complete_company_coverage() -> None:
    apple = _vector_evidence(
        evidence_id="evidence-apple",
        ticker="AAPL",
    )

    microsoft = _vector_evidence(
        evidence_id="evidence-msft",
        ticker="MSFT",
    )

    result = evaluate_company_citation_grounding(
        answer=("Apple and Microsoft both increased revenue. evidence-apple evidence-msft"),
        cited_evidence_ids=[
            "evidence-apple",
            "evidence-msft",
        ],
        evidence=[
            apple,
            microsoft,
        ],
    )

    assert result.passed


def test_single_company_claim_rejects_wrong_company_evidence() -> None:
    apple = _vector_evidence(
        evidence_id="evidence-apple",
        ticker="AAPL",
    )

    result = evaluate_company_citation_grounding(
        answer=("Microsoft increased revenue. evidence-apple"),
        cited_evidence_ids=["evidence-apple"],
        evidence=[apple],
    )

    assert not result.passed


def test_plain_now_in_prose_is_not_treated_as_servicenow_ticker() -> None:
    nvidia = _vector_evidence(
        evidence_id="evidence-nvda",
        ticker="NVDA",
    )

    result = evaluate_company_citation_grounding(
        answer=("NVIDIA now operates across accelerated computing markets. evidence-nvda"),
        cited_evidence_ids=["evidence-nvda"],
        evidence=[nvidia],
    )

    assert result.passed


def test_servicenow_company_name_still_requires_servicenow_evidence() -> None:
    nvidia = _vector_evidence(
        evidence_id="evidence-nvda",
        ticker="NVDA",
    )

    result = evaluate_company_citation_grounding(
        answer=("NVIDIA and ServiceNow operate in different markets. evidence-nvda"),
        cited_evidence_ids=["evidence-nvda"],
        evidence=[nvidia],
    )

    assert not result.passed
    assert any("missing_company_evidence:NOW" in reason for reason in result.failure_reasons)
