"""Focused deterministic company/citation grounding tests."""

from fingraph.agent import Evidence
from fingraph.answer.grounding import evaluate_company_citation_grounding


def _vector(
    evidence_id: str,
    ticker: str,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type="vector",
        ticker=ticker,
        filing_id=f"filing-{ticker}",
        section_name="Item 1A. Risk Factors",
        text=f"{ticker} risk evidence.",
        source_url=f"https://sec.example/{ticker}",
        rank=1,
    )


def test_multi_company_alignment_rejects_cross_company_inline_id() -> None:
    evidence = [
        _vector("evidence_meta", "META"),
        _vector("evidence_msft", "MSFT"),
    ]

    result = evaluate_company_citation_grounding(
        answer=(
            "Microsoft described infrastructure risks "
            "(evidence_meta) (evidence_msft).\n\n"
            "Meta described AI risks (evidence_meta)."
        ),
        cited_evidence_ids=[
            "evidence_meta",
            "evidence_msft",
        ],
        evidence=evidence,
    )

    assert not result.passed
    assert any("evidence_meta:company_mismatch" in reason for reason in result.failure_reasons)


def test_multi_company_alignment_accepts_company_local_ids() -> None:
    evidence = [
        _vector("evidence_meta", "META"),
        _vector("evidence_msft", "MSFT"),
    ]

    result = evaluate_company_citation_grounding(
        answer=(
            "Microsoft described infrastructure risks "
            "(evidence_msft).\n\n"
            "Meta described AI risks (evidence_meta)."
        ),
        cited_evidence_ids=[
            "evidence_meta",
            "evidence_msft",
        ],
        evidence=evidence,
    )

    assert result.passed


def test_multi_company_alignment_requires_inline_company_mapping() -> None:
    evidence = [
        _vector("evidence_meta", "META"),
        _vector("evidence_msft", "MSFT"),
    ]

    result = evaluate_company_citation_grounding(
        answer=(
            "Microsoft described infrastructure risks.\n\nMeta described AI risks (evidence_meta)."
        ),
        cited_evidence_ids=[
            "evidence_meta",
            "evidence_msft",
        ],
        evidence=evidence,
    )

    assert not result.passed
    assert any(
        "company_claim_without_inline_evidence" in reason for reason in result.failure_reasons
    )


def test_single_company_answer_does_not_require_inline_mapping() -> None:
    evidence = [
        _vector("evidence_msft", "MSFT"),
    ]

    result = evaluate_company_citation_grounding(
        answer="Microsoft described infrastructure risks.",
        cited_evidence_ids=["evidence_msft"],
        evidence=evidence,
    )

    assert result.passed
