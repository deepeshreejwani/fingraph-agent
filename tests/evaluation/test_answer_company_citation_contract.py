"""Answer-contract regression for company/citation grounding."""

from fingraph.agent import Evidence
from fingraph.answer.models import AnswerCitation, SynthesisPlan
from fingraph.evaluation.answers import (
    AnswerGoldenCase,
    evaluate_answer_contract,
)


def _evidence(
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


def _citation(
    label: int,
    evidence_id: str,
    ticker: str,
) -> AnswerCitation:
    return AnswerCitation(
        label=label,
        evidence_id=evidence_id,
        source_type="vector",
        ticker=ticker,
        filing_id=f"filing-{ticker}",
        section_name="Item 1A. Risk Factors",
        source_url=f"https://sec.example/{ticker}",
    )


def test_answer_contract_flags_cross_company_citation() -> None:
    evidence = [
        _evidence("evidence_meta", "META"),
        _evidence("evidence_msft", "MSFT"),
    ]

    result = evaluate_answer_contract(
        case=AnswerGoldenCase(
            id="multi-company-grounding",
            question="Compare Microsoft and Meta risks.",
            expected_route="vector",
            min_citations=2,
            semantic_judge=False,
        ),
        route="vector",
        plan=SynthesisPlan(
            answer=(
                "Microsoft described infrastructure risks "
                "(evidence_meta) (evidence_msft).\n\n"
                "Meta described AI risks (evidence_meta)."
            ),
            cited_evidence_ids=[
                "evidence_meta",
                "evidence_msft",
            ],
            insufficient_evidence=False,
        ),
        citations=[
            _citation(1, "evidence_meta", "META"),
            _citation(2, "evidence_msft", "MSFT"),
        ],
        calculations=[],
        evidence=evidence,
    )

    assert not result.passed
    assert not result.citation_entity_passed
    assert "citation_entity_failure" in result.failure_categories
