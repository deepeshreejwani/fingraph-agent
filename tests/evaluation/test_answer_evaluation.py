"""Focused answer-evaluation tests."""

from fingraph.agent import Evidence
from fingraph.answer.models import (
    AnswerCitation,
    DeterministicCalculation,
    SynthesisPlan,
)
from fingraph.evaluation.answers import (
    ANSWER_EVALUATOR_VERSION,
    AnswerGoldenCase,
    ExpectedCalculation,
    case_fingerprint,
    evaluate_answer_contract,
    evaluation_fingerprint,
    reusable_result,
)


def test_case_fingerprint_is_stable() -> None:
    case = AnswerGoldenCase(
        id="example",
        question="What was Microsoft revenue in 2026?",
        expected_route="graph",
    )

    assert case_fingerprint(case) == case_fingerprint(case)


def test_evaluation_fingerprint_is_stable() -> None:
    configuration = {
        "llm": {
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
            "reasoning_effort": "low",
        },
        "retrieval": {
            "collection": "fingraph_chunks",
            "evidence_limit": 5,
        },
        "serving_code_hash": "code-hash",
        "processed_corpus_hash": "corpus-hash",
    }

    assert evaluation_fingerprint(configuration) == evaluation_fingerprint(configuration)


def test_evaluation_fingerprint_is_order_independent() -> None:
    first = {
        "llm": {
            "model": "openai/gpt-oss-120b",
            "provider": "groq",
        },
        "retrieval": {
            "evidence_limit": 5,
            "collection": "fingraph_chunks",
        },
    }

    second = {
        "retrieval": {
            "collection": "fingraph_chunks",
            "evidence_limit": 5,
        },
        "llm": {
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
        },
    }

    assert evaluation_fingerprint(first) == evaluation_fingerprint(second)


def test_evaluation_fingerprint_changes_with_configuration() -> None:
    baseline = {
        "llm": {
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
            "reasoning_effort": "low",
        },
        "retrieval": {
            "collection": "fingraph_chunks",
            "evidence_limit": 5,
        },
        "serving_code_hash": "code-hash",
        "processed_corpus_hash": "corpus-hash",
    }

    changed = {
        **baseline,
        "retrieval": {
            "collection": "fingraph_chunks",
            "evidence_limit": 10,
        },
    }

    assert evaluation_fingerprint(baseline) != evaluation_fingerprint(changed)


def test_evaluation_fingerprint_changes_with_serving_code() -> None:
    baseline = {
        "serving_code_hash": "code-v1",
        "processed_corpus_hash": "corpus-v1",
    }

    changed = {
        "serving_code_hash": "code-v2",
        "processed_corpus_hash": "corpus-v1",
    }

    assert evaluation_fingerprint(baseline) != evaluation_fingerprint(changed)


def test_evaluation_fingerprint_changes_with_corpus() -> None:
    baseline = {
        "serving_code_hash": "code-v1",
        "processed_corpus_hash": "corpus-v1",
    }

    changed = {
        "serving_code_hash": "code-v1",
        "processed_corpus_hash": "corpus-v2",
    }

    assert evaluation_fingerprint(baseline) != evaluation_fingerprint(changed)


def test_valid_graph_answer_contract_passes() -> None:
    case = AnswerGoldenCase(
        id="msft",
        question=("What was Microsoft revenue in 2026?"),
        expected_route="graph",
        expected_fiscal_years=(2026,),
    )

    evidence = [
        Evidence(
            evidence_id="graph-1",
            source_type="graph",
            ticker="MSFT",
            filing_id="filing-1",
            source_url=("https://sec.example/msft"),
            structured_payload={
                "rows": [
                    {
                        "year": 2026,
                        "revenue": (331839000000),
                    }
                ]
            },
        )
    ]

    citation = AnswerCitation(
        label=1,
        evidence_id="graph-1",
        source_type="graph",
        ticker="MSFT",
        filing_id="filing-1",
        source_url=("https://sec.example/msft"),
    )

    result = evaluate_answer_contract(
        case=case,
        route="graph",
        plan=SynthesisPlan(
            answer=("Microsoft revenue was $331.839B."),
            cited_evidence_ids=["graph-1"],
            insufficient_evidence=False,
        ),
        citations=[citation],
        calculations=[],
        evidence=evidence,
    )

    assert result.passed


def test_unknown_citation_fails_contract() -> None:
    case = AnswerGoldenCase(
        id="citation",
        question="Question",
        expected_route="vector",
    )

    evidence = [
        Evidence(
            evidence_id="real",
            source_type="vector",
            ticker="NVDA",
            filing_id="filing",
            section_name=("Item 1A. Risk Factors"),
            text="Evidence",
            source_url=("https://sec.example/nvda"),
            provenance={
                "fiscal_year": 2026,
            },
        )
    ]

    result = evaluate_answer_contract(
        case=case,
        route="vector",
        plan=SynthesisPlan(
            answer="Answer",
            cited_evidence_ids=["fake"],
            insufficient_evidence=False,
        ),
        citations=[
            AnswerCitation(
                label=1,
                evidence_id="fake",
                source_type="vector",
                ticker="NVDA",
                filing_id="filing",
                source_url=("https://sec.example/nvda"),
            )
        ],
        calculations=[],
        evidence=evidence,
    )

    assert not result.passed

    assert "citation_failure" in result.failure_categories


def test_temporal_contract_requires_expected_year() -> None:
    case = AnswerGoldenCase(
        id="temporal",
        question=("What risks does NVIDIA describe in its 10-K?"),
        expected_route="vector",
        expected_fiscal_years=(2026,),
    )

    evidence = [
        Evidence(
            evidence_id="old",
            source_type="vector",
            ticker="NVDA",
            filing_id="filing-2022",
            section_name=("Item 1A. Risk Factors"),
            text="Old evidence",
            source_url=("https://sec.example/2022"),
            provenance={
                "fiscal_year": 2022,
            },
        )
    ]

    result = evaluate_answer_contract(
        case=case,
        route="vector",
        plan=SynthesisPlan(
            answer="Answer",
            cited_evidence_ids=["old"],
            insufficient_evidence=False,
        ),
        citations=[
            AnswerCitation(
                label=1,
                evidence_id="old",
                source_type="vector",
                ticker="NVDA",
                filing_id="filing-2022",
                source_url=("https://sec.example/2022"),
            )
        ],
        calculations=[],
        evidence=evidence,
    )

    assert not result.temporal_passed


def test_deterministic_calculation_matches_decimal_formatting() -> None:
    case = AnswerGoldenCase(
        id="calc",
        question="Question",
        expected_route="graph",
        expected_calculation=(
            ExpectedCalculation(
                start_year=2024,
                end_year=2026,
                start_value=("245122000000"),
                end_value=("331839000000"),
                absolute_change=("86717000000"),
                percentage_change=("35.377"),
            )
        ),
    )

    result = evaluate_answer_contract(
        case=case,
        route="graph",
        plan=SynthesisPlan(
            answer="Answer",
            cited_evidence_ids=["graph"],
            insufficient_evidence=False,
        ),
        citations=[
            AnswerCitation(
                label=1,
                evidence_id="graph",
                source_type="graph",
                filing_id="filing",
                source_url=("https://sec.example"),
            )
        ],
        calculations=[
            DeterministicCalculation(
                ticker="MSFT",
                metric=("revenue_value"),
                unit="USD",
                source_evidence_ids=["graph"],
                start_year=2024,
                end_year=2026,
                start_value=("245122000000.0"),
                end_value=("331839000000.0"),
                start_fact_id="msft-revenue-2024",
                end_fact_id="msft-revenue-2026",
                start_accession_number="msft-2024-10k",
                end_accession_number="msft-2026-10k",
                start_period_end_date="2024-06-30",
                end_period_end_date="2026-06-30",
                absolute_change=("86717000000.0"),
                percentage_change=("35.377"),
            )
        ],
        evidence=[
            Evidence(
                evidence_id="graph",
                source_type="graph",
                filing_id="filing",
                source_url=("https://sec.example"),
                structured_payload={
                    "rows": [
                        {
                            "year": 2024,
                        },
                        {
                            "year": 2026,
                        },
                    ]
                },
            )
        ],
    )

    assert result.calculation_passed


def test_completed_matching_case_is_reusable() -> None:
    case = AnswerGoldenCase(
        id="resume",
        question="Question",
        expected_route="vector",
    )

    evaluation_hash = evaluation_fingerprint(
        {
            "model": "openai/gpt-oss-120b",
            "serving_code_hash": "code-v1",
        }
    )

    previous = {
        "id": "resume",
        "case_hash": (case_fingerprint(case)),
        "evaluator_version": (ANSWER_EVALUATOR_VERSION),
        "evaluation_hash": evaluation_hash,
        "status": "completed",
        "passed": True,
    }

    assert reusable_result(
        previous=previous,
        case=case,
        evaluation_hash=evaluation_hash,
        rerun_failed=False,
    )


def test_completed_case_without_evaluation_hash_is_not_reused() -> None:
    case = AnswerGoldenCase(
        id="resume",
        question="Question",
        expected_route="vector",
    )

    evaluation_hash = evaluation_fingerprint(
        {
            "model": "openai/gpt-oss-120b",
        }
    )

    previous = {
        "id": "resume",
        "case_hash": (case_fingerprint(case)),
        "evaluator_version": (ANSWER_EVALUATOR_VERSION),
        "status": "completed",
        "passed": True,
    }

    assert not reusable_result(
        previous=previous,
        case=case,
        evaluation_hash=evaluation_hash,
        rerun_failed=False,
    )


def test_completed_case_with_changed_evaluation_hash_is_not_reused() -> None:
    case = AnswerGoldenCase(
        id="resume",
        question="Question",
        expected_route="vector",
    )

    previous_evaluation_hash = evaluation_fingerprint(
        {
            "model": "openai/gpt-oss-120b",
            "evidence_limit": 5,
        }
    )

    current_evaluation_hash = evaluation_fingerprint(
        {
            "model": "openai/gpt-oss-120b",
            "evidence_limit": 10,
        }
    )

    previous = {
        "id": "resume",
        "case_hash": (case_fingerprint(case)),
        "evaluator_version": (ANSWER_EVALUATOR_VERSION),
        "evaluation_hash": previous_evaluation_hash,
        "status": "completed",
        "passed": True,
    }

    assert not reusable_result(
        previous=previous,
        case=case,
        evaluation_hash=current_evaluation_hash,
        rerun_failed=False,
    )


def test_deferred_case_is_not_reused() -> None:
    case = AnswerGoldenCase(
        id="resume",
        question="Question",
        expected_route="vector",
    )

    evaluation_hash = evaluation_fingerprint(
        {
            "model": "openai/gpt-oss-120b",
        }
    )

    previous = {
        "id": "resume",
        "case_hash": (case_fingerprint(case)),
        "evaluator_version": (ANSWER_EVALUATOR_VERSION),
        "evaluation_hash": evaluation_hash,
        "status": "deferred",
        "passed": False,
    }

    assert not reusable_result(
        previous=previous,
        case=case,
        evaluation_hash=evaluation_hash,
        rerun_failed=False,
    )
