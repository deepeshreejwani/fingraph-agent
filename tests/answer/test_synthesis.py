"""Focused grounded-answer tests."""

from fingraph.agent import Evidence
from fingraph.answer.context_budget import (
    build_bounded_synthesis_context,
)
from fingraph.answer.models import (
    SynthesisPlan,
)
from fingraph.answer.synthesis import (
    SYSTEM_INSTRUCTIONS,
    build_answer_citations,
    build_deterministic_calculations,
)


def _financial_row(
    *,
    ticker: str,
    year: int,
    concept: str,
    value: int | float,
    fact_id: str,
    accession_number: str,
    unit: str = "USD",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, object]:
    """Build one canonical raw financial row for synthesis tests."""

    resolved_start_date = start_date or f"{year - 1}-07-01"
    resolved_end_date = end_date or f"{year}-06-30"

    return {
        "ticker": ticker,
        "year": year,
        "metric_concept": concept,
        "value": value,
        "unit": unit,
        "fact_id": fact_id,
        "start_date": resolved_start_date,
        "end_date": resolved_end_date,
        "accession_number": accession_number,
    }


def test_graph_change_is_calculated_deterministically() -> None:
    evidence = [
        Evidence(
            evidence_id="graph-1",
            source_type="graph",
            ticker="MSFT",
            structured_payload={
                "rows": [
                    _financial_row(
                        ticker="MSFT",
                        year=2024,
                        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                        value=245122000000,
                        fact_id="msft-revenue-2024",
                        accession_number="msft-2024-10k",
                    ),
                    _financial_row(
                        ticker="MSFT",
                        year=2026,
                        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                        value=331839000000,
                        fact_id="msft-revenue-2026",
                        accession_number="msft-2026-10k",
                    ),
                ]
            },
        )
    ]

    calculations = build_deterministic_calculations(
        question="What was the percentage change in Microsoft revenue from 2024 to 2026?",
        evidence=evidence,
    )

    assert len(calculations) == 1

    result = calculations[0]

    assert result.ticker == "MSFT"
    assert result.unit == "USD"
    assert result.metric == "revenue"
    assert result.absolute_change == "86717000000"
    assert result.percentage_change == "35.377"
    assert result.start_fact_id == "msft-revenue-2024"
    assert result.end_fact_id == "msft-revenue-2026"


def test_no_calculation_for_non_calculation_question() -> None:
    evidence = [
        Evidence(
            evidence_id="graph-1",
            source_type="graph",
            structured_payload={
                "rows": [
                    {
                        "year": 2026,
                        "revenue": 331839000000,
                    }
                ]
            },
        )
    ]

    calculations = build_deterministic_calculations(
        question="What was Microsoft revenue in 2026?",
        evidence=evidence,
    )

    assert calculations == []


def test_vector_citation_uses_trusted_metadata() -> None:
    evidence = [
        Evidence(
            evidence_id="vector-1",
            source_type="vector",
            ticker="NVDA",
            filing_id="accession-1",
            section_name=("Item 1A. Risk Factors"),
            text="Risk text",
            source_url=("https://www.sec.gov/example"),
        )
    ]

    plan = SynthesisPlan(
        answer="NVIDIA describes risk.",
        cited_evidence_ids=["vector-1"],
        insufficient_evidence=False,
    )

    citations = build_answer_citations(
        plan=plan,
        evidence=evidence,
    )

    assert len(citations) == 1

    assert citations[0].filing_id == "accession-1"

    assert citations[0].source_url == ("https://www.sec.gov/example")


def test_graph_citations_expand_filing_provenance() -> None:
    evidence = [
        Evidence(
            evidence_id="graph-1",
            source_type="graph",
            ticker="MSFT",
            structured_payload={
                "rows": [
                    {
                        "year": 2026,
                        "revenue": (331839000000),
                    }
                ]
            },
            provenance={
                "filings": [
                    {
                        "accession_number": ("0001193125-26-323660"),
                        "source_url": ("https://www.sec.gov/msft"),
                        "filing_date": ("2026-07-29"),
                    }
                ]
            },
        )
    ]

    plan = SynthesisPlan(
        answer=("Microsoft reported revenue."),
        cited_evidence_ids=["graph-1"],
        insufficient_evidence=False,
    )

    citations = build_answer_citations(
        plan=plan,
        evidence=evidence,
    )

    assert len(citations) == 1

    assert citations[0].filing_id == ("0001193125-26-323660")

    assert citations[0].source_url == ("https://www.sec.gov/msft")


def test_graph_cagr_is_calculated_deterministically() -> None:
    evidence = [
        Evidence(
            evidence_id="graph-cagr",
            source_type="graph",
            ticker="MSFT",
            structured_payload={
                "rows": [
                    _financial_row(
                        ticker="MSFT",
                        year=2024,
                        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                        value=245122000000,
                        fact_id="msft-revenue-2024",
                        accession_number="msft-2024-10k",
                    ),
                    _financial_row(
                        ticker="MSFT",
                        year=2026,
                        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                        value=331839000000,
                        fact_id="msft-revenue-2026",
                        accession_number="msft-2026-10k",
                    ),
                ]
            },
        )
    ]

    calculations = build_deterministic_calculations(
        question="What was Microsoft's revenue CAGR from 2024 to 2026?",
        evidence=evidence,
    )

    assert len(calculations) == 1

    result = calculations[0]

    assert result.calculation_type == "cagr"
    assert result.ticker == "MSFT"
    assert result.metric == "revenue"
    assert result.start_year == 2024
    assert result.end_year == 2026
    assert result.periods == 2
    assert result.cagr_percentage == "16.352"


def test_graph_operating_margin_is_calculated_deterministically() -> None:
    evidence = [
        Evidence(
            evidence_id="graph-margin",
            source_type="graph",
            ticker="MSFT",
            structured_payload={
                "rows": [
                    _financial_row(
                        ticker="MSFT",
                        year=2025,
                        concept="OperatingIncomeLoss",
                        value=109433000000,
                        fact_id="msft-op-income-2025",
                        accession_number="msft-2025-10k",
                    ),
                    _financial_row(
                        ticker="MSFT",
                        year=2025,
                        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                        value=281724000000,
                        fact_id="msft-revenue-2025",
                        accession_number="msft-2025-10k",
                    ),
                ]
            },
        )
    ]

    calculations = build_deterministic_calculations(
        question="What was Microsoft's operating margin in 2025?",
        evidence=evidence,
    )

    assert len(calculations) == 1

    result = calculations[0]

    assert result.calculation_type == "margin"
    assert result.ticker == "MSFT"
    assert result.metric == "operating_margin"
    assert result.year == 2025
    assert result.numerator_metric == "operating_income"
    assert result.numerator_value == "109433000000"
    assert result.denominator_metric == "revenue"
    assert result.denominator_value == "281724000000"
    assert result.margin_percentage == "38.844"
    assert result.numerator_fact_id == "msft-op-income-2025"
    assert result.denominator_fact_id == "msft-revenue-2025"


def test_margin_is_not_guessed_without_required_operands() -> None:
    evidence = [
        Evidence(
            evidence_id="graph-margin-missing",
            source_type="graph",
            ticker="MSFT",
            structured_payload={
                "rows": [
                    _financial_row(
                        ticker="MSFT",
                        year=2025,
                        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                        value=281724000000,
                        fact_id="msft-revenue-2025",
                        accession_number="msft-2025-10k",
                    )
                ]
            },
        )
    ]

    calculations = build_deterministic_calculations(
        question="What was Microsoft's operating margin in 2025?",
        evidence=evidence,
    )

    assert calculations == []


def test_context_budget_truncates_narrative_text_deterministically() -> None:
    evidence = [
        Evidence(
            evidence_id="vector-long",
            source_type="vector",
            ticker="NVDA",
            filing_id="filing-1",
            text=("Long risk narrative. " * 4000),
            source_url=("https://www.sec.gov/example"),
            provenance={
                "fiscal_year": 2026,
            },
        ),
        Evidence(
            evidence_id="graph-keep",
            source_type="graph",
            ticker="NVDA",
            structured_payload={
                "rows": [
                    {
                        "year": 2026,
                        "revenue": (130497000000),
                    }
                ]
            },
            provenance={
                "filings": [
                    {
                        "accession_number": ("filing-2"),
                        "source_url": ("https://www.sec.gov/graph"),
                    }
                ]
            },
        ),
    ]

    first = build_bounded_synthesis_context(
        question=("What risks does NVIDIA describe?"),
        evidence=evidence,
        calculations_payload=[],
        response_schema=(SynthesisPlan.model_json_schema()),
        system_instructions=(SYSTEM_INSTRUCTIONS),
        input_token_budget=4000,
    )

    second = build_bounded_synthesis_context(
        question=("What risks does NVIDIA describe?"),
        evidence=evidence,
        calculations_payload=[],
        response_schema=(SynthesisPlan.model_json_schema()),
        system_instructions=(SYSTEM_INSTRUCTIONS),
        input_token_budget=4000,
    )

    assert first == second

    assert first.truncated

    assert first.truncated_evidence_ids == ("vector-long",)

    assert first.estimated_input_tokens <= (first.input_token_budget - first.reserve_tokens)

    assert first.evidence_payload[0]["evidence_id"] == "vector-long"

    assert len(first.evidence_payload[0]["text"]) < len(evidence[0].text or "")

    assert first.evidence_payload[1]["evidence_id"] == "graph-keep"

    assert first.evidence_payload[1]["structured_payload"] == evidence[1].structured_payload

    assert first.evidence_payload[1]["provenance"] == evidence[1].provenance


def test_context_budget_does_not_modify_small_evidence() -> None:
    evidence = [
        Evidence(
            evidence_id=("vector-small"),
            source_type="vector",
            ticker="MSFT",
            filing_id="filing-1",
            text=("Microsoft describes a supported filing risk."),
            source_url=("https://www.sec.gov/example"),
            provenance={
                "fiscal_year": 2026,
            },
        )
    ]

    result = build_bounded_synthesis_context(
        question=("What risk does Microsoft describe?"),
        evidence=evidence,
        calculations_payload=[],
        response_schema=(SynthesisPlan.model_json_schema()),
        system_instructions=(SYSTEM_INSTRUCTIONS),
        input_token_budget=16000,
    )

    assert not result.truncated

    assert result.truncated_evidence_ids == ()

    assert result.evidence_payload[0]["text"] == evidence[0].text

    assert result.original_text_chars == result.retained_text_chars


def test_graph_operating_margin_from_metric_concept_rows() -> None:
    evidence = [
        Evidence(
            evidence_id="graph-operating-margin-runtime-shape",
            source_type="graph",
            ticker="MSFT",
            structured_payload={
                "rows": [
                    _financial_row(
                        ticker="MSFT",
                        year=2026,
                        concept="OperatingIncomeLoss",
                        value=155237000000.0,
                        fact_id="msft-op-income-2026",
                        accession_number="msft-2026-10k",
                    ),
                    _financial_row(
                        ticker="MSFT",
                        year=2026,
                        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                        value=331839000000.0,
                        fact_id="msft-revenue-2026",
                        accession_number="msft-2026-10k",
                    ),
                ]
            },
        )
    ]

    calculations = build_deterministic_calculations(
        question="What was Microsoft's operating margin in 2026?",
        evidence=evidence,
    )

    assert len(calculations) == 1

    result = calculations[0]

    assert result.calculation_type == "margin"
    assert result.metric == "operating_margin"
    assert result.year == 2026
    assert result.numerator_metric == "operating_income"
    assert result.numerator_value == "155237000000.0"
    assert result.denominator_metric == "revenue"
    assert result.denominator_value == "331839000000.0"
    assert result.margin_percentage == "46.781"


def test_graph_net_margin_from_metric_concept_rows() -> None:
    evidence = [
        Evidence(
            evidence_id="graph-net-margin-runtime-shape",
            source_type="graph",
            ticker="MSFT",
            structured_payload={
                "rows": [
                    _financial_row(
                        ticker="MSFT",
                        year=2026,
                        concept="NetIncomeLoss",
                        value=133749000000.0,
                        fact_id="msft-net-income-2026",
                        accession_number="msft-2026-10k",
                    ),
                    _financial_row(
                        ticker="MSFT",
                        year=2026,
                        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                        value=331839000000.0,
                        fact_id="msft-revenue-2026",
                        accession_number="msft-2026-10k",
                    ),
                ]
            },
        )
    ]

    calculations = build_deterministic_calculations(
        question="What was Microsoft's net margin in 2026?",
        evidence=evidence,
    )

    assert len(calculations) == 1

    result = calculations[0]

    assert result.calculation_type == "margin"
    assert result.metric == "net_margin"
    assert result.year == 2026
    assert result.numerator_metric == "net_income"
    assert result.numerator_value == "133749000000.0"
    assert result.denominator_metric == "revenue"
    assert result.denominator_value == "331839000000.0"
    assert result.margin_percentage == "40.305"


def test_change_never_combines_different_companies() -> None:
    evidence = [
        Evidence(
            evidence_id="graph-cross-company-change",
            source_type="graph",
            structured_payload={
                "rows": [
                    _financial_row(
                        ticker="MSFT",
                        year=2024,
                        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                        value=200,
                        fact_id="msft-revenue-2024",
                        accession_number="msft-2024-10k",
                    ),
                    _financial_row(
                        ticker="AAPL",
                        year=2025,
                        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                        value=110,
                        fact_id="aapl-revenue-2025",
                        accession_number="aapl-2025-10k",
                    ),
                ]
            },
        )
    ]

    calculations = build_deterministic_calculations(
        question="What was the revenue change from 2024 to 2025?",
        evidence=evidence,
    )

    assert calculations == []


def test_multi_company_change_emits_separate_company_calculations() -> None:
    evidence = [
        Evidence(
            evidence_id="graph-multi-company-change",
            source_type="graph",
            structured_payload={
                "rows": [
                    _financial_row(
                        ticker="AAPL",
                        year=2024,
                        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                        value=100,
                        fact_id="aapl-revenue-2024",
                        accession_number="aapl-2024-10k",
                    ),
                    _financial_row(
                        ticker="AAPL",
                        year=2025,
                        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                        value=110,
                        fact_id="aapl-revenue-2025",
                        accession_number="aapl-2025-10k",
                    ),
                    _financial_row(
                        ticker="MSFT",
                        year=2024,
                        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                        value=200,
                        fact_id="msft-revenue-2024",
                        accession_number="msft-2024-10k",
                    ),
                    _financial_row(
                        ticker="MSFT",
                        year=2025,
                        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                        value=240,
                        fact_id="msft-revenue-2025",
                        accession_number="msft-2025-10k",
                    ),
                ]
            },
        )
    ]

    calculations = build_deterministic_calculations(
        question="What was the revenue change for Apple and Microsoft from 2024 to 2025?",
        evidence=evidence,
    )

    assert [item.ticker for item in calculations] == ["AAPL", "MSFT"]
    assert [item.percentage_change for item in calculations] == ["10.000", "20.000"]


def test_margin_never_combines_different_companies() -> None:
    evidence = [
        Evidence(
            evidence_id="graph-cross-company-margin",
            source_type="graph",
            structured_payload={
                "rows": [
                    _financial_row(
                        ticker="AAPL",
                        year=2025,
                        concept="OperatingIncomeLoss",
                        value=40,
                        fact_id="aapl-op-income-2025",
                        accession_number="aapl-2025-10k",
                    ),
                    _financial_row(
                        ticker="MSFT",
                        year=2025,
                        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                        value=200,
                        fact_id="msft-revenue-2025",
                        accession_number="msft-2025-10k",
                    ),
                ]
            },
        )
    ]

    calculations = build_deterministic_calculations(
        question="What was operating margin in 2025?",
        evidence=evidence,
    )

    assert calculations == []


def test_margin_rejects_mismatched_economic_periods() -> None:
    evidence = [
        Evidence(
            evidence_id="graph-period-mismatch",
            source_type="graph",
            ticker="MSFT",
            structured_payload={
                "rows": [
                    _financial_row(
                        ticker="MSFT",
                        year=2025,
                        concept="OperatingIncomeLoss",
                        value=40,
                        fact_id="msft-op-income-2025",
                        accession_number="msft-2025-10k",
                        start_date="2024-07-01",
                        end_date="2025-06-30",
                    ),
                    _financial_row(
                        ticker="MSFT",
                        year=2025,
                        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                        value=200,
                        fact_id="msft-revenue-2025",
                        accession_number="msft-2025-10k",
                        start_date="2024-01-01",
                        end_date="2024-12-31",
                    ),
                ]
            },
        )
    ]

    calculations = build_deterministic_calculations(
        question="What was Microsoft's operating margin in 2025?",
        evidence=evidence,
    )

    assert calculations == []
