"""Normalization of SEC company facts into canonical financial facts."""

from collections.abc import Mapping
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from fingraph.domain.models import FinancialFact
from fingraph.utils.ids import stable_id


def normalize_company_facts(
    payload: Mapping[str, Any],
    *,
    allowed_accessions: set[str] | None = None,
) -> list[FinancialFact]:
    """Normalize 10-K numeric XBRL facts from SEC companyfacts JSON."""

    cik_raw = payload.get("cik")

    if cik_raw is None:
        raise ValueError("Company facts payload has no CIK.")

    cik = str(cik_raw).zfill(10)

    facts_root = payload.get("facts")

    if not isinstance(
        facts_root,
        Mapping,
    ):
        return []

    normalized: list[FinancialFact] = []

    for (
        taxonomy,
        taxonomy_facts,
    ) in facts_root.items():
        if not isinstance(
            taxonomy_facts,
            Mapping,
        ):
            continue

        for (
            concept,
            concept_payload,
        ) in taxonomy_facts.items():
            if not isinstance(
                concept_payload,
                Mapping,
            ):
                continue

            units = concept_payload.get("units")

            if not isinstance(
                units,
                Mapping,
            ):
                continue

            for (
                unit,
                observations,
            ) in units.items():
                if not isinstance(
                    observations,
                    list,
                ):
                    continue

                for observation in observations:
                    if not isinstance(
                        observation,
                        Mapping,
                    ):
                        continue

                    fact = _normalize_observation(
                        cik=cik,
                        taxonomy=str(taxonomy),
                        concept=str(concept),
                        unit=str(unit),
                        observation=(observation),
                        allowed_accessions=(allowed_accessions),
                    )

                    if fact is not None:
                        normalized.append(fact)

    unique = {fact.fact_id: fact for fact in normalized}

    return sorted(
        unique.values(),
        key=lambda fact: (
            fact.end_date,
            fact.concept,
            fact.fact_id,
        ),
    )


def _normalize_observation(
    *,
    cik: str,
    taxonomy: str,
    concept: str,
    unit: str,
    observation: Mapping[
        str,
        Any,
    ],
    allowed_accessions: (set[str] | None),
) -> FinancialFact | None:
    """Normalize one SEC XBRL observation."""

    if observation.get("form") != "10-K":
        return None

    accession_raw = observation.get("accn")

    end_raw = observation.get("end")

    value_raw = observation.get("val")

    if not accession_raw or not end_raw or value_raw is None:
        return None

    accession = str(accession_raw)

    if allowed_accessions is not None and accession not in allowed_accessions:
        return None

    try:
        value = Decimal(str(value_raw))
    except (
        InvalidOperation,
        ValueError,
    ):
        return None

    start_raw = observation.get("start")

    start_date = date.fromisoformat(str(start_raw)) if start_raw else None

    end_date = date.fromisoformat(str(end_raw))

    fiscal_year_raw = observation.get("fy")

    fiscal_year = int(fiscal_year_raw) if fiscal_year_raw is not None else None

    fiscal_period_raw = observation.get("fp")

    fiscal_period = str(fiscal_period_raw) if fiscal_period_raw else None

    filed_raw = observation.get("filed")

    filed_date = date.fromisoformat(str(filed_raw)) if filed_raw else None

    frame_raw = observation.get("frame")

    frame = str(frame_raw) if frame_raw else None

    fact_id = stable_id(
        "fact",
        cik,
        taxonomy,
        concept,
        unit,
        start_date,
        end_date,
        accession,
        fiscal_year,
        fiscal_period,
        filed_date,
        frame,
        value,
    )

    return FinancialFact(
        fact_id=fact_id,
        cik=cik,
        taxonomy=taxonomy,
        concept=concept,
        value=value,
        unit=unit,
        start_date=start_date,
        end_date=end_date,
        fiscal_year=fiscal_year,
        fiscal_period=(fiscal_period),
        accession_number=(accession),
        filed_date=filed_date,
        frame=frame,
    )
