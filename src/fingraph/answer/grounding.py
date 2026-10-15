"""Deterministic company-to-citation grounding checks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from fingraph.agent.evidence import Evidence
from fingraph.company_registry import COMPANIES_BY_TICKER, extract_company_tickers

_SEGMENT_SPLIT_RE = re.compile(r"(?:\n\s*\n|\n(?=\s*[-*•]))")


@dataclass(frozen=True)
class CompanyCitationGroundingResult:
    """Deterministic result for company-local citation alignment."""

    passed: bool
    failure_reasons: tuple[str, ...]


def _has_company_alias_mention(
    *,
    text: str,
    aliases: tuple[str, ...],
) -> bool:
    """Return whether prose explicitly names a company through a registry alias."""

    for alias in aliases:
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
            re.IGNORECASE,
        )

        if pattern.search(text):
            return True

    return False


def _has_explicit_ticker_syntax(
    *,
    text: str,
    ticker: str,
) -> bool:
    """Return whether an otherwise ambiguous ticker is written as a ticker."""

    escaped = re.escape(ticker)

    patterns = (
        re.compile(rf"(?<![A-Za-z0-9])\${escaped}(?![A-Za-z0-9])"),
        re.compile(rf"\(\s*{escaped}\s*\)"),
        re.compile(
            rf"(?i)(?<![A-Za-z0-9])"
            rf"(?:ticker|nasdaq|nyse)\s*(?::|=)?\s*{escaped}"
            rf"(?![A-Za-z0-9])"
        ),
    )

    return any(pattern.search(text) for pattern in patterns)


def _mentioned_tickers(
    text: str,
) -> set[str]:
    """
    Extract supported company mentions from generated answer prose.

    Registry aliases are always accepted. A plain ticker token is also accepted
    when the registry intentionally exposes that ticker as an alias. Otherwise,
    the token must use explicit ticker syntax so ordinary prose such as ``NOW``
    cannot be mistaken for a company mention.
    """

    extracted = set(extract_company_tickers(text))

    mentioned: set[str] = set()

    for ticker in extracted:
        profile = COMPANIES_BY_TICKER[ticker]

        normalized_aliases = tuple(alias.lower() for alias in profile.aliases)

        if ticker.lower() in normalized_aliases:
            mentioned.add(ticker)
            continue

        if _has_company_alias_mention(
            text=text,
            aliases=profile.aliases,
        ):
            mentioned.add(ticker)
            continue

        if _has_explicit_ticker_syntax(
            text=text,
            ticker=ticker,
        ):
            mentioned.add(ticker)

    return mentioned


def _evidence_tickers(
    item: Evidence,
) -> set[str]:
    """Return companies that one evidence item can truthfully support."""

    tickers: set[str] = set()

    if item.ticker:
        tickers.add(str(item.ticker).upper())

    if item.source_type != "graph":
        return tickers

    filings = item.provenance.get(
        "filings",
        [],
    )

    if isinstance(
        filings,
        list,
    ):
        for filing in filings:
            if not isinstance(
                filing,
                dict,
            ):
                continue

            raw_ticker = filing.get("ticker")

            if raw_ticker is not None:
                tickers.add(str(raw_ticker).upper())

    payload = item.structured_payload or {}

    rows = payload.get(
        "rows",
        [],
    )

    if isinstance(
        rows,
        list,
    ):
        for row in rows:
            if not isinstance(
                row,
                dict,
            ):
                continue

            raw_ticker = row.get("ticker")

            if raw_ticker is not None:
                tickers.add(str(raw_ticker).upper())

    return tickers


def is_usable_evidence(
    item: Evidence,
) -> bool:
    """Return whether one evidence wrapper contains material answer evidence."""

    if item.source_type == "graph":
        payload = item.structured_payload or {}

        rows = payload.get(
            "rows",
            [],
        )

        return isinstance(
            rows,
            list,
        ) and any(
            isinstance(
                row,
                dict,
            )
            and bool(row)
            for row in rows
        )

    if item.source_type == "vector":
        return bool(item.text and item.text.strip())

    return False


def _validate_single_company_grounding(
    *,
    ticker: str,
    cited_ids: set[str],
    evidence_by_id: dict[str, Evidence],
) -> CompanyCitationGroundingResult:
    """
    Validate one-company answers without requiring inline citation placement.

    A declared citation must contain usable evidence that supports the single
    company named in the answer.
    """

    failures: list[str] = []

    if not cited_ids:
        return CompanyCitationGroundingResult(
            passed=False,
            failure_reasons=("single_company_claim_without_cited_evidence",),
        )

    aligned = False

    for evidence_id in sorted(cited_ids):
        item = evidence_by_id.get(evidence_id)

        if item is None:
            failures.append(f"{evidence_id}:unknown_evidence_id")
            continue

        if not is_usable_evidence(item):
            failures.append(f"{evidence_id}:unusable_evidence")
            continue

        supported = _evidence_tickers(item)

        if ticker not in supported:
            failures.append(f"{evidence_id}:company_mismatch")
            continue

        aligned = True

    if not aligned:
        failures.append(f"missing_company_evidence:{ticker}")

    return CompanyCitationGroundingResult(
        passed=aligned,
        failure_reasons=tuple(failures),
    )


def evaluate_company_citation_grounding(
    *,
    answer: str,
    cited_evidence_ids: list[str],
    evidence: list[Evidence],
) -> CompanyCitationGroundingResult:
    """
    Verify deterministic alignment between company claims and citations.

    Single-company answers require at least one declared usable citation that
    supports that company, but do not require the literal evidence ID inline.

    Multi-company answers additionally require company-local inline evidence
    IDs so every company-specific claim can be audited independently.
    """

    evidence_by_id = {item.evidence_id: item for item in evidence}

    cited_ids = set(cited_evidence_ids)

    failures: list[str] = []

    answer_without_ids = answer

    for evidence_id in evidence_by_id:
        answer_without_ids = answer_without_ids.replace(
            evidence_id,
            "",
        )

    all_answer_tickers = _mentioned_tickers(answer_without_ids)

    if not all_answer_tickers:
        return CompanyCitationGroundingResult(
            passed=True,
            failure_reasons=(),
        )

    if len(all_answer_tickers) == 1:
        ticker = next(iter(all_answer_tickers))

        return _validate_single_company_grounding(
            ticker=ticker,
            cited_ids=cited_ids,
            evidence_by_id=evidence_by_id,
        )

    segments = [segment.strip() for segment in _SEGMENT_SPLIT_RE.split(answer) if segment.strip()]

    for index, segment in enumerate(
        segments,
        start=1,
    ):
        segment_without_ids = segment

        for evidence_id in evidence_by_id:
            segment_without_ids = segment_without_ids.replace(
                evidence_id,
                "",
            )

        mentioned = _mentioned_tickers(segment_without_ids)

        if not mentioned:
            continue

        inline_ids = [evidence_id for evidence_id in evidence_by_id if evidence_id in segment]

        if not inline_ids:
            failures.append(f"segment_{index}:company_claim_without_inline_evidence")
            continue

        covered: set[str] = set()

        for evidence_id in inline_ids:
            if evidence_id not in cited_ids:
                failures.append(f"segment_{index}:{evidence_id}:inline_id_not_declared")
                continue

            item = evidence_by_id[evidence_id]

            if not is_usable_evidence(item):
                failures.append(f"segment_{index}:{evidence_id}:unusable_evidence")
                continue

            supported = _evidence_tickers(item)

            overlap = supported.intersection(mentioned)

            if not overlap:
                failures.append(f"segment_{index}:{evidence_id}:company_mismatch")
                continue

            covered.update(overlap)

        missing = mentioned - covered

        if missing:
            failures.append(
                f"segment_{index}:missing_company_evidence:" + ",".join(sorted(missing))
            )

    return CompanyCitationGroundingResult(
        passed=not failures,
        failure_reasons=tuple(failures),
    )
