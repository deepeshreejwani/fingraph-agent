"""Deterministic extraction of important 10-K sections."""

import re
from dataclasses import dataclass

from fingraph.domain.models import Section
from fingraph.utils.ids import stable_id

MIN_SECTION_CHARS = 1_000


@dataclass(frozen=True)
class SectionDefinition:
    """Definition of one target 10-K section."""

    name: str
    start_pattern: re.Pattern[str]
    end_patterns: tuple[re.Pattern[str], ...]
    max_candidate_chars: int
    fallback_start_pattern: re.Pattern[str] | None = None
    fallback_end_patterns: tuple[re.Pattern[str], ...] = ()


def _start_item_pattern(
    item: str,
    title_pattern: str = "",
) -> re.Pattern[str]:
    """
    Match a target section heading at the start of a line.

    SEC filings commonly render headings as either one line:

        Item 1. Business:

    or across adjacent lines:

        Item 1.
        Business:

    Optional punctuation after both the item marker and title is accepted.
    """

    item_marker = (
        rf"ITEM\s*"
        rf"{re.escape(item)}"
        rf"(?![A-Z0-9])"
        rf"\s*[.\-:–—]?"
    )

    trailing_punctuation = r"[.\-:–—]?"

    if title_pattern:
        pattern = (
            rf"^\s*{item_marker}"
            rf"(?:\s*{title_pattern})?"
            rf"\s*{trailing_punctuation}"
            rf"\s*$"
        )
    else:
        pattern = (
            rf"^\s*{item_marker}"
            rf"\s*{trailing_punctuation}"
            rf"\s*$"
        )

    return re.compile(
        pattern,
        re.IGNORECASE | re.MULTILINE,
    )


def _end_item_pattern(
    item: str,
) -> re.Pattern[str]:
    """Match a following Item heading while rejecting inline references."""

    item_marker = (
        rf"ITEM\s*"
        rf"{re.escape(item)}"
        rf"(?![A-Z0-9])"
        rf"\s*[.\-:–—]?"
    )

    return re.compile(
        (
            rf"^\s*{item_marker}"
            rf"(?!\s+of\s+Part\b)"
            rf"(?:\s+[^\n]{{0,100}})?"
            rf"\s*$"
        ),
        re.IGNORECASE | re.MULTILINE,
    )


def _heading_pattern(
    title_pattern: str,
) -> re.Pattern[str]:
    """
    Match a standalone semantic filing heading.

    This supports filing layouts that organize narrative content by descriptive
    headings and provide the formal SEC Item mapping separately in a
    cross-reference index.
    """

    return re.compile(
        (
            rf"^\s*"
            rf"{title_pattern}"
            rf"\s*[.\-:–—]?"
            rf"\s*$"
        ),
        re.IGNORECASE | re.MULTILINE,
    )


MARKET_RISK_FALLBACK_END_PATTERNS = (
    _heading_pattern(
        r"CRITICAL\s+ACCOUNTING\s+ESTIMATES",
    ),
    _heading_pattern(
        r"CRITICAL\s+ACCOUNTING\s+POLICIES"
        r"(?:\s+AND\s+ESTIMATES)?",
    ),
    _heading_pattern(
        r"CRITICAL\s+ACCOUNTING\s+POLICIES"
        r"\s+AND\s+ESTIMATES",
    ),
    _heading_pattern(
        r"RISK\s+FACTORS",
    ),
    _heading_pattern(
        r"FINANCIAL\s+STATEMENTS"
        r"(?:\s+AND\s+SUPPLEMENTARY\s+DATA)?",
    ),
    _end_item_pattern("8"),
)


SECTION_DEFINITIONS = (
    SectionDefinition(
        name="Item 1. Business",
        start_pattern=_start_item_pattern(
            "1",
            r"(?:BUSINESS)?",
        ),
        end_patterns=(_end_item_pattern("1A"),),
        max_candidate_chars=200_000,
        fallback_start_pattern=_heading_pattern(
            r"OUR\s+BUSINESS",
        ),
        fallback_end_patterns=(
            _heading_pattern(
                r"MANAGEMENT(?:['’]S)?\s+DISCUSSION\s+AND\s+ANALYSIS",
            ),
        ),
    ),
    SectionDefinition(
        name="Item 1A. Risk Factors",
        start_pattern=_start_item_pattern(
            "1A",
            r"(?:RISK\s+FACTORS)?",
        ),
        end_patterns=(
            _end_item_pattern("1B"),
            _end_item_pattern("1C"),
            _end_item_pattern("2"),
        ),
        max_candidate_chars=250_000,
        fallback_start_pattern=_heading_pattern(
            r"RISK\s+FACTORS",
        ),
        fallback_end_patterns=(
            _heading_pattern(
                r"UNRESOLVED\s+STAFF\s+COMMENTS",
            ),
            _heading_pattern(
                r"CYBERSECURITY",
            ),
            _heading_pattern(
                r"PROPERTIES",
            ),
        ),
    ),
    SectionDefinition(
        name="Item 7. MD&A",
        start_pattern=_start_item_pattern(
            "7",
            (
                r"(?:MANAGEMENT(?:['’]S)?\s+"
                r"DISCUSSION(?:\s+AND\s+ANALYSIS)?"
                r"(?:\s+OF\s+FINANCIAL\s+CONDITION"
                r"\s+AND\s+RESULTS\s+OF\s+OPERATIONS)?)?"
            ),
        ),
        end_patterns=(
            _end_item_pattern("7A"),
            _end_item_pattern("8"),
        ),
        max_candidate_chars=250_000,
        fallback_start_pattern=_heading_pattern(
            r"MANAGEMENT(?:['’]S)?\s+DISCUSSION\s+AND\s+ANALYSIS",
        ),
        fallback_end_patterns=(
            _heading_pattern(
                r"RISK\s+FACTORS",
            ),
            _heading_pattern(
                r"FINANCIAL\s+STATEMENTS"
                r"(?:\s+AND\s+SUPPLEMENTARY\s+DATA)?",
            ),
        ),
    ),
    SectionDefinition(
        name="Item 7A. Market Risk",
        start_pattern=_start_item_pattern(
            "7A",
            (
                r"(?:QUANTITATIVE\s+AND\s+QUALITATIVE"
                r"(?:\s+DISCLOSURES)?"
                r"(?:\s+ABOUT\s+MARKET\s+RISK)?)?"
            ),
        ),
        end_patterns=(
            _end_item_pattern("8"),
            _heading_pattern(
                r"CRITICAL\s+ACCOUNTING\s+ESTIMATES",
            ),
            _heading_pattern(
                r"CRITICAL\s+ACCOUNTING\s+POLICIES"
                r"(?:\s+AND\s+ESTIMATES)?",
            ),
            _heading_pattern(
                r"RISK\s+FACTORS",
            ),
            _heading_pattern(
                r"FINANCIAL\s+STATEMENTS"
                r"(?:\s+AND\s+SUPPLEMENTARY\s+DATA)?",
            ),
        ),
        max_candidate_chars=75_000,
        fallback_start_pattern=_heading_pattern(
            (
                r"QUANTITATIVE\s+AND\s+QUALITATIVE"
                r"\s+DISCLOSURES\s+ABOUT\s+MARKET\s+RISK"
            ),
        ),
        fallback_end_patterns=MARKET_RISK_FALLBACK_END_PATTERNS,
    ),
)


def extract_sections(
    text: str,
    *,
    filing_id: str,
) -> list[Section]:
    """
    Extract target filing sections using deterministic heading boundaries.

    Canonical SEC Item headings are always preferred. Semantic-heading
    fallback is used only when the canonical extraction yields no valid
    narrative candidate for that section.
    """

    sections: list[Section] = []

    for definition in SECTION_DEFINITIONS:
        extracted = _extract_best_candidate(
            text,
            start_pattern=definition.start_pattern,
            end_patterns=definition.end_patterns,
            max_candidate_chars=definition.max_candidate_chars,
        )

        if (
            extracted is None
            and definition.fallback_start_pattern is not None
            and definition.fallback_end_patterns
        ):
            extracted = _extract_best_candidate(
                text,
                start_pattern=definition.fallback_start_pattern,
                end_patterns=definition.fallback_end_patterns,
                max_candidate_chars=definition.max_candidate_chars,
            )

        if extracted is None:
            continue

        section_id = stable_id(
            "section",
            filing_id,
            definition.name,
        )

        sections.append(
            Section(
                section_id=section_id,
                filing_id=filing_id,
                section_name=definition.name,
                text=extracted,
            )
        )

    return sections


def _extract_best_candidate(
    text: str,
    *,
    start_pattern: re.Pattern[str],
    end_patterns: tuple[re.Pattern[str], ...],
    max_candidate_chars: int,
) -> str | None:
    """
    Extract the longest plausible candidate between deterministic boundaries.

    Short candidates are rejected because filing tables of contents and
    cross-reference indexes commonly repeat headings without carrying the
    underlying narrative.

    Implausibly large candidates are also rejected. They usually indicate that
    the intended following boundary was absent or expressed using a filing
    layout variant, which is safer to treat as extraction failure than to index
    unrelated downstream filing content.
    """

    starts = list(start_pattern.finditer(text))

    if not starts:
        return None

    candidates: list[str] = []

    for start_match in starts:
        end = _find_nearest_end(
            text,
            start_match.end(),
            end_patterns,
        )

        if end is None:
            continue

        candidate = text[start_match.start() : end].strip()

        candidate_length = len(candidate)

        if candidate_length < MIN_SECTION_CHARS:
            continue

        if candidate_length > max_candidate_chars:
            continue

        candidates.append(candidate)

    if not candidates:
        return None

    return max(
        candidates,
        key=len,
    )


def _find_nearest_end(
    text: str,
    start: int,
    patterns: tuple[
        re.Pattern[str],
        ...,
    ],
) -> int | None:
    """Return the nearest valid following boundary."""

    matches = [
        match.start()
        for pattern in patterns
        if (
            match := pattern.search(
                text,
                start,
            )
        )
        is not None
    ]

    if not matches:
        return None

    return min(matches)
