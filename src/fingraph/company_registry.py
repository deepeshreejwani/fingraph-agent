"""Canonical supported-company registry and boundary-aware mention matching."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CompanyProfile:
    """One company supported by the serving corpus."""

    ticker: str
    display_name: str
    aliases: tuple[str, ...]


COMPANIES: tuple[CompanyProfile, ...] = (
    CompanyProfile("AAPL", "Apple", ("apple", "apple inc.", "aapl")),
    CompanyProfile("MSFT", "Microsoft", ("microsoft", "microsoft corporation", "msft")),
    CompanyProfile("GOOGL", "Alphabet", ("alphabet", "alphabet inc.", "google", "googl")),
    CompanyProfile("AMZN", "Amazon", ("amazon", "amazon.com", "amazon.com inc.", "amzn")),
    CompanyProfile("META", "Meta", ("meta", "meta platforms", "facebook")),
    CompanyProfile("NVDA", "NVIDIA", ("nvidia", "nvidia corporation", "nvda")),
    CompanyProfile("AMD", "Advanced Micro Devices", ("advanced micro devices", "amd")),
    CompanyProfile("INTC", "Intel", ("intel", "intel corporation", "intc")),
    CompanyProfile("AVGO", "Broadcom", ("broadcom", "broadcom inc.", "avgo")),
    CompanyProfile("QCOM", "Qualcomm", ("qualcomm", "qualcomm incorporated", "qcom")),
    CompanyProfile("CSCO", "Cisco", ("cisco", "cisco systems", "csco")),
    CompanyProfile(
        "IBM",
        "IBM",
        ("ibm", "international business machines"),
    ),
    CompanyProfile("ORCL", "Oracle", ("oracle", "oracle corporation", "orcl")),
    CompanyProfile("CRM", "Salesforce", ("salesforce", "salesforce inc.", "crm")),
    CompanyProfile("ADBE", "Adobe", ("adobe", "adobe inc.", "adbe")),
    CompanyProfile("NOW", "ServiceNow", ("servicenow", "service now")),
    CompanyProfile("PANW", "Palo Alto Networks", ("palo alto networks", "panw")),
    CompanyProfile("NFLX", "Netflix", ("netflix", "nflx")),
    CompanyProfile("TSLA", "Tesla", ("tesla", "tesla inc.", "tsla")),
    CompanyProfile("TXN", "Texas Instruments", ("texas instruments", "txn")),
    CompanyProfile("MU", "Micron Technology", ("micron", "micron technology")),
    CompanyProfile("AMAT", "Applied Materials", ("applied materials", "amat")),
    CompanyProfile("LRCX", "Lam Research", ("lam research", "lrcx")),
    CompanyProfile("KLAC", "KLA", ("kla", "kla corporation", "kla-tencor", "klac")),
    CompanyProfile("PLTR", "Palantir", ("palantir", "palantir technologies", "pltr")),
)


COMPANIES_BY_TICKER: dict[str, CompanyProfile] = {company.ticker: company for company in COMPANIES}


def _boundary_pattern(value: str) -> re.Pattern[str]:
    """Compile an ASCII alphanumeric-boundary matcher for one alias."""

    return re.compile(
        rf"(?<![a-z0-9]){re.escape(value.lower())}(?![a-z0-9])",
        re.IGNORECASE,
    )


_ALIAS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (company.ticker, _boundary_pattern(alias)) for company in COMPANIES for alias in company.aliases
)

_TICKER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (
        company.ticker,
        re.compile(rf"(?<![A-Za-z0-9]){re.escape(company.ticker)}(?![A-Za-z0-9])"),
    )
    for company in COMPANIES
)


def extract_company_tickers(text: str) -> tuple[str, ...]:
    """Return supported companies explicitly mentioned in text."""

    found: set[str] = set()

    for ticker, pattern in _TICKER_PATTERNS:
        if pattern.search(text):
            found.add(ticker)

    for ticker, pattern in _ALIAS_PATTERNS:
        if pattern.search(text):
            found.add(ticker)

    return tuple(sorted(found))


def normalized_company_aliases() -> frozenset[str]:
    """Return normalized aliases/tickers accepted by the supported universe."""

    aliases = {
        re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip()
        for company in COMPANIES
        for alias in (*company.aliases, company.ticker)
    }

    return frozenset(alias for alias in aliases if alias)


def canonical_company_tickers() -> dict[str, tuple[str, ...]]:
    """Return the schema-facing ticker-to-alias mapping."""

    return {company.ticker: company.aliases for company in COMPANIES}


def company_schema_lines() -> str:
    """Render the controlled company universe for the Text-to-Cypher prompt."""

    return "\n".join(f"- {company.display_name} -> {company.ticker}" for company in COMPANIES)
