"""SEC XBRL company facts acquisition."""

from collections.abc import Mapping
from typing import Any

from fingraph.ingestion.sec.client import SecClient

COMPANY_FACTS_BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts"


def company_facts_url(cik: str) -> str:
    """Build the SEC company facts URL."""

    return f"{COMPANY_FACTS_BASE_URL}/CIK{cik}.json"


def fetch_company_facts(
    client: SecClient,
    cik: str,
) -> Mapping[str, Any]:
    """Fetch raw SEC XBRL company facts."""

    return client.get_json(company_facts_url(cik))
