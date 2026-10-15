"""SEC EDGAR acquisition."""

from fingraph.ingestion.sec.client import (
    SecClient,
    SecClientError,
)
from fingraph.ingestion.sec.companies import (
    fetch_company_universe,
    select_companies,
)
from fingraph.ingestion.sec.filings import (
    discover_10k_filings,
    download_filing,
)
from fingraph.ingestion.sec.xbrl import (
    fetch_company_facts,
)

__all__ = [
    "SecClient",
    "SecClientError",
    "discover_10k_filings",
    "download_filing",
    "fetch_company_facts",
    "fetch_company_universe",
    "select_companies",
]
