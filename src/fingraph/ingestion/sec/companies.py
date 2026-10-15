"""SEC company universe acquisition."""

from collections.abc import Mapping

from fingraph.ingestion.sec.client import SecClient
from fingraph.ingestion.sec.models import CompanyReference

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def normalize_cik(value: int | str) -> str:
    """Normalize a CIK to the SEC 10-digit representation."""

    return str(value).strip().zfill(10)


def fetch_company_universe(
    client: SecClient,
) -> list[CompanyReference]:
    """Fetch the SEC ticker/company mapping."""

    payload = client.get_json(COMPANY_TICKERS_URL)

    companies: list[CompanyReference] = []

    for raw in payload.values():
        if not isinstance(raw, Mapping):
            continue

        cik = raw.get("cik_str")
        ticker = raw.get("ticker")
        name = raw.get("title")

        if cik is None or not ticker or not name:
            continue

        companies.append(
            CompanyReference(
                cik=normalize_cik(cik),
                ticker=str(ticker).upper(),
                name=str(name),
            )
        )

    return sorted(
        companies,
        key=lambda company: company.ticker,
    )


def select_companies(
    universe: list[CompanyReference],
    tickers: set[str],
) -> list[CompanyReference]:
    """Select requested tickers from the SEC universe."""

    normalized = {ticker.strip().upper() for ticker in tickers}

    selected = [company for company in universe if company.ticker in normalized]

    found = {company.ticker for company in selected}

    missing = sorted(normalized - found)

    if missing:
        raise ValueError(f"Tickers not found in SEC universe: {missing}")

    return selected
