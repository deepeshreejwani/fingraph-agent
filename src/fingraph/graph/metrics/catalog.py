"""Canonical financial metric definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDefinition:
    """Canonical metric mapped to allowed XBRL concepts."""

    name: str
    concepts: tuple[str, ...]
    unit: str | None = None


METRICS: dict[str, MetricDefinition] = {
    "revenue": MetricDefinition(
        name="revenue",
        concepts=(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
        ),
        unit="USD",
    ),
    "net_income": MetricDefinition(
        name="net_income",
        concepts=(
            "NetIncomeLoss",
            "ProfitLoss",
        ),
        unit="USD",
    ),
    "operating_income": MetricDefinition(
        name="operating_income",
        concepts=("OperatingIncomeLoss",),
        unit="USD",
    ),
    "total_assets": MetricDefinition(
        name="total_assets",
        concepts=("Assets",),
        unit="USD",
    ),
    "total_liabilities": MetricDefinition(
        name="total_liabilities",
        concepts=("Liabilities",),
        unit="USD",
    ),
    "cash": MetricDefinition(
        name="cash",
        concepts=(
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ),
        unit="USD",
    ),
}


def get_metric(
    name: str,
) -> MetricDefinition:
    """Return a known canonical metric."""

    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")

    try:
        return METRICS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(METRICS))

        raise ValueError(f"Unknown metric '{name}'. Supported metrics: {supported}") from exc
