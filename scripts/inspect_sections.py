"""Inspect extracted sections for locally cached SEC filings."""

import argparse
from pathlib import Path

from fingraph.ingestion.parsing import (
    extract_sections,
    html_to_text,
)

RAW_ROOT = Path("data/raw/sec")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ticker",
        required=True,
    )

    args = parser.parse_args()

    ticker = args.ticker.upper()

    filing_dir = RAW_ROOT / ticker / "filings"

    if not filing_dir.exists():
        raise SystemExit(f"No cached filings found for {ticker}")

    for filing_path in sorted(filing_dir.glob("*.html")):
        html = filing_path.read_text(
            encoding="utf-8",
        )

        text = html_to_text(html)

        sections = extract_sections(
            text,
            filing_id=filing_path.stem,
        )

        print()
        print("=" * 80)
        print(filing_path.name)
        print("=" * 80)

        for section in sections:
            print(f"{section.section_name:<30} {len(section.text):>10,} chars")

        print(f"TOTAL: {len(sections)} sections")


if __name__ == "__main__":
    main()
