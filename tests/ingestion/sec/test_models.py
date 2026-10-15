"""Regression tests for SEC transport models."""

from datetime import date

import pytest
from pydantic import ValidationError

from fingraph.ingestion.sec.models import (
    DownloadedFiling,
    FilingReference,
)
from fingraph.utils.hashing import sha256_text


def filing_reference() -> FilingReference:
    """Return a minimal valid filing reference."""

    return FilingReference(
        cik="0000320193",
        accession_number=("0000320193-25-000079"),
        form="10-K",
        filing_date=date(
            2025,
            10,
            31,
        ),
        report_date=date(
            2025,
            9,
            27,
        ),
        primary_document=("aapl-20250927.htm"),
        primary_document_url=("https://www.sec.gov/example.htm"),
    )


def test_downloaded_filing_preserves_trailing_newline() -> None:
    content = "<html><body>filing</body></html>\n"

    downloaded = DownloadedFiling(
        reference=filing_reference(),
        source_url=("https://www.sec.gov/example.htm"),
        content=content,
        content_hash=sha256_text(content),
    )

    assert downloaded.content == content

    assert downloaded.content.endswith("\n")

    assert downloaded.content_hash == sha256_text(downloaded.content)


def test_downloaded_filing_preserves_leading_and_trailing_whitespace() -> None:
    content = "\n  <html>filing</html>  \n"

    downloaded = DownloadedFiling(
        reference=filing_reference(),
        source_url=("https://www.sec.gov/example.htm"),
        content=content,
        content_hash=sha256_text(content),
    )

    assert downloaded.content == content


def test_downloaded_filing_rejects_mismatched_hash() -> None:
    content = "<html>filing</html>\n"

    with pytest.raises(
        ValidationError,
        match=("content hash does not match"),
    ):
        DownloadedFiling(
            reference=filing_reference(),
            source_url=("https://www.sec.gov/example.htm"),
            content=content,
            content_hash=("0" * 64),
        )
