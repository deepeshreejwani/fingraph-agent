"""Deterministic baseline chunking for filing sections."""

from fingraph.domain.models import Chunk, Section
from fingraph.utils.hashing import sha256_text
from fingraph.utils.ids import stable_id


def chunk_section(
    section: Section,
    *,
    cik: str,
    ticker: str,
    fiscal_year: int | None,
    source_url: str,
    chunk_size_words: int = 500,
    overlap_words: int = 75,
) -> list[Chunk]:
    """Split a filing section into deterministic overlapping word chunks."""

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive.")

    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative.")

    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words.")

    words = section.text.split()

    if not words:
        return []

    step = chunk_size_words - overlap_words

    chunks: list[Chunk] = []

    for index, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + chunk_size_words]

        if not chunk_words:
            continue

        text = " ".join(chunk_words)

        content_hash = sha256_text(text)

        chunk_id = stable_id(
            "chunk",
            section.filing_id,
            section.section_id,
            index,
            content_hash,
        )

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                filing_id=section.filing_id,
                section_id=section.section_id,
                section_name=section.section_name,
                cik=cik,
                ticker=ticker,
                fiscal_year=fiscal_year,
                text=text,
                content_hash=content_hash,
                source_url=source_url,
            )
        )

        if start + chunk_size_words >= len(words):
            break

    return chunks
