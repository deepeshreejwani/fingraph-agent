"""SEC filing HTML normalization."""

import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

WHITESPACE_PATTERN = re.compile(r"[ \t\f\v]+")
BLANK_LINES_PATTERN = re.compile(r"\n{3,}")

warnings.filterwarnings(
    "ignore",
    category=XMLParsedAsHTMLWarning,
)


def html_to_text(html: str) -> str:
    """Convert SEC filing HTML/XHTML into normalized readable text."""

    soup = BeautifulSoup(
        html,
        "lxml",
    )

    for element in soup(
        [
            "script",
            "style",
            "noscript",
        ]
    ):
        element.decompose()

    text = soup.get_text(
        separator="\n",
    )

    lines: list[str] = []

    for raw_line in text.splitlines():
        line = WHITESPACE_PATTERN.sub(
            " ",
            raw_line,
        ).strip()

        if line:
            lines.append(line)

    normalized = "\n".join(lines)

    return BLANK_LINES_PATTERN.sub(
        "\n\n",
        normalized,
    ).strip()
