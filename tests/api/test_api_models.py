"""Focused API schema regression tests."""

import pytest
from pydantic import ValidationError

from fingraph.api.models import (
    AskRequest,
)


def test_question_is_trimmed() -> None:
    request = AskRequest(question=("  What was Microsoft revenue?  "))

    assert request.question == "What was Microsoft revenue?"


@pytest.mark.parametrize(
    "question",
    [
        "",
        " ",
        "  ",
        "a",
        "ab",
        " a ",
    ],
)
def test_short_question_is_rejected(
    question: str,
) -> None:
    with pytest.raises(ValidationError):
        AskRequest(question=question)


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        AskRequest.model_validate(
            {
                "question": ("What was Microsoft revenue?"),
                "unexpected": True,
            }
        )
