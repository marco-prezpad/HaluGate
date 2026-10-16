"""
test_response_length.py

Covers the monotone scoring, the bounds, whitespace handling, independence
from the context, and the inherited batch validation.

Author: Marco Pérez Padilla
Date:   14-10-2026
"""

from __future__ import annotations

import pytest

from src.errors import EmptyBatchError
from src.strategies.response_length import ResponseLengthDetector


def test_longer_responses_get_higher_scores() -> None:
    _, scores = ResponseLengthDetector().predict_batch(
        ["c", "c", "c"], ["one", "one two three", " ".join(["w"] * 50)]
    )

    assert scores[0] < scores[1] < scores[2]


def test_scores_stay_in_the_unit_interval() -> None:
    _, scores = ResponseLengthDetector().predict_batch(
        ["c", "c"], ["", " ".join(["w"] * 5000)]
    )

    assert all(0.0 <= score < 1.0 for score in scores)


def test_an_empty_response_scores_zero() -> None:
    assert ResponseLengthDetector().score("context", "   ") == 0.0


def test_the_score_is_n_over_n_plus_one() -> None:
    assert ResponseLengthDetector().score("c", "a b c") == pytest.approx(3 / 4)


def test_repeated_whitespace_is_not_counted_as_words() -> None:
    detector = ResponseLengthDetector()

    assert detector.score("c", "a    b\n\nc") == detector.score("c", "a b c")


def test_the_context_is_ignored() -> None:
    detector = ResponseLengthDetector()

    assert detector.score("short", "a b c") == detector.score("a very long context " * 50, "a b c")


def test_batch_validation_is_inherited() -> None:
    with pytest.raises(EmptyBatchError):
        ResponseLengthDetector().predict_batch([], [])
    with pytest.raises(ValueError, match="equal length"):
        ResponseLengthDetector().predict_batch(["a"], ["b", "c"])