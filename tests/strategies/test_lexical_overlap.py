"""
test_lexical_overlap.py

Covers grounded vs. ungrounded responses, the degenerate too-short-response
case, threshold sensitivity, constructor validation, and the score/predict
contract.

Author: Marco Pérez Padilla
Date:   04-10-2026
"""

from __future__ import annotations

import pytest

from src.strategies.lexical_overlap import LexicalOverlapDetector


def test_grounded_response_is_not_flagged() -> None:
    detector = LexicalOverlapDetector()
    context = "The Eiffel Tower is located in Paris and was completed in 1889."
    response = "The Eiffel Tower is located in Paris."
    assert detector.predict(context, response) is False


def test_ungrounded_response_is_flagged() -> None:
    detector = LexicalOverlapDetector()
    context = "The Eiffel Tower is located in Paris and was completed in 1889."
    response = "Kangaroos are native to Australia and hop very fast."
    assert detector.predict(context, response) is True


def test_response_too_short_for_n_gram_is_flagged() -> None:
    detector = LexicalOverlapDetector(n_gram_size=3)
    assert detector.predict("Some context here.", "Hi") is True


def test_predict_batch_returns_scores_between_zero_and_one() -> None:
    detector = LexicalOverlapDetector()
    _, scores = detector.predict_batch(
        ["context one", "context two"], ["response one", "totally unrelated text"]
    )
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_stricter_threshold_flags_more_responses() -> None:
    context = "the quick brown fox jumps over the lazy dog"
    response = "the quick brown fox runs"

    lenient_detector = LexicalOverlapDetector(overlap_threshold=0.05)
    strict_detector = LexicalOverlapDetector(overlap_threshold=0.9)

    lenient_result = lenient_detector.predict(context, response)
    strict_result = strict_detector.predict(context, response)
    assert strict_result or not lenient_result


@pytest.mark.parametrize("invalid_n_gram_size", [0, -1])
def test_constructor_rejects_invalid_n_gram_size(invalid_n_gram_size: int) -> None:
    with pytest.raises(ValueError, match="n_gram_size"):
        LexicalOverlapDetector(n_gram_size=invalid_n_gram_size)


@pytest.mark.parametrize("invalid_threshold", [-0.1, 1.1])
def test_constructor_rejects_invalid_threshold(invalid_threshold: float) -> None:
    with pytest.raises(ValueError, match="overlap_threshold"):
        LexicalOverlapDetector(overlap_threshold=invalid_threshold)


def test_score_is_consistent_with_predict_batch_scores() -> None:
    detector = LexicalOverlapDetector()
    context, response = "the quick brown fox", "the quick brown fox jumps"
    _labels, batch_scores = detector.predict_batch([context], [response])
    assert detector.score(context, response) == pytest.approx(batch_scores[0])