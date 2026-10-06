"""
test_calibration.py

Covers threshold search on separable and overlapping score distributions,
the single-class guard, and input validation.

Author: Marco Pérez Padilla
Date:   05-10-2026
"""

from __future__ import annotations

import pytest

from src.calibration import find_optimal_threshold
from src.errors import CalibrationError


def test_perfectly_separable_scores_achieve_f1_of_one() -> None:
    scores = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
    true_labels = [0, 0, 0, 1, 1, 1]
    result = find_optimal_threshold(scores, true_labels)
    assert result.f1_score == pytest.approx(1.0)
    assert 0.3 < result.threshold <= 0.8


def test_the_lexical_overlap_regression_case_is_handled() -> None:
    scores = [0.05, 0.10, 0.12, 0.40, 0.55, 0.60]
    true_labels = [0, 0, 0, 1, 1, 1]
    result = find_optimal_threshold(scores, true_labels)
    assert result.f1_score == pytest.approx(1.0)
    assert result.threshold <= 0.40


def test_overlapping_distributions_find_best_achievable_f1() -> None:
    scores = [0.2, 0.5, 0.5, 0.8]
    true_labels = [0, 0, 1, 1]
    result = find_optimal_threshold(scores, true_labels)
    assert 0.0 <= result.f1_score <= 1.0


def test_raises_on_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        find_optimal_threshold([0.1, 0.2], [1])


def test_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        find_optimal_threshold([], [])


def test_raises_calibration_error_on_single_class_labels() -> None:
    with pytest.raises(CalibrationError):
        find_optimal_threshold([0.1, 0.5, 0.9], [1, 1, 1])


def test_result_threshold_is_among_observed_scores() -> None:
    scores = [0.1, 0.4, 0.4, 0.9]
    true_labels = [0, 1, 0, 1]
    result = find_optimal_threshold(scores, true_labels)
    assert result.threshold in scores