"""
calibration.py

Finds the decision threshold on a detector's continuous score that
maximizes F1 against a validation set, instead of trusting a hardcoded
value (like 0.5, or a fixed overlap_threshold) to be meaningful for every
strategy and every configuration of that strategy.

This exists because of a concrete, measured failure: the lexical-overlap
detector's overlap_threshold=0.15 gave a reasonable decision boundary under
n_gram_size=3, but under n_gram_size=1 the same threshold value meant
something entirely different on the differently-scaled score distribution
(recall collapsed to ~0.03 despite AUC improving to ~0.91). Calibrating the
threshold against data, per (dataset, strategy) pair, removes that failure
mode -- and it generalizes: DistilBERT and NLI will have their own score
distributions too, with no guarantee that 0.5 is the right cut point either.

Author: Marco Pérez Padilla
Date:   05-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass

from src.errors import CalibrationError


@dataclass(frozen=True)
class CalibrationResult:
    """The chosen threshold and the F1 it achieved on the calibration set."""

    threshold: float
    f1_score: float


def find_optimal_threshold(scores: list[float], true_labels: list[int]) -> CalibrationResult:
    """Search every observed score as a candidate threshold and return the
    one that maximizes F1 against ``true_labels``.

    A sample is predicted positive (hallucination) when its score is
    greater than or equal to the candidate threshold.

    Raises:
        ValueError: if ``scores``/``true_labels`` are empty or of
            mismatched length.
        CalibrationError: if ``true_labels`` contains only one class (F1 is
            undefined/meaningless to optimize in that case).
    """
    if len(scores) != len(true_labels):
        raise ValueError(
            f"scores and true_labels must have equal length: "
            f"{len(scores)} != {len(true_labels)}"
        )
    if len(scores) == 0:
        raise ValueError("Cannot calibrate a threshold on an empty validation set.")
    if len(set(true_labels)) < 2:
        raise CalibrationError(
            "Validation set contains only one class; cannot calibrate a "
            "meaningful decision threshold against it."
        )

    from sklearn.metrics import f1_score

    candidate_thresholds = sorted(set(scores))
    best_threshold = candidate_thresholds[0]
    best_f1 = -1.0

    for threshold in candidate_thresholds:
        predicted_labels = [int(score >= threshold) for score in scores]
        current_f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return CalibrationResult(threshold=float(best_threshold), f1_score=float(best_f1))