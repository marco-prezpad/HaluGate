"""
test_base.py

Verifies the shared validation logic (empty-batch, mismatched-length) and
the predict()/score() -> predict_batch() delegation, using a minimal
concrete subclass since HallucinationDetector is abstract.

Author: Marco Pérez Padilla
Date:   04-10-2026
"""

from __future__ import annotations

import pytest

from src.errors import EmptyBatchError
from src.strategies.base import HallucinationDetector


class _AlwaysHallucinationDetector(HallucinationDetector):
    name = "always_hallucination"

    def predict_batch(self, contexts, responses):
        self._validate_batch(contexts, responses)
        return [True] * len(contexts), [1.0] * len(contexts)


def test_predict_delegates_to_predict_batch() -> None:
    detector = _AlwaysHallucinationDetector()
    assert detector.predict("context", "response") is True


def test_score_delegates_to_predict_batch() -> None:
    detector = _AlwaysHallucinationDetector()
    assert detector.score("context", "response") == 1.0


def test_predict_batch_raises_on_empty_input() -> None:
    detector = _AlwaysHallucinationDetector()
    with pytest.raises(EmptyBatchError):
        detector.predict_batch([], [])


def test_predict_batch_raises_on_mismatched_lengths() -> None:
    detector = _AlwaysHallucinationDetector()
    with pytest.raises(ValueError, match="equal length"):
        detector.predict_batch(["a", "b"], ["only one"])


def test_cannot_instantiate_abstract_base_directly() -> None:
    with pytest.raises(TypeError):
        HallucinationDetector()  # type: ignore[abstract]