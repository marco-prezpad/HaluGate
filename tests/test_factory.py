"""
test_factory.py

Verifies the factory instantiates the correct concrete class per name,
forwards constructor parameters correctly, and raises on unknown names.

Author: Marco Pérez Padilla
Date:   06-10-2026
"""

from __future__ import annotations

import pytest

from src.errors import UnknownStrategyError
from src.factory import available_strategies, get_detector
from src.strategies.distilbert_classifier import DistilBERTDetector
from src.strategies.lexical_overlap import LexicalOverlapDetector


def test_get_detector_returns_lexical_overlap_detector() -> None:
    detector = get_detector("lexical", {"n_gram_size": 2, "overlap_threshold": 0.3})
    assert isinstance(detector, LexicalOverlapDetector)


def test_get_detector_returns_distilbert_detector_with_default_model_path() -> None:
    detector = get_detector("distilbert", {})
    assert isinstance(detector, DistilBERTDetector)
    assert detector.model_path == "models/distilbert-halu"


def test_get_detector_returns_distilbert_detector_with_custom_model_path() -> None:
    detector = get_detector("distilbert", {"model_path": "custom/path", "batch_size": 4})
    assert isinstance(detector, DistilBERTDetector)
    assert detector.model_path == "custom/path"
    assert detector.batch_size == 4


def test_get_detector_with_no_params_uses_defaults() -> None:
    detector = get_detector("lexical")
    assert isinstance(detector, LexicalOverlapDetector)


def test_get_detector_raises_on_unknown_name() -> None:
    with pytest.raises(UnknownStrategyError):
        get_detector("made_up_strategy")


def test_available_strategies_lists_lexical_and_distilbert() -> None:
    assert set(available_strategies()) == {"lexical", "distilbert"}