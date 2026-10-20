"""
test_factory.py

Verifies the factory instantiates the correct concrete class per name,
forwards constructor parameters correctly, and raises on unknown names.

Author: Marco Pérez Padilla
Date:   12-10-2026
"""

from __future__ import annotations

import pytest

from src.errors import UnknownStrategyError
from src.factory import available_strategies, get_detector
from src.strategies.distilbert_classifier import DistilBERTDetector
from src.strategies.lexical_overlap import LexicalOverlapDetector
from src.strategies.nli_detector import NLIDetector


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


def test_distilbert_arms_default_to_their_own_checkpoints() -> None:
    assert get_detector("distilbert_transfer").model_path == "models/distilbert-halu"
    assert get_detector("distilbert_indomain").model_path == "models/distilbert-ragtruth"


def test_distilbert_arms_accept_a_custom_checkpoint_and_parameters() -> None:
    detector = get_detector("distilbert_indomain", {"model_path": "elsewhere", "batch_size": 2})
    assert detector.model_path == "elsewhere"
    assert detector.batch_size == 2


def test_get_detector_returns_nli_detector() -> None:
    detector = get_detector("nli", {"batch_size": 4})
    assert isinstance(detector, NLIDetector)
    assert detector.batch_size == 4


def test_get_detector_with_no_params_uses_defaults() -> None:
    detector = get_detector("lexical")
    assert isinstance(detector, LexicalOverlapDetector)


def test_get_detector_raises_on_unknown_name() -> None:
    with pytest.raises(UnknownStrategyError):
        get_detector("made_up_strategy")


def test_available_strategies_lists_all_registered() -> None:
    assert set(available_strategies()) == {
        "lexical",
        "distilbert",
        "distilbert_transfer",
        "distilbert_indomain",
        "nli",
        "nli_chunked",
        "length",
    }


def test_get_detector_returns_the_response_length_baseline() -> None:
    from src.strategies.response_length import ResponseLengthDetector

    assert isinstance(get_detector("length"), ResponseLengthDetector)