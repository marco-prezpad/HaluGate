"""
factory.py

Instantiates the correct HallucinationDetector strategy from a name string,
so evaluate.py never needs an if/elif chain per architecture: adding a new
detector only requires adding one entry to the registry and one file under
src/strategies/.

The two DistilBERT arms used on RAGTruth are the same class with a
different default checkpoint: "distilbert_transfer" is the model trained on
HaluEval, "distilbert_indomain" the one fine-tuned on RAGTruth. "distilbert"
is kept for the HaluEval runs.

Author: Marco Pérez Padilla
Date:   12-10-2026
"""

from __future__ import annotations

from typing import Any, Callable

from src.errors import UnknownStrategyError
from src.strategies.base import HallucinationDetector
from src.strategies.distilbert_classifier import DistilBERTDetector
from src.strategies.lexical_overlap import LexicalOverlapDetector
from src.strategies.nli_detector import NLIDetector
from src.strategies.response_length import ResponseLengthDetector

HALUEVAL_DISTILBERT_PATH = "models/distilbert-halu"
RAGTRUTH_DISTILBERT_PATH = "models/distilbert-ragtruth"


def _distilbert_builder(
    default_model_path: str,
) -> Callable[[dict[str, Any]], HallucinationDetector]:
    """Builder for a DistilBERT detector whose checkpoint defaults to a given path."""
    return lambda params: DistilBERTDetector(
        model_path=params.pop("model_path", default_model_path), **params
    )


_STRATEGY_REGISTRY: dict[str, Callable[[dict[str, Any]], HallucinationDetector]] = {
    "lexical": lambda params: LexicalOverlapDetector(**params),
    "distilbert": _distilbert_builder(HALUEVAL_DISTILBERT_PATH),
    "distilbert_transfer": _distilbert_builder(HALUEVAL_DISTILBERT_PATH),
    "distilbert_indomain": _distilbert_builder(RAGTRUTH_DISTILBERT_PATH),
    "nli": lambda params: NLIDetector(**params),
    "length": lambda params: ResponseLengthDetector(**params),
}


def get_detector(name: str, params: dict[str, Any] | None = None) -> HallucinationDetector:
    """Instantiate the detector registered under ``name``.

    Raises:
        UnknownStrategyError: if ``name`` is not in the registry.
    """
    if name not in _STRATEGY_REGISTRY:
        raise UnknownStrategyError(name, list(_STRATEGY_REGISTRY.keys()))

    resolved_params = dict(params) if params else {}
    return _STRATEGY_REGISTRY[name](resolved_params)


def available_strategies() -> list[str]:
    """List every strategy name the factory can currently build."""
    return list(_STRATEGY_REGISTRY.keys())