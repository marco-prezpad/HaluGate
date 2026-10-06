"""
factory.py

Instantiates the correct HallucinationDetector strategy from a name string,
so evaluate.py never needs an if/elif chain per architecture: adding a new
detector only requires adding one entry to the registry and one file under
src/strategies/.

Author: Marco Pérez Padilla
Date:   06-10-2026
"""

from __future__ import annotations

from typing import Any, Callable

from src.errors import UnknownStrategyError
from src.strategies.base import HallucinationDetector
from src.strategies.distilbert_classifier import DistilBERTDetector
from src.strategies.lexical_overlap import LexicalOverlapDetector

_STRATEGY_REGISTRY: dict[str, Callable[[dict[str, Any]], HallucinationDetector]] = {
    "lexical": lambda params: LexicalOverlapDetector(**params),
    "distilbert": lambda params: DistilBERTDetector(
        model_path=params.pop("model_path", "models/distilbert-halu"),
        **params,
    ),
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