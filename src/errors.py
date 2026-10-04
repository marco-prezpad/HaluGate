"""
errors.py

Custom exception hierarchy for HaluGate. Centralizing errors here avoids
overloading built-in exceptions (ValueError, KeyError, ...) with
project-specific semantics, and lets calling code distinguish between
recoverable and non-recoverable failures.

Author: Marco Pérez Padilla
Date:   04-10-2026
"""

from __future__ import annotations


class HaluGateError(Exception):
    """Base class for all HaluGate-specific exceptions."""


class UnknownStrategyError(HaluGateError):
    """Raised when the factory receives a strategy name not in the registry."""

    def __init__(self, strategy_name: str, available_strategies: list[str]) -> None:
        self.strategy_name = strategy_name
        self.available_strategies = available_strategies
        message = (
            f"Unknown detector strategy '{strategy_name}'. "
            f"Available strategies: {sorted(available_strategies)}"
        )
        super().__init__(message)


class DatasetFormatError(HaluGateError):
    """Raised when a raw dataset file does not match the expected schema."""

    def __init__(self, dataset_name: str, reason: str) -> None:
        self.dataset_name = dataset_name
        self.reason = reason
        super().__init__(f"Malformed dataset '{dataset_name}': {reason}")


class DatasetNotFoundError(HaluGateError):
    """Raised when a dataset file is missing from disk."""

    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path
        super().__init__(
            f"Dataset file not found at '{dataset_path}'. "
            "Run the corresponding download/preprocessing step first."
        )