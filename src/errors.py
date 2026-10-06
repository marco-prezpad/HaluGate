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

class EmptyBatchError(HaluGateError):
    """Raised when a detector receives an empty batch of samples to score."""

    def __init__(self, strategy_name: str) -> None:
        self.strategy_name = strategy_name
        super().__init__(f"Strategy '{strategy_name}' received an empty batch to predict on.")

class CheckpointCorruptedError(HaluGateError):
    """Raised when a checkpoint file exists but cannot be safely resumed from."""

    def __init__(self, checkpoint_path: str, reason: str) -> None:
        self.checkpoint_path = checkpoint_path
        self.reason = reason
        super().__init__(
            f"Checkpoint at '{checkpoint_path}' is corrupted or incompatible: {reason}. "
            "Delete it manually to restart this strategy from scratch."
        )

class ConfigValidationError(HaluGateError):
    """Raised when config.yaml is missing required fields or has invalid values."""

    def __init__(self, field_name: str, reason: str) -> None:
        self.field_name = field_name
        self.reason = reason
        super().__init__(f"Invalid config field '{field_name}': {reason}")


class CalibrationError(HaluGateError):
    """Raised when a decision threshold cannot be meaningfully calibrated.

    Typically means the validation set contains only one class, so F1 (or
    any other threshold-dependent metric) is undefined/meaningless to
    optimize against it.
    """


class ModelLoadError(HaluGateError):
    """Raised when a detector's underlying model fails to load."""

    def __init__(self, model_identifier: str, cause: Exception) -> None:
        self.model_identifier = model_identifier
        self.cause = cause
        super().__init__(
            f"Failed to load model '{model_identifier}': {cause!r}"
        )