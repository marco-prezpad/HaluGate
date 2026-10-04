"""
test_errors.py

Verifies every custom exception carries the attributes and message content
that calling code (and users reading a traceback) rely on.

Author: Marco Pérez Padilla
Date:   04-10-2026
"""

from __future__ import annotations

from src.errors import (
    DatasetFormatError,
    DatasetNotFoundError,
    HaluGateError,
    UnknownStrategyError,
    EmptyBatchError,
    CheckpointCorruptedError,
)


def test_unknown_strategy_error_lists_available_strategies() -> None:
    error = UnknownStrategyError("made_up", ["halueval"])
    assert error.strategy_name == "made_up"
    assert "halueval" in str(error)
    assert isinstance(error, HaluGateError)


def test_dataset_format_error_includes_dataset_name_and_reason() -> None:
    error = DatasetFormatError("halueval", "missing 'knowledge' field")
    assert "halueval" in str(error)
    assert "missing 'knowledge' field" in str(error)


def test_dataset_not_found_error_includes_path() -> None:
    error = DatasetNotFoundError("data/raw/missing.jsonl")
    assert "data/raw/missing.jsonl" in str(error)


def test_dataset_errors_inherit_from_halugate_error() -> None:
    assert isinstance(DatasetFormatError("x", "y"), HaluGateError)
    assert isinstance(DatasetNotFoundError("x"), HaluGateError)


def test_empty_batch_error_includes_strategy_name() -> None:
    error = EmptyBatchError("lexical")
    assert "lexical" in str(error)


def test_checkpoint_corrupted_error_includes_path_and_reason() -> None:
    error = CheckpointCorruptedError("/tmp/checkpoint.json", "invalid JSON")
    assert "/tmp/checkpoint.json" in str(error)
    assert "invalid JSON" in str(error)