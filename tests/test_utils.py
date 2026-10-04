"""
test_utils.py

Covers metric computation edge cases, the checkpoint manager's
save/load/corruption-handling behavior, and the length-sort helper.

Author: Marco Pérez Padilla
Date:   04-10-2026
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.errors import CheckpointCorruptedError
from src.utils import (
    CheckpointManager,
    CheckpointState,
    compute_classification_metrics,
    set_global_seed,
    sort_indices_by_text_length,
)


class TestComputeClassificationMetrics:
    def test_perfect_predictions_yield_all_ones(self) -> None:
        metrics = compute_classification_metrics([1, 0, 1, 0], [1, 0, 1, 0])
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.accuracy == 1.0

    def test_all_wrong_predictions_yield_zero_f1(self) -> None:
        metrics = compute_classification_metrics([1, 0, 1, 0], [0, 1, 0, 1])
        assert metrics.f1_score == 0.0
        assert metrics.accuracy == 0.0

    def test_auc_roc_is_none_without_scores(self) -> None:
        metrics = compute_classification_metrics([1, 0], [1, 0])
        assert metrics.auc_roc is None

    def test_auc_roc_is_none_when_ground_truth_is_single_class(self) -> None:
        metrics = compute_classification_metrics(
            [1, 1, 1], [1, 0, 1], predicted_scores=[0.9, 0.2, 0.8]
        )
        assert metrics.auc_roc is None

    def test_auc_roc_computed_when_both_classes_present(self) -> None:
        metrics = compute_classification_metrics(
            [1, 0, 1, 0], [1, 0, 1, 0], predicted_scores=[0.9, 0.1, 0.8, 0.2]
        )
        assert metrics.auc_roc == 1.0

    def test_raises_on_empty_labels(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_classification_metrics([], [])

    def test_raises_on_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_classification_metrics([1, 0], [1])

    def test_sample_count_matches_input_length(self) -> None:
        metrics = compute_classification_metrics([1, 0, 1], [1, 1, 1])
        assert metrics.sample_count == 3


class TestCheckpointManager:
    def test_load_returns_none_when_no_checkpoint_exists(self, tmp_path: Path) -> None:
        manager = CheckpointManager(tmp_path)
        assert manager.load("lexical", "halueval") is None

    def test_save_then_load_round_trips_state(self, tmp_path: Path) -> None:
        manager = CheckpointManager(tmp_path)
        state = CheckpointState(
            strategy_name="lexical",
            dataset_name="halueval",
            next_sample_index=42,
            predicted_labels=[1, 0, 1],
            predicted_scores=[0.9, 0.1, 0.7],
        )
        manager.save(state)
        assert manager.load("lexical", "halueval") == state

    def test_clear_removes_checkpoint_file(self, tmp_path: Path) -> None:
        manager = CheckpointManager(tmp_path)
        state = CheckpointState(strategy_name="lexical", dataset_name="halueval", next_sample_index=5)
        manager.save(state)
        manager.clear("lexical", "halueval")
        assert manager.load("lexical", "halueval") is None

    def test_clear_on_missing_checkpoint_does_not_raise(self, tmp_path: Path) -> None:
        manager = CheckpointManager(tmp_path)
        manager.clear("lexical", "halueval")

    def test_load_raises_on_invalid_json(self, tmp_path: Path) -> None:
        manager = CheckpointManager(tmp_path)
        checkpoint_path = tmp_path / "checkpoint_halueval__lexical.json"
        checkpoint_path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(CheckpointCorruptedError):
            manager.load("lexical", "halueval")

    def test_load_raises_on_schema_version_mismatch(self, tmp_path: Path) -> None:
        manager = CheckpointManager(tmp_path)
        checkpoint_path = tmp_path / "checkpoint_halueval__lexical.json"
        checkpoint_path.write_text(
            json.dumps({"schema_version": 999, "strategy_name": "lexical"}), encoding="utf-8"
        )
        with pytest.raises(CheckpointCorruptedError, match="schema version"):
            manager.load("lexical", "halueval")


class TestSortIndicesByTextLength:
    def test_sorts_shortest_to_longest(self) -> None:
        texts = ["a longer sentence here", "short", "medium length one"]
        assert sort_indices_by_text_length(texts) == [1, 2, 0]

    def test_empty_list_returns_empty_order(self) -> None:
        assert sort_indices_by_text_length([]) == []


def test_set_global_seed_does_not_raise() -> None:
    set_global_seed(123)