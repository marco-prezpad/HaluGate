"""
test_evaluate.py

Covers config validation and the resumable-checkpointing evaluation loop:
a pre-existing checkpoint is honored (already-scored samples are not
re-scored) and progress is checkpointed incrementally.

Author: Marco Pérez Padilla
Date:   05-10-2026
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.errors import ConfigValidationError
from src.evaluate import (
    _run_strategy_with_checkpointing,
    _write_latency_records,
    _write_metrics_table,
    load_config,
)
from src.latency_bench import LatencyRecord
from src.strategies.base import HallucinationDetector
from src.utils import CheckpointManager, CheckpointState, configure_logging


class _CountingDetector(HallucinationDetector):
    name = "counting"
    scored_batches: list[list[str]] = []

    def predict_batch(self, contexts, responses):
        _CountingDetector.scored_batches.append(list(contexts))
        return [False] * len(contexts), [0.1] * len(contexts)


@pytest.fixture(autouse=True)
def _reset_counting_detector() -> None:
    _CountingDetector.scored_batches = []


def _patch_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.evaluate as evaluate_module
    monkeypatch.setattr(evaluate_module, "get_detector", lambda name, params: _CountingDetector())


class TestLoadConfig:
    def test_raises_when_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigValidationError, match="not found"):
            load_config(tmp_path / "missing.yaml")

    def test_raises_when_required_field_missing(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"datasets": ["halueval"]}), encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="strategies"):
            load_config(config_path)

    def test_loads_valid_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.dump({"datasets": ["halueval"], "strategies": ["lexical"],
                       "data_root": "data", "results_root": "results"}),
            encoding="utf-8",
        )
        config = load_config(config_path)
        assert config["datasets"] == ["halueval"]


class TestRunStrategyWithCheckpointing:
    def test_scores_all_samples_when_no_checkpoint_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_factory(monkeypatch)
        checkpoint_manager = CheckpointManager(tmp_path)
        logger = configure_logging()

        labels, scores = _run_strategy_with_checkpointing(
            strategy_name="counting", strategy_params={}, dataset_name="halueval",
            contexts=[f"c{i}" for i in range(10)], responses=[f"r{i}" for i in range(10)],
            checkpoint_manager=checkpoint_manager, checkpoint_every=3, logger=logger,
        )
        assert len(labels) == 10
        assert len(scores) == 10

    def test_resumes_from_existing_checkpoint_without_rescoring(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_factory(monkeypatch)
        checkpoint_manager = CheckpointManager(tmp_path)
        logger = configure_logging()

        checkpoint_manager.save(
            CheckpointState(
                strategy_name="counting", dataset_name="halueval", next_sample_index=6,
                predicted_labels=[0] * 6, predicted_scores=[0.1] * 6,
            )
        )

        labels, _scores = _run_strategy_with_checkpointing(
            strategy_name="counting", strategy_params={}, dataset_name="halueval",
            contexts=[f"c{i}" for i in range(10)], responses=[f"r{i}" for i in range(10)],
            checkpoint_manager=checkpoint_manager, checkpoint_every=3, logger=logger,
        )
        assert len(labels) == 10
        total_rescored = sum(len(batch) for batch in _CountingDetector.scored_batches)
        assert total_rescored == 4  # only the remaining 4 of 10


class TestWriteMetricsTable:
    def test_writes_header_and_rows(self, tmp_path: Path) -> None:
        output_path = tmp_path / "tables" / "metrics.csv"
        rows = [
            {"dataset": "halueval", "strategy": "lexical", "f1_score": 0.5},
        ]
        _write_metrics_table(rows, output_path)
        content = output_path.read_text(encoding="utf-8")
        assert "dataset,strategy,f1_score" in content


class TestWriteLatencyRecords:
    def test_writes_parquet_with_expected_columns(self, tmp_path: Path) -> None:
        records = [LatencyRecord(example_index=0, latency_ms=1.5)]
        output_path = _write_latency_records(records, "halueval", "lexical", tmp_path)
        frame = pd.read_parquet(output_path)
        assert list(frame.columns) == ["example_index", "latency_ms"]