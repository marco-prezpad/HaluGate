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
from src.data_loading import HallucinationSample

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


class _SplitStubLoader:
    """Fake dataset loader for the split/strategy-selection tests."""

    name = "fake"

    def __init__(self, samples: list[HallucinationSample]) -> None:
        self._samples = samples

    def load(self) -> list[HallucinationSample]:
        return self._samples


class _TaggedDetector(HallucinationDetector):
    """Scores 1.0 for responses starting with 'HALL', 0.0 otherwise."""

    name = "tagged"

    def predict_batch(self, contexts, responses):
        scores = [1.0 if response.startswith("HALL") else 0.0 for response in responses]
        return [score >= 0.5 for score in scores], scores


def _official_split_samples(train_sources: int, test_sources: int) -> list[HallucinationSample]:
    samples = []
    for source in range(train_sources + test_sources):
        split = "train" if source < train_sources else "test"
        for response in range(6):
            hallucinated = response % 2 == 0
            samples.append(
                HallucinationSample(
                    context=f"passages {source}",
                    response=f"{'HALL ' if hallucinated else ''}answer {source}-{response}",
                    is_hallucination=hallucinated,
                    source_dataset="fake",
                    group_id=f"s{source}",
                    official_split=split,
                )
            )
    return samples


def _run_with_stub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    samples: list[HallucinationSample],
    extra_config: dict[str, Any],
) -> list[dict[str, Any]]:
    import src.evaluate as evaluate_module

    monkeypatch.setattr(
        evaluate_module, "get_dataset_loader", lambda name, root: _SplitStubLoader(samples)
    )
    monkeypatch.setattr(evaluate_module, "get_detector", lambda name, params: _TaggedDetector())

    config = {
        "data_root": "data",
        "results_root": str(tmp_path / "results"),
        "random_seed": 0,
        "checkpoint_every": 10,
        "latency_sample_size": 5,
        "splits": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "datasets": ["fake"],
        "strategies": ["tagged"],
        "strategy_params": {},
        **extra_config,
    }
    return evaluate_module.run_evaluation(config, configure_logging())


class TestRunEvaluationSplitsAndStrategies:
    def test_official_test_partition_is_what_gets_scored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        samples = _official_split_samples(train_sources=30, test_sources=10)

        rows = _run_with_stub(monkeypatch, tmp_path, samples, {})

        assert rows[0]["sample_count"] == 10 * 6
        assert rows[0]["auc_roc"] == pytest.approx(1.0)

    def test_strategies_by_dataset_overrides_the_global_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        samples = _official_split_samples(train_sources=30, test_sources=10)

        rows = _run_with_stub(
            monkeypatch,
            tmp_path,
            samples,
            {"strategies": ["tagged", "other"], "strategies_by_dataset": {"fake": ["tagged"]}},
        )

        assert [row["strategy"] for row in rows] == ["tagged"]

    def test_global_strategies_are_used_when_the_dataset_has_no_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        samples = _official_split_samples(train_sources=30, test_sources=10)

        rows = _run_with_stub(
            monkeypatch,
            tmp_path,
            samples,
            {"strategies": ["tagged", "other"], "strategies_by_dataset": {"another_dataset": ["x"]}},
        )

        assert [row["strategy"] for row in rows] == ["tagged", "other"]