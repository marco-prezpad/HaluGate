"""
evaluate.py

Orchestrates the full HaluGate evaluation: for every dataset and every
detector strategy listed in config.yaml, it:

  1. Splits the dataset into train/validation/test (grouped by shared
     context, see src/splitting.py).
  2. Scores the validation split and calibrates a decision threshold that
     maximizes F1 against it (see src/calibration.py) -- rather than
     trusting a hardcoded threshold to be meaningful for every strategy
     and configuration.
  3. Scores the test split and reports metrics using that calibrated
     threshold. This keeps the reported F1/precision/recall honest: the
     threshold is chosen on data the final metrics are not computed on.

Usage:
    python -m src.evaluate --config configs/config.yaml

Author: Marco Pérez Padilla
Date:   05-10-2026
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.calibration import find_optimal_threshold
from src.data_loading import get_dataset_loader
from src.errors import ConfigValidationError
from src.factory import get_detector
from src.latency_bench import LatencyRecord, measure_latency
from src.splitting import split_dataset
from src.utils import (
    CheckpointManager,
    CheckpointState,
    ClassificationMetrics,
    compute_classification_metrics,
    configure_logging,
    set_global_seed,
)

DEFAULT_CHECKPOINT_EVERY = 50
DEFAULT_SPLITS = {"train": 0.7, "validation": 0.15, "test": 0.15}


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and minimally validate config.yaml."""
    if not config_path.exists():
        raise ConfigValidationError("config_path", f"file not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config: dict[str, Any] = yaml.safe_load(config_file)

    for required_field in ("datasets", "strategies", "data_root", "results_root"):
        if required_field not in config:
            raise ConfigValidationError(required_field, "missing from config.yaml")
    if not config["datasets"]:
        raise ConfigValidationError("datasets", "must list at least one dataset name")
    if not config["strategies"]:
        raise ConfigValidationError("strategies", "must list at least one strategy name")

    return config


def _run_strategy_with_checkpointing(
    strategy_name: str,
    strategy_params: dict[str, Any],
    dataset_name: str,
    contexts: list[str],
    responses: list[str],
    checkpoint_manager: CheckpointManager,
    checkpoint_every: int,
    logger: Any,
) -> tuple[list[int], list[float]]:
    """Score every sample for one (dataset[+split], strategy) pair, resuming
    from a checkpoint if one exists. ``dataset_name`` is an opaque
    checkpoint key, so callers pass e.g. "halueval__validation" and
    "halueval__test" to checkpoint each split independently.
    """
    state = checkpoint_manager.load(strategy_name, dataset_name)
    if state is None:
        state = CheckpointState(
            strategy_name=strategy_name, dataset_name=dataset_name, next_sample_index=0
        )
    elif state.next_sample_index > 0:
        logger.info(
            "Resuming %s on %s from sample %d/%d (checkpoint found)",
            strategy_name, dataset_name, state.next_sample_index, len(contexts),
        )

    if state.next_sample_index >= len(contexts):
        return state.predicted_labels, state.predicted_scores

    detector = get_detector(strategy_name, strategy_params)

    for batch_start in range(state.next_sample_index, len(contexts), checkpoint_every):
        batch_end = min(batch_start + checkpoint_every, len(contexts))
        batch_labels, batch_scores = detector.predict_batch(
            contexts[batch_start:batch_end], responses[batch_start:batch_end]
        )
        state.predicted_labels.extend(int(label) for label in batch_labels)
        state.predicted_scores.extend(float(score) for score in batch_scores)
        state.next_sample_index = batch_end
        checkpoint_manager.save(state)
        logger.debug(
            "Checkpointed %s on %s at sample %d/%d",
            strategy_name, dataset_name, batch_end, len(contexts),
        )

    return state.predicted_labels, state.predicted_scores


def _write_metrics_table(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_latency_records(
    records: list[LatencyRecord], dataset_name: str, strategy_name: str, latency_dir: Path
) -> Path:
    """Persist raw per-sample latencies for one pair as a Parquet file."""
    latency_dir.mkdir(parents=True, exist_ok=True)
    output_path = latency_dir / f"{dataset_name}__{strategy_name}.parquet"
    frame = pd.DataFrame([record.to_dict() for record in records])
    frame.to_parquet(output_path, index=False)
    return output_path


def _plot_frontier(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Plot the latency-accuracy frontier: F1 score vs. mean latency."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 5))

    datasets = sorted({row["dataset"] for row in rows})
    markers = ["o", "s", "^", "D", "P"]

    for dataset_index, dataset_name in enumerate(datasets):
        dataset_rows = [row for row in rows if row["dataset"] == dataset_name]
        latencies_ms = [row["mean_latency_ms"] for row in dataset_rows]
        f1_scores = [row["f1_score"] for row in dataset_rows]
        labels = [row["strategy"] for row in dataset_rows]

        axis.scatter(
            latencies_ms, f1_scores, label=dataset_name,
            marker=markers[dataset_index % len(markers)], s=80,
            color="steelblue" if dataset_index == 0 else "firebrick",
        )
        for x_coord, y_coord, label in zip(latencies_ms, f1_scores, labels):
            axis.annotate(label, (x_coord, y_coord), textcoords="offset points", xytext=(6, 4))

    axis.set_xlabel("Mean latency per sample (ms)")
    axis.set_ylabel("F1 score (test split, calibrated threshold)")
    axis.set_title("HaluGate: latency-accuracy frontier")
    axis.legend(title="Dataset")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_evaluation(config: dict[str, Any], logger: Any) -> list[dict[str, Any]]:
    """Evaluate every (dataset, strategy) combination and return one summary
    row per combination. For each pair: calibrate a decision threshold on
    the validation split, then compute final metrics on the test split
    using that threshold.
    """
    set_global_seed(config.get("random_seed", 42))

    data_root = Path(config["data_root"])
    results_root = Path(config["results_root"])
    checkpoint_manager = CheckpointManager(results_root / "logs" / "checkpoints")
    latency_dir = results_root / "latency"
    checkpoint_every = config.get("checkpoint_every", DEFAULT_CHECKPOINT_EVERY)
    strategy_params_by_name = config.get("strategy_params", {})
    split_ratios = config.get("splits", DEFAULT_SPLITS)
    random_seed = config.get("random_seed", 42)

    summary_rows: list[dict[str, Any]] = []

    for dataset_name in config["datasets"]:
        logger.info("Loading dataset '%s'", dataset_name)
        loader = get_dataset_loader(dataset_name, data_root)
        samples = loader.load()
        logger.info("Loaded %d samples from '%s'", len(samples), dataset_name)

        splits = split_dataset(samples, split_ratios, random_seed)
        logger.info(
            "Split '%s' into train=%d validation=%d test=%d",
            dataset_name, len(splits.train), len(splits.validation), len(splits.test),
        )

        validation_contexts = [s.context for s in splits.validation]
        validation_responses = [s.response for s in splits.validation]
        validation_ground_truth = [int(s.is_hallucination) for s in splits.validation]

        test_contexts = [s.context for s in splits.test]
        test_responses = [s.response for s in splits.test]
        test_ground_truth = [int(s.is_hallucination) for s in splits.test]

        strategies_for_dataset = config.get("strategies_by_dataset", {}).get(
            dataset_name, config["strategies"]
        )
        for strategy_name in strategies_for_dataset:
            strategy_params = dict(strategy_params_by_name.get(strategy_name, {}))

            logger.info("Calibrating '%s' on '%s' validation split", strategy_name, dataset_name)
            _, validation_scores = _run_strategy_with_checkpointing(
                strategy_name=strategy_name, strategy_params=strategy_params,
                dataset_name=f"{dataset_name}__validation", contexts=validation_contexts,
                responses=validation_responses, checkpoint_manager=checkpoint_manager,
                checkpoint_every=checkpoint_every, logger=logger,
            )
            calibration_result = find_optimal_threshold(validation_scores, validation_ground_truth)
            logger.info(
                "Calibrated threshold=%.4f (validation F1=%.3f) for %s/%s",
                calibration_result.threshold, calibration_result.f1_score,
                dataset_name, strategy_name,
            )

            logger.info("Evaluating '%s' on '%s' test split", strategy_name, dataset_name)
            _, test_scores = _run_strategy_with_checkpointing(
                strategy_name=strategy_name, strategy_params=strategy_params,
                dataset_name=f"{dataset_name}__test", contexts=test_contexts,
                responses=test_responses, checkpoint_manager=checkpoint_manager,
                checkpoint_every=checkpoint_every, logger=logger,
            )
            test_predicted_labels = [
                int(score >= calibration_result.threshold) for score in test_scores
            ]
            metrics: ClassificationMetrics = compute_classification_metrics(
                test_ground_truth, test_predicted_labels, test_scores
            )

            latency_sample_size = min(len(test_contexts), config.get("latency_sample_size", 200))
            detector_for_latency = get_detector(strategy_name, strategy_params)
            latency_records, latency_stats = measure_latency(
                detector_for_latency,
                test_contexts[:latency_sample_size], test_responses[:latency_sample_size],
            )
            _write_latency_records(latency_records, dataset_name, strategy_name, latency_dir)

            checkpoint_manager.clear(strategy_name, f"{dataset_name}__validation")
            checkpoint_manager.clear(strategy_name, f"{dataset_name}__test")

            summary_rows.append({
                "dataset": dataset_name,
                "strategy": strategy_name,
                "calibrated_threshold": calibration_result.threshold,
                "calibration_f1": calibration_result.f1_score,
                **metrics.to_dict(),
                "mean_latency_ms": latency_stats.mean_ms,
                "std_latency_ms": latency_stats.std_ms,
                "p50_latency_ms": latency_stats.p50_ms,
                "p95_latency_ms": latency_stats.p95_ms,
                "p99_latency_ms": latency_stats.p99_ms,
            })
            logger.info(
                "Finished %s/%s: test F1=%.3f test AUC=%s mean_latency=%.2fms",
                dataset_name, strategy_name, metrics.f1_score,
                f"{metrics.auc_roc:.3f}" if metrics.auc_roc is not None else "n/a",
                latency_stats.mean_ms,
            )

    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HaluGate evaluation pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    results_root = Path(config["results_root"])
    logger = configure_logging(results_root / "logs" / "evaluate.log")

    summary_rows = run_evaluation(config, logger)

    _write_metrics_table(summary_rows, results_root / "tables" / "metrics_summary.csv")
    _plot_frontier(summary_rows, results_root / "figures" / "latency_accuracy_frontier.png")
    logger.info("Evaluation complete. Results written under %s", results_root)


if __name__ == "__main__":
    main()