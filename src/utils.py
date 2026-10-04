"""
utils.py

Cross-cutting helpers shared by every module: reproducibility seeding,
classification metrics, a lightweight JSON-based checkpoint manager for
resumable evaluation runs, logging setup, and a length-sort helper used by
the batched strategies to minimize padding.

Author: Marco Pérez Padilla
Date:   04-10-2026
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.errors import CheckpointCorruptedError

CHECKPOINT_SCHEMA_VERSION = 1


def set_global_seed(seed: int) -> None:
    """Seed every source of randomness we rely on for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def configure_logging(log_path: Path | None = None, verbose: bool = True) -> logging.Logger:
    """Configure a project-wide logger that writes to stdout and, optionally, to disk."""
    logger = logging.getLogger("halugate")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


@dataclass
class ClassificationMetrics:
    """Container for the metrics reported per detector in the results tables."""

    true_positive_rate: float
    false_positive_rate: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    auc_roc: float | None = None
    sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_classification_metrics(
    ground_truth_labels: list[int],
    predicted_labels: list[int],
    predicted_scores: list[float] | None = None,
) -> ClassificationMetrics:
    """Compute standard binary-classification metrics.

    Args:
        ground_truth_labels: 1 if the response is a hallucination, else 0.
        predicted_labels: the detector's binary decision for each sample.
        predicted_scores: optional continuous scores, used for AUC-ROC when
            available. If omitted, ``auc_roc`` is left as ``None``.

    Raises:
        ValueError: if the input lists are empty or of mismatched length.
    """
    if len(ground_truth_labels) == 0:
        raise ValueError("Cannot compute metrics on an empty set of labels.")
    if len(ground_truth_labels) != len(predicted_labels):
        raise ValueError(
            "ground_truth_labels and predicted_labels must have the same length: "
            f"{len(ground_truth_labels)} != {len(predicted_labels)}"
        )

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    true_positives = sum(
        1 for gt, pred in zip(ground_truth_labels, predicted_labels) if gt == 1 and pred == 1
    )
    false_positives = sum(
        1 for gt, pred in zip(ground_truth_labels, predicted_labels) if gt == 0 and pred == 1
    )
    actual_positive_count = sum(1 for gt in ground_truth_labels if gt == 1)
    actual_negative_count = sum(1 for gt in ground_truth_labels if gt == 0)

    true_positive_rate = (
        true_positives / actual_positive_count if actual_positive_count > 0 else 0.0
    )
    false_positive_rate = (
        false_positives / actual_negative_count if actual_negative_count > 0 else 0.0
    )

    auc_roc_value: float | None = None
    if predicted_scores is not None and len(set(ground_truth_labels)) > 1:
        auc_roc_value = float(roc_auc_score(ground_truth_labels, predicted_scores))

    return ClassificationMetrics(
        true_positive_rate=true_positive_rate,
        false_positive_rate=false_positive_rate,
        precision=float(precision_score(ground_truth_labels, predicted_labels, zero_division=0)),
        recall=float(recall_score(ground_truth_labels, predicted_labels, zero_division=0)),
        f1_score=float(f1_score(ground_truth_labels, predicted_labels, zero_division=0)),
        accuracy=float(accuracy_score(ground_truth_labels, predicted_labels)),
        auc_roc=auc_roc_value,
        sample_count=len(ground_truth_labels),
    )


@dataclass
class CheckpointState:
    """Serializable progress snapshot for a single strategy's evaluation run."""

    strategy_name: str
    dataset_name: str
    next_sample_index: int
    predicted_labels: list[int] = field(default_factory=list)
    predicted_scores: list[float] = field(default_factory=list)
    latencies_seconds: list[float] = field(default_factory=list)
    schema_version: int = CHECKPOINT_SCHEMA_VERSION


class CheckpointManager:
    """Saves and restores :class:`CheckpointState` snapshots as JSON files.

    Evaluation over transformer-based strategies can take hours; this lets a
    crashed or interrupted run resume from the last saved sample index
    instead of restarting from scratch.
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, strategy_name: str, dataset_name: str) -> Path:
        safe_name = f"{dataset_name}__{strategy_name}".replace("/", "_")
        return self.checkpoint_dir / f"checkpoint_{safe_name}.json"

    def load(self, strategy_name: str, dataset_name: str) -> CheckpointState | None:
        """Return the saved state, or ``None`` if no checkpoint exists yet."""
        path = self._checkpoint_path(strategy_name, dataset_name)
        if not path.exists():
            return None

        try:
            raw_state = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CheckpointCorruptedError(str(path), f"invalid JSON ({exc})") from exc

        if raw_state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointCorruptedError(
                str(path),
                f"schema version {raw_state.get('schema_version')!r} "
                f"!= expected {CHECKPOINT_SCHEMA_VERSION}",
            )

        try:
            return CheckpointState(**raw_state)
        except TypeError as exc:
            raise CheckpointCorruptedError(str(path), f"missing fields ({exc})") from exc

    def save(self, state: CheckpointState) -> None:
        """Atomically persist ``state`` to disk (write to tmp file, then rename)."""
        path = self._checkpoint_path(state.strategy_name, state.dataset_name)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(asdict(state)), encoding="utf-8")
        tmp_path.replace(path)

    def clear(self, strategy_name: str, dataset_name: str) -> None:
        """Delete the checkpoint once a strategy has finished successfully."""
        path = self._checkpoint_path(strategy_name, dataset_name)
        path.unlink(missing_ok=True)


def sort_indices_by_text_length(texts: list[str]) -> list[int]:
    """Return indices that sort ``texts`` from shortest to longest.

    Feeding a batched transformer model samples of similar length minimizes
    padding -- a meaningful, well-known throughput win, no exotic tricks.
    """
    return sorted(range(len(texts)), key=lambda i: len(texts[i]))