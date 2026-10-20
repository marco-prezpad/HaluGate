"""
reporting.py

Turns scores into the uncertainty-aware columns and tables defined in
paper/protocol.md: bootstrap intervals for the AUC, for its difference with
the response-length baseline and for the within-model AUC; per-model rows
with an interpretability flag; and the frontier table.

Author: Marco Pérez Padilla
Date:   17-10-2026
"""

from __future__ import annotations

from typing import Any

from src.breakdown import auc_by_group
from src.statistics import (
    AucInterval,
    BootstrapError,
    DetectorResult,
    bootstrap_auc_difference,
    bootstrap_auc_interval,
    bootstrap_stratified_auc,
    bootstrap_stratified_auc_difference,
    dominators,
)

MIN_POSITIVES_FOR_INTERPRETATION = 20

UNCERTAINTY_COLUMNS = (
    "auc_lower",
    "auc_upper",
    "auc_minus_length",
    "auc_minus_length_lower",
    "auc_minus_length_upper",
    "within_model_auc",
    "within_model_auc_lower",
    "within_model_auc_upper",
    "within_model_minus_length",
    "within_model_minus_length_lower",
    "within_model_minus_length_upper",
    "beats_length_within_model",
)


def uncertainty_columns(
    labels: list[int],
    scores: list[float],
    source_groups: list[str],
    models: list[str],
    length_scores: list[float],
    n_resamples: int,
    random_seed: int,
) -> dict[str, float | bool | None]:
    """Bootstrap-based columns for one (dataset, detector) pair.

    All intervals resample whole source groups. The within-model columns
    are filled only when the samples record a generating model; any column

    ``beats_length_within_model`` is True when the paired interval of the
    within-model difference with the length baseline lies entirely above 0
    (protocol, amendment 3). It is False, not None, when the interval is
    computed and reaches 0, and it is also False for the baseline itself.
    """
    columns: dict[str, float | bool | None] = {name: None for name in UNCERTAINTY_COLUMNS}

    try:
        pooled = bootstrap_auc_interval(
            labels, scores, source_groups, n_resamples, random_seed=random_seed
        )
        columns["auc_lower"], columns["auc_upper"] = pooled.lower, pooled.upper

        versus_length = bootstrap_auc_difference(
            labels, scores, length_scores, source_groups, n_resamples, random_seed=random_seed
        )
        columns["auc_minus_length"] = versus_length.difference
        columns["auc_minus_length_lower"] = versus_length.lower
        columns["auc_minus_length_upper"] = versus_length.upper

        if any(models):
            within = bootstrap_stratified_auc(
                labels, scores, models, source_groups, n_resamples, random_seed=random_seed
            )
            columns["within_model_auc"] = within.auc
            columns["within_model_auc_lower"] = within.lower
            columns["within_model_auc_upper"] = within.upper

            within_versus_length = bootstrap_stratified_auc_difference(
                labels, scores, length_scores, models, source_groups, n_resamples,
                random_seed=random_seed,
            )
            columns["within_model_minus_length"] = within_versus_length.difference
            columns["within_model_minus_length_lower"] = within_versus_length.lower
            columns["within_model_minus_length_upper"] = within_versus_length.upper
            columns["beats_length_within_model"] = within_versus_length.lower > 0.0
    except BootstrapError:
        pass

    return columns


def per_group_rows(
    dataset_name: str,
    strategy_name: str,
    labels: list[int],
    scores: list[float],
    groups: list[str],
    n_resamples: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    """One row per group (generating model) with the AUC, its bootstrap
    interval and whether the group has enough positives to be interpreted.
    """
    rows: list[dict[str, Any]] = []
    for group_auc in auc_by_group(labels, scores, groups):
        lower = upper = None
        if group_auc.auc is not None:
            indices = [i for i, group in enumerate(groups) if group == group_auc.group]
            try:
                interval = bootstrap_auc_interval(
                    [labels[i] for i in indices],
                    [scores[i] for i in indices],
                    None,
                    n_resamples,
                    random_seed=random_seed,
                )
                lower, upper = interval.lower, interval.upper
            except BootstrapError:
                pass
        rows.append(
            {
                "dataset": dataset_name,
                "strategy": strategy_name,
                "model": group_auc.group,
                "n": group_auc.sample_count,
                "positives": group_auc.positive_count,
                "auc": group_auc.auc,
                "auc_lower": lower,
                "auc_upper": upper,
                "interpretable": group_auc.positive_count >= MIN_POSITIVES_FOR_INTERPRETATION,
            }
        )
    return rows


def frontier_rows(
    dataset_name: str,
    summary_rows: list[dict[str, Any]],
    reference_strategies: list[str],
) -> list[dict[str, Any]]:
    """Frontier status of every strategy of one dataset.

    Reference strategies are listed but excluded from the dominance rule,
    both as dominators and as candidates.
    """
    candidates = [
        row
        for row in summary_rows
        if row["strategy"] not in reference_strategies and row.get("auc_lower") is not None
    ]
    results = [
        DetectorResult(
            row["strategy"],
            row["p50_latency_ms"],
            AucInterval(row["auc_roc"], row["auc_lower"], row["auc_upper"]),
        )
        for row in candidates
    ]
    dominated_by = dominators(results)

    table: list[dict[str, Any]] = []
    for row in summary_rows:
        name = row["strategy"]
        if name in reference_strategies:
            status, by = "reference", []
        elif name not in dominated_by:
            status, by = "not_evaluated", []
        else:
            by = dominated_by[name]
            status = "dominated" if by else "frontier"
        table.append(
            {
                "dataset": dataset_name,
                "strategy": name,
                "p50_latency_ms": row["p50_latency_ms"],
                "auc": row["auc_roc"],
                "auc_lower": row.get("auc_lower"),
                "auc_upper": row.get("auc_upper"),
                "status": status,
                "dominated_by": ";".join(by),
            }
        )
    return table


def frontier_table(
    summary_rows: list[dict[str, Any]], reference_strategies: list[str]
) -> list[dict[str, Any]]:
    """``frontier_rows`` for every dataset present in ``summary_rows``, in order."""
    datasets: list[str] = []
    for row in summary_rows:
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])

    table: list[dict[str, Any]] = []
    for dataset_name in datasets:
        table.extend(
            frontier_rows(
                dataset_name,
                [row for row in summary_rows if row["dataset"] == dataset_name],
                reference_strategies,
            )
        )
    return table