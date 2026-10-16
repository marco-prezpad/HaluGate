"""
statistics.py

Statistics required by paper/protocol.md:

- grouped bootstrap intervals (resampling whole groups, e.g. all responses
  that share one source document, so correlated samples are not treated as
  independent) for the AUC-ROC, for the AUC difference between two
  detectors scored on the same samples, and for the within-stratum AUC;
- the within-stratum ("stratified") AUC itself: the probability that a
  hallucinated response outscores a correct one *written by the same
  model*, which removes the advantage of merely recognising the model;
- the Pareto-dominance rule that decides which detectors sit on the
  latency-accuracy frontier.

Author: Marco Pérez Padilla
Date:   17-10-2026
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from src.errors import HaluGateError


class BootstrapError(HaluGateError):
    """Raised when a bootstrap interval cannot be computed from the data."""


@dataclass(frozen=True)
class AucInterval:
    """Point estimate and percentile bootstrap interval of an AUC."""

    auc: float
    lower: float
    upper: float


@dataclass(frozen=True)
class DifferenceInterval:
    """Point estimate and percentile bootstrap interval of an AUC difference."""

    difference: float
    lower: float
    upper: float


def stratified_auc(labels: list[int], scores: list[float], strata: list[str]) -> float | None:
    """Pair-weighted mean of the AUC inside each stratum.

    Only pairs of samples from the same stratum are compared, and each
    stratum counts in proportion to its number of (positive, negative)
    pairs, so a stratum with a single positive barely moves the result.
    Strata with a single class are ignored. Returns ``None`` if no stratum
    has both classes.
    """
    label_array = np.asarray(labels)
    score_array = np.asarray(scores, dtype=float)
    strata_array = np.asarray(strata)

    weighted_sum = 0.0
    total_pairs = 0
    for stratum in np.unique(strata_array):
        mask = strata_array == stratum
        positives = int(label_array[mask].sum())
        negatives = int(mask.sum()) - positives
        if positives == 0 or negatives == 0:
            continue
        pairs = positives * negatives
        weighted_sum += pairs * roc_auc_score(label_array[mask], score_array[mask])
        total_pairs += pairs

    return weighted_sum / total_pairs if total_pairs else None


def _validate(
    sizes: list[int], labels: list[int], n_resamples: int, confidence: float
) -> None:
    if len(set(sizes)) != 1:
        raise BootstrapError("all inputs must have the same length")
    if sizes[0] == 0:
        raise BootstrapError("cannot bootstrap an empty sample")
    if len(set(labels)) < 2:
        raise BootstrapError("AUC is undefined when only one class is present")
    if n_resamples < 1:
        raise BootstrapError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < confidence < 1.0:
        raise BootstrapError(f"confidence must be in (0, 1), got {confidence}")


def _grouped_bootstrap(
    group_ids: list[str],
    n_resamples: int,
    random_seed: int,
    statistic: Callable[[np.ndarray], float | None],
) -> list[float]:
    """Evaluate ``statistic`` on resamples that draw whole groups with replacement.

    ``statistic`` receives the indices of one resample and returns a value,
    or ``None`` when it is undefined (for example a single class).
    """
    indices_by_group: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        indices_by_group.setdefault(group_id, []).append(index)
    group_index_arrays = [np.asarray(indices) for indices in indices_by_group.values()]

    generator = np.random.default_rng(random_seed)
    values: list[float] = []
    for _ in range(n_resamples):
        chosen = generator.integers(0, len(group_index_arrays), size=len(group_index_arrays))
        sample_indices = np.concatenate([group_index_arrays[g] for g in chosen])
        value = statistic(sample_indices)
        if value is not None:
            values.append(value)

    if not values:
        raise BootstrapError("no bootstrap resample contained both classes")
    return values


def _percentile_interval(values: list[float], confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0 * 100.0
    lower, upper = np.percentile(values, [tail, 100.0 - tail])
    return float(lower), float(upper)


def _auc_if_both_classes(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def bootstrap_auc_interval(
    labels: list[int],
    scores: list[float],
    groups: list[str] | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> AucInterval:
    """Percentile bootstrap CI for AUC-ROC, resampling whole groups.

    Args:
        labels: ground truth, 1 = hallucination.
        scores: continuous scores, higher = more likely hallucinated.
        groups: group id per sample (e.g. source document). ``None`` makes
            every sample its own group.
        n_resamples: number of bootstrap resamples.
        confidence: central interval mass, in (0, 1).
        random_seed: seed of the resampling generator.

    Raises:
        BootstrapError: on empty or mismatched inputs, single-class labels,
            invalid parameters, or when no resample contained both classes.
    """
    sizes = [len(labels), len(scores)] + ([len(groups)] if groups is not None else [])
    _validate(sizes, labels, n_resamples, confidence)

    label_array = np.asarray(labels)
    score_array = np.asarray(scores, dtype=float)
    group_ids = groups if groups is not None else [str(i) for i in range(len(labels))]

    values = _grouped_bootstrap(
        group_ids,
        n_resamples,
        random_seed,
        lambda idx: _auc_if_both_classes(label_array[idx], score_array[idx]),
    )
    lower, upper = _percentile_interval(values, confidence)
    return AucInterval(float(roc_auc_score(label_array, score_array)), lower, upper)


def bootstrap_auc_difference(
    labels: list[int],
    scores_a: list[float],
    scores_b: list[float],
    groups: list[str] | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> DifferenceInterval:
    """Paired bootstrap CI for AUC(a) - AUC(b) on the same samples.

    Both detectors are evaluated on the very same resamples, which is what
    makes the interval much tighter than comparing two separate intervals.

    Raises:
        BootstrapError: under the same conditions as ``bootstrap_auc_interval``.
    """
    sizes = [len(labels), len(scores_a), len(scores_b)]
    sizes += [len(groups)] if groups is not None else []
    _validate(sizes, labels, n_resamples, confidence)

    label_array = np.asarray(labels)
    array_a = np.asarray(scores_a, dtype=float)
    array_b = np.asarray(scores_b, dtype=float)
    group_ids = groups if groups is not None else [str(i) for i in range(len(labels))]

    def _difference(idx: np.ndarray) -> float | None:
        auc_a = _auc_if_both_classes(label_array[idx], array_a[idx])
        auc_b = _auc_if_both_classes(label_array[idx], array_b[idx])
        return None if auc_a is None or auc_b is None else auc_a - auc_b

    values = _grouped_bootstrap(group_ids, n_resamples, random_seed, _difference)
    lower, upper = _percentile_interval(values, confidence)
    estimate = float(roc_auc_score(label_array, array_a) - roc_auc_score(label_array, array_b))
    return DifferenceInterval(estimate, lower, upper)


def bootstrap_stratified_auc(
    labels: list[int],
    scores: list[float],
    strata: list[str],
    groups: list[str] | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> AucInterval:
    """Bootstrap CI for the within-stratum AUC (see ``stratified_auc``).

    Raises:
        BootstrapError: under the same conditions as ``bootstrap_auc_interval``,
            or when no stratum has both classes.
    """
    sizes = [len(labels), len(scores), len(strata)]
    sizes += [len(groups)] if groups is not None else []
    _validate(sizes, labels, n_resamples, confidence)

    estimate = stratified_auc(labels, scores, strata)
    if estimate is None:
        raise BootstrapError("no stratum contains both classes")

    label_list = np.asarray(labels)
    score_array = np.asarray(scores, dtype=float)
    strata_array = np.asarray(strata)
    group_ids = groups if groups is not None else [str(i) for i in range(len(labels))]

    values = _grouped_bootstrap(
        group_ids,
        n_resamples,
        random_seed,
        lambda idx: stratified_auc(
            label_list[idx].tolist(), score_array[idx].tolist(), strata_array[idx].tolist()
        ),
    )
    lower, upper = _percentile_interval(values, confidence)
    return AucInterval(float(estimate), lower, upper)


@dataclass(frozen=True)
class DetectorResult:
    """What the frontier rule needs to know about one detector."""

    name: str
    p50_latency_ms: float
    auc_interval: AucInterval


def dominates(candidate: DetectorResult, other: DetectorResult) -> bool:
    """True if ``candidate`` is at least as fast and its AUC interval lies
    entirely above ``other``'s (protocol: "Frontier rule").
    """
    return (
        candidate.p50_latency_ms <= other.p50_latency_ms
        and candidate.auc_interval.lower > other.auc_interval.upper
    )


def dominators(results: list[DetectorResult]) -> dict[str, list[str]]:
    """For each detector, the names of the detectors that dominate it."""
    return {
        result.name: [
            other.name for other in results if other is not result and dominates(other, result)
        ]
        for result in results
    }


def pareto_frontier(results: list[DetectorResult]) -> list[str]:
    """Names of the detectors that no other detector dominates, in input order."""
    dominated_by = dominators(results)
    return [result.name for result in results if not dominated_by[result.name]]