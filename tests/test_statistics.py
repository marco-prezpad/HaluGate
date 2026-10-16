"""
test_statistics.py

Covers the grouped bootstrap interval (validity, determinism, group
resampling, error handling), the paired AUC difference, the stratified
(within-model) AUC and its interval, and the frontier rule, including the
exact situation measured on HaluEval.

Author: Marco Pérez Padilla
Date:   17-10-2026
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.statistics import (
    AucInterval,
    BootstrapError,
    DetectorResult,
    bootstrap_auc_difference,
    bootstrap_auc_interval,
    bootstrap_stratified_auc,
    dominates,
    dominators,
    pareto_frontier,
    stratified_auc,
)


def _noisy_scores(n: int, seed: int = 0) -> tuple[list[int], list[float]]:
    generator = np.random.default_rng(seed)
    labels = generator.integers(0, 2, size=n)
    scores = labels * 0.6 + generator.normal(0.0, 0.5, size=n)
    return labels.tolist(), scores.tolist()


class TestBootstrapAucInterval:
    def test_interval_brackets_the_point_estimate(self) -> None:
        labels, scores = _noisy_scores(400)

        result = bootstrap_auc_interval(labels, scores, n_resamples=300)

        assert result.lower <= result.auc <= result.upper
        assert 0.5 < result.auc < 1.0

    def test_same_seed_gives_identical_interval(self) -> None:
        labels, scores = _noisy_scores(200)

        first = bootstrap_auc_interval(labels, scores, n_resamples=100, random_seed=3)
        second = bootstrap_auc_interval(labels, scores, n_resamples=100, random_seed=3)

        assert first == second

    def test_perfect_scores_give_a_degenerate_interval_at_one(self) -> None:
        labels = [0, 1] * 50
        scores = [float(label) for label in labels]

        result = bootstrap_auc_interval(labels, scores, n_resamples=50)

        assert (result.auc, result.lower, result.upper) == (1.0, 1.0, 1.0)

    def test_grouped_resampling_is_wider_than_independent_resampling(self) -> None:
        # 40 groups of 5 samples sharing one score offset: samples within a
        # group are strongly correlated, so ignoring groups understates the
        # uncertainty.
        generator = np.random.default_rng(1)
        labels, scores, groups = [], [], []
        for group_index in range(40):
            group_offset = generator.normal(0.0, 1.0)
            group_label = group_index % 2
            for _ in range(5):
                labels.append(group_label)
                scores.append(group_label * 0.5 + group_offset + generator.normal(0.0, 0.1))
                groups.append(f"g{group_index}")

        grouped = bootstrap_auc_interval(labels, scores, groups, n_resamples=400)
        independent = bootstrap_auc_interval(labels, scores, None, n_resamples=400)

        assert (grouped.upper - grouped.lower) > (independent.upper - independent.lower)

    def test_higher_confidence_gives_a_wider_interval(self) -> None:
        labels, scores = _noisy_scores(300)

        narrow = bootstrap_auc_interval(labels, scores, n_resamples=300, confidence=0.5)
        wide = bootstrap_auc_interval(labels, scores, n_resamples=300, confidence=0.99)

        assert (wide.upper - wide.lower) > (narrow.upper - narrow.lower)

    @pytest.mark.parametrize(
        "labels, scores, groups, kwargs",
        [
            ([], [], None, {}),
            ([0, 1], [0.1], None, {}),
            ([0, 1], [0.1, 0.9], ["a"], {}),
            ([1, 1, 1], [0.1, 0.2, 0.3], None, {}),
            ([0, 1], [0.1, 0.9], None, {"n_resamples": 0}),
            ([0, 1], [0.1, 0.9], None, {"confidence": 1.0}),
            ([0, 1], [0.1, 0.9], None, {"confidence": 0.0}),
        ],
    )
    def test_rejects_invalid_inputs(self, labels, scores, groups, kwargs) -> None:
        with pytest.raises(BootstrapError):
            bootstrap_auc_interval(labels, scores, groups, **kwargs)

    def test_raises_when_no_resample_contains_both_classes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _AlwaysFirstGroup:
            """Fake generator: every resample draws group 0 repeatedly."""

            def integers(self, low: int, high: int, size: int) -> np.ndarray:
                return np.zeros(size, dtype=int)

        monkeypatch.setattr(np.random, "default_rng", lambda seed: _AlwaysFirstGroup())
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        groups = ["negatives", "negatives", "positives", "positives"]

        with pytest.raises(BootstrapError, match="both classes"):
            bootstrap_auc_interval(labels, scores, groups, n_resamples=5)


def _result(name: str, latency: float, lower: float, upper: float) -> DetectorResult:
    return DetectorResult(name, latency, AucInterval((lower + upper) / 2, lower, upper))


class TestFrontierRule:
    def test_faster_and_clearly_better_dominates(self) -> None:
        assert dominates(_result("a", 1.0, 0.9, 0.95), _result("b", 5.0, 0.6, 0.7))

    def test_slower_detector_never_dominates(self) -> None:
        assert not dominates(_result("a", 9.0, 0.9, 0.95), _result("b", 5.0, 0.6, 0.7))

    def test_overlapping_intervals_do_not_dominate(self) -> None:
        assert not dominates(_result("a", 1.0, 0.80, 0.90), _result("b", 5.0, 0.85, 0.95))

    def test_equal_latency_still_dominates_if_intervals_separate(self) -> None:
        assert dominates(_result("a", 5.0, 0.9, 0.95), _result("b", 5.0, 0.6, 0.7))

    def test_a_detector_never_dominates_itself(self) -> None:
        only = _result("a", 1.0, 0.9, 0.95)
        assert pareto_frontier([only]) == ["a"]

    def test_the_measured_halueval_situation(self) -> None:
        lexical = _result("lexical", 0.04, 0.905, 0.928)
        distilbert = _result("distilbert", 8.6, 0.995, 0.998)
        nli = _result("nli", 31.5, 0.50, 0.55)

        assert pareto_frontier([lexical, distilbert, nli]) == ["lexical", "distilbert"]

    def test_tradeoff_pair_both_stay_on_the_frontier(self) -> None:
        fast = _result("fast", 1.0, 0.7, 0.8)
        accurate = _result("accurate", 50.0, 0.9, 0.95)
        assert pareto_frontier([fast, accurate]) == ["fast", "accurate"]


class TestStratifiedAuc:
    def test_strata_are_weighted_by_their_number_of_pairs(self) -> None:
        # stratum g1: AUC 1.0 over 1 pair; stratum g2: AUC 0.0 over 4 pairs
        labels = [0, 1] + [0, 0, 1, 1]
        scores = [0.1, 0.9] + [0.9, 0.8, 0.1, 0.2]
        strata = ["g1", "g1"] + ["g2"] * 4

        assert stratified_auc(labels, scores, strata) == pytest.approx((1 * 1.0 + 4 * 0.0) / 5)

    def test_strata_with_a_single_class_are_ignored(self) -> None:
        labels = [0, 1, 0, 0]
        scores = [0.1, 0.9, 0.9, 0.1]
        strata = ["a", "a", "only_negatives", "only_negatives"]

        assert stratified_auc(labels, scores, strata) == pytest.approx(1.0)

    def test_none_when_no_stratum_has_both_classes(self) -> None:
        assert stratified_auc([0, 0, 1], [0.1, 0.2, 0.3], ["a", "a", "b"]) is None

    def test_it_removes_the_advantage_of_recognising_the_stratum(self) -> None:
        # "hard" hallucinates more and scores high; "easy" rarely does and
        # scores low. Inside each stratum the scores carry no signal.
        labels = [1, 0, 1, 0] + [0, 0, 0, 1]
        scores = [0.8, 0.8, 0.8, 0.8] + [0.2, 0.2, 0.2, 0.2]
        strata = ["hard"] * 4 + ["easy"] * 4

        assert roc_auc_score(labels, scores) > 0.6
        assert stratified_auc(labels, scores, strata) == pytest.approx(0.5)

    def test_with_a_single_stratum_it_equals_the_plain_auc(self) -> None:
        labels, scores = _noisy_scores(200, seed=4)

        assert stratified_auc(labels, scores, ["only"] * 200) == pytest.approx(
            roc_auc_score(labels, scores)
        )


class TestBootstrapAucDifference:
    def test_identical_scores_give_a_zero_difference(self) -> None:
        labels, scores = _noisy_scores(300)

        result = bootstrap_auc_difference(labels, scores, scores, n_resamples=100)

        assert (result.difference, result.lower, result.upper) == (0.0, 0.0, 0.0)

    def test_a_clearly_better_detector_has_an_interval_above_zero(self) -> None:
        generator = np.random.default_rng(2)
        labels = generator.integers(0, 2, size=400)
        good = labels * 1.5 + generator.normal(0, 0.5, size=400)
        poor = generator.normal(0, 1.0, size=400)

        result = bootstrap_auc_difference(labels.tolist(), good.tolist(), poor.tolist(), n_resamples=200)

        assert result.difference > 0.3 and result.lower > 0.0

    def test_swapping_the_detectors_flips_the_sign(self) -> None:
        generator = np.random.default_rng(3)
        labels = generator.integers(0, 2, size=300)
        good = labels * 1.5 + generator.normal(0, 0.5, size=300)
        poor = generator.normal(0, 1.0, size=300)

        forward = bootstrap_auc_difference(labels.tolist(), good.tolist(), poor.tolist(), n_resamples=100)
        backward = bootstrap_auc_difference(labels.tolist(), poor.tolist(), good.tolist(), n_resamples=100)

        assert backward.difference == pytest.approx(-forward.difference)
        assert backward.upper == pytest.approx(-forward.lower)

    def test_same_seed_gives_the_same_interval(self) -> None:
        labels, scores = _noisy_scores(200)
        other = [score + 0.1 * i % 3 for i, score in enumerate(scores)]

        first = bootstrap_auc_difference(labels, scores, other, n_resamples=60, random_seed=5)
        second = bootstrap_auc_difference(labels, scores, other, n_resamples=60, random_seed=5)

        assert first == second

    @pytest.mark.parametrize(
        "labels, scores_a, scores_b, groups",
        [
            ([0, 1], [0.1, 0.9], [0.2], None),
            ([0, 1], [0.1, 0.9], [0.2, 0.8], ["g"]),
            ([1, 1], [0.1, 0.9], [0.2, 0.8], None),
            ([], [], [], None),
        ],
    )
    def test_rejects_invalid_inputs(self, labels, scores_a, scores_b, groups) -> None:
        with pytest.raises(BootstrapError):
            bootstrap_auc_difference(labels, scores_a, scores_b, groups)


class TestBootstrapStratifiedAuc:
    def test_interval_brackets_the_point_estimate(self) -> None:
        labels, scores = _noisy_scores(600, seed=6)
        strata = [f"m{i % 3}" for i in range(600)]

        result = bootstrap_stratified_auc(labels, scores, strata, n_resamples=200)

        assert result.lower <= result.auc <= result.upper

    def test_point_estimate_matches_the_stratified_auc(self) -> None:
        labels, scores = _noisy_scores(300, seed=7)
        strata = [f"m{i % 2}" for i in range(300)]

        result = bootstrap_stratified_auc(labels, scores, strata, n_resamples=50)

        assert result.auc == pytest.approx(stratified_auc(labels, scores, strata))

    def test_same_seed_gives_the_same_interval(self) -> None:
        labels, scores = _noisy_scores(300, seed=8)
        strata = [f"m{i % 2}" for i in range(300)]

        first = bootstrap_stratified_auc(labels, scores, strata, n_resamples=60, random_seed=1)
        second = bootstrap_stratified_auc(labels, scores, strata, n_resamples=60, random_seed=1)

        assert first == second

    def test_raises_when_no_stratum_has_both_classes(self) -> None:
        with pytest.raises(BootstrapError, match="stratum"):
            bootstrap_stratified_auc([0, 0, 1], [0.1, 0.2, 0.3], ["a", "a", "b"], n_resamples=10)

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(BootstrapError):
            bootstrap_stratified_auc([0, 1], [0.1, 0.9], ["a"])


class TestDominators:
    def test_lists_who_dominates_each_detector(self) -> None:
        lexical = _result("lexical", 0.04, 0.905, 0.928)
        distilbert = _result("distilbert", 8.6, 0.995, 0.998)
        nli = _result("nli", 31.5, 0.50, 0.55)

        assert dominators([lexical, distilbert, nli]) == {
            "lexical": [],
            "distilbert": [],
            "nli": ["lexical", "distilbert"],
        }

    def test_a_single_detector_is_dominated_by_nobody(self) -> None:
        assert dominators([_result("only", 1.0, 0.5, 0.6)]) == {"only": []}