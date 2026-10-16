"""
test_breakdown.py

Covers per-group AUC: correct values, ordering, counts, single-class
groups and input validation.

Author: Marco Pérez Padilla
Date:   14-10-2026
"""

from __future__ import annotations

import pytest

from src.breakdown import GroupAuc, auc_by_group


def test_computes_the_auc_inside_each_group() -> None:
    labels = [0, 1, 0, 1]
    scores = [0.1, 0.9, 0.9, 0.1]  # group a perfect, group b perfectly inverted
    groups = ["a", "a", "b", "b"]

    results = auc_by_group(labels, scores, groups)

    assert results == [GroupAuc("a", 2, 1, 1.0), GroupAuc("b", 2, 1, 0.0)]


def test_a_pooled_auc_can_hide_that_every_group_is_at_chance() -> None:
    # Group "hard" hallucinates often and scores high; group "easy" rarely
    # does and scores low. Inside each group the scores carry no signal.
    labels = [1, 0, 1, 0] + [0, 0, 0, 1]
    scores = [0.8, 0.8, 0.8, 0.8] + [0.2, 0.2, 0.2, 0.2]
    groups = ["hard"] * 4 + ["easy"] * 4

    per_group = auc_by_group(labels, scores, groups)

    assert [g.auc for g in per_group] == [0.5, 0.5]


def test_groups_are_sorted_by_name() -> None:
    results = auc_by_group([0, 1, 0, 1], [0.1, 0.9, 0.1, 0.9], ["z", "z", "b", "b"])

    assert [r.group for r in results] == ["b", "z"]


def test_a_single_class_group_has_no_auc() -> None:
    results = {
        r.group: r
        for r in auc_by_group(
            [0, 0, 0, 1], [0.1, 0.2, 0.3, 0.9], ["only_neg", "only_neg", "only_neg", "x"]
        )
    }

    assert results["only_neg"].auc is None
    assert results["only_neg"].positive_count == 0 and results["only_neg"].sample_count == 3
    assert results["x"].auc is None  # one positive and nothing else


def test_empty_input_gives_no_groups() -> None:
    assert auc_by_group([], [], []) == []


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        auc_by_group([0, 1], [0.1], ["a", "b"])