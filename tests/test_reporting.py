"""
test_reporting.py

Covers the uncertainty columns, the per-model rows with their
interpretability flag, and the frontier table (reference strategies,
domination, strategies that could not be evaluated).

Author: Marco Pérez Padilla
Date:   17-10-2026
"""

from __future__ import annotations

import numpy as np

from src.reporting import (
    MIN_POSITIVES_FOR_INTERPRETATION,
    UNCERTAINTY_COLUMNS,
    frontier_rows,
    frontier_table,
    per_group_rows,
    uncertainty_columns,
)


def _data(n: int = 400, seed: int = 0) -> tuple[list[int], list[float], list[float], list[str], list[str]]:
    generator = np.random.default_rng(seed)
    labels = generator.integers(0, 2, size=n)
    scores = labels * 1.0 + generator.normal(0, 0.7, size=n)
    length_scores = generator.normal(0, 1.0, size=n)
    sources = [f"s{i // 4}" for i in range(n)]
    models = [f"m{i % 4}" for i in range(n)]
    return labels.tolist(), scores.tolist(), length_scores.tolist(), sources, models


class TestUncertaintyColumns:
    def test_every_column_is_present_and_filled_when_models_are_known(self) -> None:
        labels, scores, length, sources, models = _data()

        columns = uncertainty_columns(labels, scores, sources, models, length, 60, 42)

        assert set(columns) == set(UNCERTAINTY_COLUMNS)
        assert all(value is not None for value in columns.values())
        assert columns["auc_lower"] <= columns["auc_upper"]
        assert columns["within_model_auc_lower"] <= columns["within_model_auc"] <= columns["within_model_auc_upper"]

    def test_within_model_columns_stay_empty_without_models(self) -> None:
        labels, scores, length, sources, _ = _data()

        columns = uncertainty_columns(labels, scores, sources, [""] * len(labels), length, 60, 42)

        assert columns["auc_lower"] is not None
        assert columns["within_model_auc"] is None
        assert columns["within_model_auc_lower"] is None

    def test_everything_is_empty_when_a_single_class_is_present(self) -> None:
        columns = uncertainty_columns([1, 1, 1], [0.1, 0.2, 0.3], ["a", "b", "c"], ["", "", ""], [1.0, 2.0, 3.0], 20, 42)

        assert all(value is None for value in columns.values())

    def test_the_difference_with_length_is_zero_for_the_length_scores_themselves(self) -> None:
        labels, _, length, sources, models = _data()

        columns = uncertainty_columns(labels, length, sources, models, length, 40, 42)

        assert columns["auc_minus_length"] == 0.0
        assert columns["auc_minus_length_lower"] == columns["auc_minus_length_upper"] == 0.0


class TestPerGroupRows:
    def test_rows_flag_groups_with_few_positives(self) -> None:
        few = MIN_POSITIVES_FOR_INTERPRETATION - 1
        labels = [1] * few + [0] * 50 + [1] * 30 + [0] * 50
        scores = [0.9] * few + [0.1] * 50 + [0.8] * 30 + [0.2] * 50
        groups = ["rare"] * (few + 50) + ["common"] * 80

        rows = {r["model"]: r for r in per_group_rows("d", "s", labels, scores, groups, 40, 42)}

        assert rows["rare"]["interpretable"] is False
        assert rows["common"]["interpretable"] is True
        assert rows["common"]["positives"] == 30 and rows["common"]["n"] == 80

    def test_rows_carry_an_interval_that_brackets_the_auc(self) -> None:
        labels, scores, _, _, models = _data()

        for row in per_group_rows("d", "s", labels, scores, models, 60, 42):
            assert row["auc_lower"] <= row["auc"] <= row["auc_upper"]
            assert row["dataset"] == "d" and row["strategy"] == "s"

    def test_a_single_class_group_has_no_auc_and_no_interval(self) -> None:
        rows = per_group_rows("d", "s", [0, 0, 0, 1, 0], [0.1, 0.2, 0.3, 0.9, 0.4], ["a", "a", "a", "b", "b"], 20, 42)

        only_negatives = next(r for r in rows if r["model"] == "a")
        assert only_negatives["auc"] is None and only_negatives["auc_lower"] is None


def _row(name: str, latency: float, auc: float, lower: float | None, upper: float | None, dataset: str = "d") -> dict:
    return {"dataset": dataset, "strategy": name, "p50_latency_ms": latency, "auc_roc": auc, "auc_lower": lower, "auc_upper": upper}


class TestFrontierRows:
    def test_reproduces_the_halueval_situation(self) -> None:
        rows = [
            _row("lexical", 0.04, 0.917, 0.905, 0.928),
            _row("distilbert", 8.6, 0.997, 0.995, 0.998),
            _row("nli", 31.5, 0.527, 0.50, 0.55),
        ]

        table = {r["strategy"]: r for r in frontier_rows("d", rows, ["length"])}

        assert table["lexical"]["status"] == "frontier"
        assert table["distilbert"]["status"] == "frontier"
        assert table["nli"]["status"] == "dominated"
        assert table["nli"]["dominated_by"] == "lexical;distilbert"

    def test_reference_strategies_are_listed_but_take_no_part_in_the_rule(self) -> None:
        rows = [
            _row("length", 0.002, 0.97, 0.96, 0.98),   # would dominate everything if it counted
            _row("distilbert", 8.6, 0.80, 0.78, 0.82),
        ]

        table = {r["strategy"]: r for r in frontier_rows("d", rows, ["length"])}

        assert table["length"]["status"] == "reference"
        assert table["distilbert"]["status"] == "frontier"

    def test_a_strategy_without_an_interval_is_not_evaluated(self) -> None:
        table = {r["strategy"]: r for r in frontier_rows("d", [_row("odd", 1.0, 0.7, None, None)], [])}

        assert table["odd"]["status"] == "not_evaluated"

    def test_a_trade_off_keeps_both_on_the_frontier(self) -> None:
        rows = [_row("fast", 1.0, 0.7, 0.65, 0.75), _row("accurate", 50.0, 0.9, 0.88, 0.92)]

        assert [r["status"] for r in frontier_rows("d", rows, [])] == ["frontier", "frontier"]


class TestFrontierTable:
    def test_each_dataset_is_ranked_on_its_own(self) -> None:
        rows = [
            _row("a", 1.0, 0.9, 0.88, 0.92, dataset="first"),
            _row("b", 5.0, 0.6, 0.55, 0.65, dataset="first"),
            _row("a", 1.0, 0.6, 0.55, 0.65, dataset="second"),
            _row("b", 5.0, 0.9, 0.88, 0.92, dataset="second"),
        ]

        table = frontier_table(rows, [])

        assert [(r["dataset"], r["strategy"], r["status"]) for r in table] == [
            ("first", "a", "frontier"),
            ("first", "b", "dominated"),
            ("second", "a", "frontier"),
            ("second", "b", "frontier"),
        ]

    def test_empty_input_gives_an_empty_table(self) -> None:
        assert frontier_table([], ["length"]) == []