"""
test_paper_tables.py

Covers the number formatting, the frontier recomputed with the median of the
repeated p50, and the main, latency and truncation tables, on small result
tables shaped like the real CSV files (tests/paper_data.py).

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.paper_tables import (
    NOT_AVAILABLE,
    fmt_interval,
    fmt_ms,
    fmt_number,
    fmt_percent_change,
    fmt_signed_interval,
    fmt_yes_no,
    frontier_changes,
    frontier_with_repeated_p50,
    latency_markdown,
    latency_rows,
    main_table_markdown,
    main_table_rows,
    markdown_table,
    per_model_rows,
    read_tables,
    repeated_latency,
    truncation_markdown,
)
from tests.paper_data import (
    auc_by_model_frame,
    latency_frame,
    metrics_frame,
    truncation_frame,
)

REFERENCE = ["length"]
NAN = float("nan")


class TestFormatting:
    def test_numbers_and_intervals(self) -> None:
        assert fmt_number(0.78812) == "0.788"
        assert fmt_interval(0.788, 0.746, 0.829) == "0.788 [0.746, 0.829]"

    def test_signed_interval_always_shows_the_sign(self) -> None:
        assert fmt_signed_interval(0.0757, 0.0357, 0.115) == "+0.076 [+0.036, +0.115]"
        assert fmt_signed_interval(-0.0468, -0.1088, 0.0193) == "-0.047 [-0.109, +0.019]"

    @pytest.mark.parametrize(
        "value, expected",
        [(0.0013, "0.0013"), (0.134, "0.134"), (9.7265, "9.73"), (163.07, "163"), (1630.4, "1630")],
    )
    def test_latency_has_three_significant_digits(self, value: float, expected: str) -> None:
        assert fmt_ms(value) == expected

    def test_missing_values_are_a_dash(self) -> None:
        assert fmt_number(NAN) == NOT_AVAILABLE
        assert fmt_interval(NAN, NAN, NAN) == NOT_AVAILABLE
        assert fmt_interval(0.5, 0.4, NAN) == NOT_AVAILABLE
        assert fmt_signed_interval(NAN, 0.0, 0.0) == NOT_AVAILABLE
        assert fmt_ms(NAN) == NOT_AVAILABLE
        assert fmt_yes_no(NAN) == NOT_AVAILABLE

    @pytest.mark.parametrize(
        "value, expected",
        [(-22.9, "-23%"), (31.6, "+32%"), (-0.2, "0%"), (0.0, "0%"), (NAN, NOT_AVAILABLE)],
    )
    def test_percent_change(self, value: float, expected: str) -> None:
        assert fmt_percent_change(value) == expected

    def test_yes_no(self) -> None:
        assert (fmt_yes_no(True), fmt_yes_no(False)) == ("yes", "no")

    def test_markdown_table(self) -> None:
        text = markdown_table(["a", "b"], [["1", "2"], ["3", "4"]])

        assert text.splitlines() == ["| a | b |", "|---|---|", "| 1 | 2 |", "| 3 | 4 |"]


class TestReadTables:
    def _write(self, directory, names: list[str]) -> None:
        for name in names:
            pd.DataFrame({"x": [1]}).to_csv(directory / f"{name}.csv", index=False)

    def test_reads_required_and_optional_tables(self, tmp_path) -> None:
        self._write(tmp_path, ["metrics_summary", "auc_by_model", "latency_summary", "truncation"])

        assert set(read_tables(tmp_path)) == {
            "metrics_summary", "auc_by_model", "latency_summary", "truncation"
        }

    def test_truncation_is_optional(self, tmp_path) -> None:
        self._write(tmp_path, ["metrics_summary", "auc_by_model", "latency_summary"])

        assert "truncation" not in read_tables(tmp_path)

    def test_names_every_missing_required_table(self, tmp_path) -> None:
        self._write(tmp_path, ["metrics_summary"])

        with pytest.raises(FileNotFoundError, match="auc_by_model, latency_summary"):
            read_tables(tmp_path)


class TestRepeatedLatency:
    def test_uses_only_the_requested_sample(self) -> None:
        first = repeated_latency(latency_frame(), "first")
        random_sample = repeated_latency(latency_frame(), "random")

        assert first[("d2", "nli_chunked")]["p50"] == pytest.approx(163.5)
        assert random_sample[("d2", "nli_chunked")]["p50"] == pytest.approx(126.0)
        assert ("d1", "lexical") not in random_sample

    def test_keeps_the_range_and_the_tail(self) -> None:
        entry = repeated_latency(latency_frame())[("d2", "distilbert_indomain")]

        assert entry["p50_min"] < entry["p50"] < entry["p50_max"]
        assert entry["p95"] == pytest.approx(18.6)


class TestFrontier:
    def _status(self, frontier) -> dict[tuple[str, str], str]:
        return {(row["dataset"], row["strategy"]): row["status"] for row in frontier}

    def test_statuses_of_the_fixture(self) -> None:
        frontier = frontier_with_repeated_p50(metrics_frame(), latency_frame(), REFERENCE)
        status = self._status(frontier)

        assert status[("d2", "lexical")] == "frontier"
        assert status[("d2", "distilbert_indomain")] == "frontier"
        assert status[("d2", "nli_chunked")] == "dominated"
        assert status[("d2", "length")] == "reference"

    def test_the_repeated_p50_is_the_latency_used(self) -> None:
        latency = latency_frame()
        mask = (latency["dataset"] == "d2") & (latency["strategy"] == "nli_chunked")
        latency.loc[mask & (latency["sample"] == "first"), "p50_median_ms"] = 5.0

        frontier = frontier_with_repeated_p50(metrics_frame(), latency, REFERENCE)

        row = next(r for r in frontier if r["strategy"] == "nli_chunked")
        assert row["p50_latency_ms"] == 5.0
        assert row["status"] == "frontier"  # faster than distilbert_indomain now

    def test_the_latency_of_the_other_sample_is_not_used(self) -> None:
        frontier = frontier_with_repeated_p50(metrics_frame(), latency_frame(), REFERENCE)

        row = next(r for r in frontier if r["strategy"] == "nli_chunked")
        assert row["p50_latency_ms"] == 163.5

    def test_a_detector_without_repetitions_is_an_error(self) -> None:
        latency = latency_frame()
        latency = latency[latency["strategy"] != "lexical"]

        with pytest.raises(ValueError, match="d1 / lexical"):
            frontier_with_repeated_p50(metrics_frame(), latency, REFERENCE)

    def test_no_change_when_the_repeated_latency_agrees(self) -> None:
        frontier = frontier_with_repeated_p50(metrics_frame(), latency_frame(), REFERENCE)

        assert frontier_changes(metrics_frame(), frontier, REFERENCE) == []

    def test_changes_are_listed(self) -> None:
        latency = latency_frame()
        mask = (latency["strategy"] == "nli_chunked") & (latency["sample"] == "first")
        latency.loc[mask, "p50_median_ms"] = 5.0
        frontier = frontier_with_repeated_p50(metrics_frame(), latency, REFERENCE)

        assert frontier_changes(metrics_frame(), frontier, REFERENCE) == [
            "d2 / nli_chunked: dominated (by distilbert_indomain) -> frontier"
        ]

    def test_a_change_of_dominators_is_a_change_even_if_the_status_is_the_same(self) -> None:
        metrics = metrics_frame()
        # lexical's interval is now above NLI chunked's, so it dominates it as well...
        metrics.loc[(metrics["dataset"] == "d2") & (metrics["strategy"] == "nli_chunked"),
                    "auc_upper"] = 0.60
        latency = latency_frame()
        # ...but only while it is faster; with the repeated p50 it no longer is
        latency.loc[(latency["strategy"] == "lexical") & (latency["dataset"] == "d2"),
                    "p50_median_ms"] = 200.0
        frontier = frontier_with_repeated_p50(metrics, latency, REFERENCE)

        assert frontier_changes(metrics, frontier, REFERENCE) == [
            "d2 / lexical: frontier -> dominated (by distilbert_indomain)",
            "d2 / nli_chunked: dominated (by lexical, distilbert_indomain) "
            "-> dominated (by distilbert_indomain)",
        ]


class TestMainTable:
    def _rows(self, dataset: str = "d2"):
        frontier = frontier_with_repeated_p50(metrics_frame(), latency_frame(), REFERENCE)
        return main_table_rows(metrics_frame(), latency_frame(), frontier, dataset)

    def test_one_row_per_detector_with_the_reference_last(self) -> None:
        assert [r["strategy"] for r in self._rows()] == [
            "lexical", "distilbert_indomain", "nli_chunked", "length"
        ]

    def test_the_reference_is_last_whatever_the_input_order(self) -> None:
        metrics = metrics_frame().iloc[::-1]
        frontier = frontier_with_repeated_p50(metrics, latency_frame(), REFERENCE)

        rows = main_table_rows(metrics, latency_frame(), frontier, "d2")

        assert rows[-1]["strategy"] == "length"

    def test_latency_is_the_repeated_p50(self) -> None:
        row = next(r for r in self._rows() if r["strategy"] == "distilbert_indomain")

        assert row["p50_ms"] == pytest.approx(9.3)
        assert row["p50_min_ms"] < 9.3 < row["p50_max_ms"]

    def test_carries_status_and_dominators(self) -> None:
        row = next(r for r in self._rows() if r["strategy"] == "nli_chunked")

        assert (row["status"], row["dominated_by"]) == ("dominated", "distilbert_indomain")

    def test_only_the_requested_dataset(self) -> None:
        assert {r["strategy"] for r in self._rows("d1")} == {"lexical", "distilbert", "length"}

    def test_markdown_formats_intervals_and_blanks_the_reference_differences(self) -> None:
        text = main_table_markdown(self._rows())
        lines = {line.split("|")[1].strip(): line for line in text.splitlines()[2:]}

        assert "0.800 [0.760, 0.840]" in lines["DistilBERT in-domain"]
        assert "+0.080 [+0.040, +0.120]" in lines["DistilBERT in-domain"]
        assert "| yes |" in lines["DistilBERT in-domain"]
        assert "| no |" in lines["NLI chunked"]
        assert "dominated by DistilBERT in-domain" in lines["NLI chunked"]
        reference_cells = [c.strip() for c in lines["Length (reference)"].split("|")[4:7]]
        assert reference_cells == [NOT_AVAILABLE] * 3

    def test_a_dataset_without_within_model_columns_shows_dashes(self) -> None:
        text = main_table_markdown(self._rows("d1"))
        lexical = next(line for line in text.splitlines() if "Lexical overlap" in line)

        cells = [c.strip() for c in lexical.split("|")]
        assert cells[3] == NOT_AVAILABLE  # within-model AUC
        assert cells[5] == NOT_AVAILABLE  # within-model difference


class TestLatencyTable:
    def test_random_sample_is_compared_with_the_protocol_sample(self) -> None:
        row = next(
            r for r in latency_rows(latency_frame())
            if (r["dataset"], r["strategy"]) == ("d2", "nli_chunked")
        )

        assert row["p50_first_ms"] == pytest.approx(163.5)
        assert row["p50_random_ms"] == pytest.approx(126.0)
        assert row["random_vs_first_pct"] == pytest.approx(100 * (126.0 - 163.5) / 163.5)

    def test_one_row_per_protocol_measurement(self) -> None:
        assert len(latency_rows(latency_frame())) == 7  # the 'random' rows are not extra rows

    def test_without_a_random_sample_the_comparison_is_a_dash(self) -> None:
        rows = latency_rows(latency_frame())
        text = latency_markdown(rows)
        lexical = next(line for line in text.splitlines() if "d1" in line and "Lexical" in line)

        assert pd.isna(next(r for r in rows if r["strategy"] == "lexical")["p50_random_ms"])
        assert lexical.split("|")[5].strip() == NOT_AVAILABLE
        assert lexical.split("|")[6].strip() == NOT_AVAILABLE

    def test_markdown_shows_the_signed_difference(self) -> None:
        text = latency_markdown(latency_rows(latency_frame()))

        assert "-23%" in text


def test_per_model_rows_keep_interpretable_models_of_the_dataset() -> None:
    rows = per_model_rows(auc_by_model_frame(), "d2")

    assert {r["model"] for r in rows} == {"model-a", "model-b"}
    assert per_model_rows(auc_by_model_frame(), "d1") == []


def test_truncation_markdown_lists_only_the_requested_split() -> None:
    text = truncation_markdown(truncation_frame(), split="test")

    assert "284 (31.6%)" in text
    assert "1622" not in text
    assert "291 / 449" in text