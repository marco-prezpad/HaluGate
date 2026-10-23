"""
test_paper_figures.py

Covers what the figures promise rather than how they look: one panel per
dataset or view, the colour of each detector family, filled markers for the
detectors on the frontier and hollow ones for the dominated, the length
baseline drawn as a reference line, and the files that are written.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from src.paper_figures import (
    COLORS,
    SURFACE,
    plot_frontier,
    plot_per_model,
    plot_vs_length,
    save_figure,
)
from src.paper_tables import frontier_with_repeated_p50, per_model_rows
from tests.paper_data import auc_by_model_frame, latency_frame, metrics_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _markers(figure, gid: str):
    return [line for axis in figure.axes for line in axis.lines if line.get_gid() == gid]


def _frontier_figure():
    frontier = frontier_with_repeated_p50(metrics_frame(), latency_frame(), ["length"])
    return plot_frontier(metrics_frame(), latency_frame(), frontier)


def test_the_palette_is_the_validated_one() -> None:
    # blue, orange, aqua: validated with scripts/validate_palette.js --pairs all
    assert {COLORS["lexical"], COLORS["distilbert_indomain"], COLORS["nli_chunked"]} == {
        "#2a78d6", "#eb6834", "#1baf7a",
    }
    assert COLORS["distilbert"] == COLORS["distilbert_transfer"] == COLORS["distilbert_indomain"]
    assert COLORS["nli"] == COLORS["nli_chunked"]


class TestFrontierFigure:
    def test_has_three_panels_with_a_log_latency_axis(self) -> None:
        figure = _frontier_figure()

        assert len(figure.axes) == 3
        assert all(axis.get_xscale() == "log" for axis in figure.axes)

    def test_frontier_detectors_are_filled_and_dominated_ones_hollow(self) -> None:
        figure = _frontier_figure()

        (indomain,) = _markers(figure, "distilbert_indomain")[:1]
        (nli_chunked,) = _markers(figure, "nli_chunked")[:1]
        assert to_hex(indomain.get_markerfacecolor()) == COLORS["distilbert_indomain"]
        assert to_hex(nli_chunked.get_markerfacecolor()) == SURFACE

    def test_markers_take_the_colour_of_their_family(self) -> None:
        figure = _frontier_figure()

        for strategy in ("lexical", "distilbert_indomain", "nli_chunked"):
            assert all(
                to_hex(line.get_color()) == COLORS[strategy] for line in _markers(figure, strategy)
            )

    def test_each_detector_appears_where_it_was_measured(self) -> None:
        figure = _frontier_figure()

        # lexical: panels (a) and (b)/(c) of the second dataset; the reference is not a marker
        assert len(_markers(figure, "lexical")) == 3
        assert len(_markers(figure, "distilbert_indomain")) == 2
        assert len(_markers(figure, "distilbert")) == 1
        assert _markers(figure, "length") == []

    def test_x_position_is_the_repeated_p50_and_y_the_auc(self) -> None:
        figure = _frontier_figure()

        panel_b = figure.axes[1]
        (line,) = [ln for ln in panel_b.lines if ln.get_gid() == "distilbert_indomain"]
        assert line.get_xdata()[0] == pytest.approx(9.3)
        assert line.get_ydata()[0] == pytest.approx(0.80)

    def test_the_within_model_panel_plots_the_within_model_auc(self) -> None:
        figure = _frontier_figure()

        (line,) = [ln for ln in figure.axes[2].lines if ln.get_gid() == "distilbert_indomain"]
        assert line.get_ydata()[0] == pytest.approx(0.70)

    def test_the_length_baseline_is_a_reference_line_with_its_interval_band(self) -> None:
        figure = _frontier_figure()

        panel_b = figure.axes[1]
        dashed = [ln for ln in panel_b.lines if ln.get_linestyle() == "--"]
        assert [ln.get_ydata()[0] for ln in dashed] == [pytest.approx(0.71)]

    def test_has_a_legend_for_fill_and_reference(self) -> None:
        legend = _frontier_figure().legends[0]

        assert [t.get_text() for t in legend.get_texts()] == [
            "on the frontier", "dominated", "length baseline (dashed) and its 95% CI (band)"
        ]


class TestVsLengthFigure:
    def test_has_one_panel_per_dataset(self) -> None:
        assert len(plot_vs_length(metrics_frame()).axes) == 2

    def test_the_reference_is_not_a_row_and_zero_is_marked(self) -> None:
        figure = plot_vs_length(metrics_frame())

        labels = [t.get_text() for t in figure.axes[1].get_yticklabels()]
        assert "Length (reference)" not in labels
        assert len(labels) == 3
        assert any(ln.get_xdata()[0] == 0 for ln in figure.axes[1].lines
                   if ln.get_linestyle() == "--")

    def test_within_model_markers_exist_only_where_the_column_does(self) -> None:
        figure = plot_vs_length(metrics_frame())

        within = "within_model_minus_length"
        assert _markers(figure, f"lexical:{within}") != []
        first_panel = [ln.get_gid() or "" for ln in figure.axes[0].lines]
        assert not any(gid.endswith(within) for gid in first_panel)

    def test_the_marker_sits_at_the_difference(self) -> None:
        figure = plot_vs_length(metrics_frame())

        gid = "nli_chunked:auc_minus_length"
        (line,) = [ln for ln in figure.axes[1].lines if ln.get_gid() == gid]
        assert line.get_xdata()[0] == pytest.approx(-0.05)

    def test_has_a_legend_for_the_two_metrics(self) -> None:
        legend = plot_vs_length(metrics_frame()).legends[0]

        assert [t.get_text() for t in legend.get_texts()] == ["pooled AUC", "within-model AUC"]


class TestPerModelFigure:
    def _figure(self):
        return plot_per_model(per_model_rows(auc_by_model_frame(), "d2"), "d2")

    def test_one_marker_per_model_and_detector(self) -> None:
        figure = self._figure()

        markers = [ln for ln in figure.axes[0].lines if ":" in (ln.get_gid() or "")]
        assert len(markers) == 2 * 4  # two interpretable models, four detectors

    def test_non_interpretable_models_are_absent(self) -> None:
        labels = " ".join(t.get_text() for t in self._figure().axes[0].get_yticklabels())

        assert "model-c" not in labels
        assert "model-a\n(36 positives)" in labels

    def test_chance_is_marked(self) -> None:
        figure = self._figure()

        assert any(ln.get_xdata()[0] == 0.5 for ln in figure.axes[0].lines
                   if ln.get_linestyle() == "--")

    def test_the_legend_lists_every_detector_with_the_reference_last(self) -> None:
        rows = per_model_rows(auc_by_model_frame(), "d2")
        rows = sorted(rows, key=lambda row: row["strategy"] != "length")  # reference first
        names = [t.get_text() for t in plot_per_model(rows, "d2").legends[0].get_texts()]

        assert names == ["Lexical overlap", "DistilBERT in-domain", "NLI chunked",
                         "Length (reference)"]


def test_save_figure_writes_png_and_pdf_and_closes_the_figure(tmp_path) -> None:
    figure = plot_vs_length(metrics_frame())
    number = figure.number

    paths = save_figure(figure, tmp_path / "out" / "vs_length")

    assert [p.suffix for p in paths] == [".png", ".pdf"]
    assert all(p.is_file() and p.stat().st_size > 1000 for p in paths)
    assert not plt.fignum_exists(number)