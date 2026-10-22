"""
paper_figures.py

The figures of the paper, drawn from the result tables (see
src/paper_tables.py): the latency-AUC frontier, the AUC difference with the
response-length baseline, and the AUC per generating model.

Colour follows the detector family and never its rank: lexical overlap is
blue, the DistilBERT arms orange, the NLI arms aqua and the length baseline
gray. The three hues pass scripts/validate_palette.js with ``--pairs all``;
every point is also labelled or listed in a legend, so no identity relies on
colour alone. Fill carries the frontier status (filled = on the frontier,
hollow = dominated) and the marker shape the variant of a family.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

from src.paper_tables import REFERENCE, display_name, repeated_latency  # noqa: E402

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e3e2de"
SURFACE = "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

COLORS = {
    "lexical": BLUE,
    "distilbert": ORANGE,
    "distilbert_transfer": ORANGE,
    "distilbert_indomain": ORANGE,
    "nli": AQUA,
    "nli_chunked": AQUA,
    REFERENCE: MUTED,
}
MARKERS = {
    "lexical": "o",
    "distilbert": "o",
    "distilbert_transfer": "s",
    "distilbert_indomain": "o",
    "nli": "^",
    "nli_chunked": "^",
    REFERENCE: "D",
}
# Drawn hollow in the per-model plot, where fill does not carry frontier status.
HOLLOW = ("distilbert_transfer",)
MARKER_SIZE = 6.5
LINE_WIDTH = 1.2


def _style(axis: Any) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=8)


def _new_figure(columns: int, size: tuple[float, float]) -> tuple[Figure, Any]:
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    figure, axes = plt.subplots(1, columns, figsize=size, facecolor=SURFACE, squeeze=False)
    for axis in axes[0]:
        _style(axis)
    return figure, axes[0]


def _point(axis: Any, x: float, y: float, strategy: str, filled: bool, **errors: Any) -> None:
    """One detector: marker with error bars, tagged with the strategy name
    (``gid``) so tests can find it."""
    color = COLORS.get(strategy, MUTED)
    container = axis.errorbar(
        x, y, **errors, fmt=MARKERS.get(strategy, "o"), color=color, ecolor=color,
        markersize=MARKER_SIZE, markerfacecolor=color if filled else SURFACE,
        markeredgecolor=color, markeredgewidth=1.4, elinewidth=LINE_WIDTH, capsize=2.5,
        zorder=3,
    )
    container.lines[0].set_gid(strategy)


def plot_frontier(
    metrics: pd.DataFrame,
    latency_summary: pd.DataFrame,
    frontier: list[dict[str, Any]],
    reference: str = REFERENCE,
) -> Figure:
    """Three panels: pooled AUC on the first dataset and on the second, and the
    within-model AUC on the second. Latency is the median of the repeated p50
    (bars: range of the repetitions); AUC bars are 95% bootstrap intervals.
    """
    datasets = list(dict.fromkeys(metrics["dataset"]))
    panels = [(datasets[0], "auc"), (datasets[-1], "auc"), (datasets[-1], "within_model_auc")]
    titles = [
        f"(a) {datasets[0]}, pooled AUC",
        f"(b) {datasets[-1]}, pooled AUC",
        f"(c) {datasets[-1]}, within-model AUC (fill as in b)",
    ]
    repeated = repeated_latency(latency_summary)
    status = {(row["dataset"], row["strategy"]): row["status"] for row in frontier}

    figure, axes = _new_figure(3, (10.2, 3.5))
    for axis, (dataset, kind), title in zip(axes, panels, titles):
        value_column = "auc_roc" if kind == "auc" else "within_model_auc"
        lower_column = "auc_lower" if kind == "auc" else "within_model_auc_lower"
        upper_column = "auc_upper" if kind == "auc" else "within_model_auc_upper"
        rows = metrics[metrics["dataset"] == dataset]

        latencies = []
        for row in rows.itertuples():
            value = getattr(row, value_column)
            lower, upper = getattr(row, lower_column), getattr(row, upper_column)
            if row.strategy == reference:
                axis.axhspan(lower, upper, color=MUTED, alpha=0.12, linewidth=0)
                axis.axhline(value, color=MUTED, linewidth=1, linestyle="--")
                first = axis is axes[0]
                axis.text(0.99 if first else 0.5, value, "length (reference)",
                          transform=axis.get_yaxis_transform(), ha="right" if first else "center",
                          va="top" if first else "bottom", color=MUTED, fontsize=7.5)
                continue
            latency = repeated[(dataset, row.strategy)]
            latencies.append(latency["p50"])
            _point(
                axis, latency["p50"], value, row.strategy,
                filled=status[(dataset, row.strategy)] == "frontier",
                xerr=[[latency["p50"] - latency["p50_min"]], [latency["p50_max"] - latency["p50"]]],
                yerr=[[value - lower], [upper - value]],
            )
            axis.annotate(display_name(row.strategy), (latency["p50"], value),
                          textcoords="offset points", xytext=(7, 6), fontsize=8, color=INK)

        axis.set_xscale("log")
        axis.set_xlim(min(latencies) / 4, max(latencies) * 8)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda tick, _: f"{tick:g}"))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_title(title, loc="left", color=INK)
        axis.set_xlabel("p50 latency per response (ms, median of repetitions)")
        axis.set_ylabel("AUC (test)")

    handles = [
        Line2D([], [], marker="o", linestyle="", color=MUTED, markerfacecolor=MUTED,
               markersize=MARKER_SIZE, label="on the frontier"),
        Line2D([], [], marker="o", linestyle="", color=MUTED, markerfacecolor=SURFACE,
               markeredgewidth=1.4, markersize=MARKER_SIZE, label="dominated"),
        Line2D([], [], color=MUTED, linestyle="--", label="length baseline, 95% CI band"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return figure


def plot_vs_length(metrics: pd.DataFrame, reference: str = REFERENCE) -> Figure:
    """AUC difference with the length baseline, pooled and within-model, with
    the paired bootstrap interval. An interval that excludes 0 beats length."""
    datasets = list(dict.fromkeys(metrics["dataset"]))
    figure, axes = _new_figure(len(datasets), (4.2 * len(datasets) + 1.5, 3.4))

    for axis, dataset in zip(axes, datasets):
        rows = [r for r in metrics[metrics["dataset"] == dataset].itertuples()
                if r.strategy != reference]
        for index, row in enumerate(rows):
            y = len(rows) - 1 - index
            color = COLORS.get(row.strategy, MUTED)
            for offset, prefix, marker, filled in (
                (0.16, "auc_minus_length", "o", True),
                (-0.16, "within_model_minus_length", "D", False),
            ):
                value = getattr(row, prefix)
                if pd.isna(value):
                    continue
                lower, upper = getattr(row, f"{prefix}_lower"), getattr(row, f"{prefix}_upper")
                container = axis.errorbar(
                    value, y + offset, xerr=[[value - lower], [upper - value]], fmt=marker,
                    color=color, ecolor=color, markersize=MARKER_SIZE,
                    markerfacecolor=color if filled else SURFACE, markeredgewidth=1.4,
                    elinewidth=LINE_WIDTH, capsize=2.5, zorder=3,
                )
                container.lines[0].set_gid(f"{row.strategy}:{prefix}")
        axis.axvline(0, color=MUTED, linewidth=1, linestyle="--")
        axis.set_yticks(range(len(rows)))
        axis.set_yticklabels([display_name(r.strategy) for r in reversed(rows)], color=INK)
        axis.set_ylim(-0.7, len(rows) - 0.3)
        axis.grid(axis="y", visible=False)
        axis.grid(axis="x", color=GRID, linewidth=0.6)
        axis.set_title(f"{'(a)' if dataset == datasets[0] else '(b)'} {dataset}", loc="left",
                       color=INK)
        axis.set_xlabel("AUC − AUC of length (95% paired bootstrap CI)")

    handles = [
        Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=MARKER_SIZE,
               label="pooled AUC"),
        Line2D([], [], marker="D", linestyle="", color=MUTED, markerfacecolor=SURFACE,
               markeredgewidth=1.4, markersize=MARKER_SIZE, label="within-model AUC"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    return figure


def plot_per_model(per_model: list[dict[str, Any]], dataset: str) -> Figure:
    """AUC of every detector for each generating model with enough positives
    (n positives in the row label); the dashed line is chance (0.5)."""
    models = list(dict.fromkeys(row["model"] for row in per_model))
    strategies = list(dict.fromkeys(row["strategy"] for row in per_model))
    strategies.sort(key=lambda name: name == REFERENCE)
    positives = {row["model"]: row["positives"] for row in per_model}
    spacing = 0.15

    figure, axes = _new_figure(1, (6.4, 0.9 * len(models) + 1.8))
    axis = axes[0]
    for model_index, model in enumerate(models):
        y = len(models) - 1 - model_index
        for strategy_index, strategy in enumerate(strategies):
            row = next(r for r in per_model if r["model"] == model and r["strategy"] == strategy)
            offset = (strategy_index - (len(strategies) - 1) / 2) * spacing
            color = COLORS.get(strategy, MUTED)
            container = axis.errorbar(
                row["auc"], y - offset, xerr=[[row["auc"] - row["auc_lower"]],
                                              [row["auc_upper"] - row["auc"]]],
                fmt=MARKERS.get(strategy, "o"), color=color, ecolor=color,
                markersize=MARKER_SIZE - 1, markeredgewidth=1.3, elinewidth=1.0, capsize=2,
                markerfacecolor=SURFACE if strategy in HOLLOW else color, zorder=3,
            )
            container.lines[0].set_gid(f"{strategy}:{model}")
    axis.axvline(0.5, color=MUTED, linewidth=1, linestyle="--")
    axis.set_yticks(range(len(models)))
    axis.set_yticklabels([f"{m}\n({positives[m]} positives)" for m in reversed(models)], color=INK)
    axis.set_ylim(-0.6, len(models) - 0.4)
    axis.grid(axis="y", visible=False)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_xlabel("AUC (test, 95% bootstrap CI); dashed line = chance")
    axis.set_title(f"{dataset}: AUC per generating model", loc="left", color=INK)

    handles = [
        Line2D([], [], marker=MARKERS.get(s, "o"), linestyle="", color=COLORS.get(s, MUTED),
               markerfacecolor=SURFACE if s in HOLLOW else COLORS.get(s, MUTED),
               markeredgewidth=1.3, markersize=MARKER_SIZE - 1, label=display_name(s))
        for s in strategies
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0.13, 1, 1))
    return figure


def save_figure(figure: Figure, base_path: Path) -> list[Path]:
    """Write ``base_path`` as PNG (200 dpi) and PDF; returns the two paths."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    paths = [base_path.with_suffix(".png"), base_path.with_suffix(".pdf")]
    for path in paths:
        figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return paths