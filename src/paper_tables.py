"""
paper_tables.py

The tables of the paper, built from the CSV files that evaluate.py,
latency_repeats.py and truncation_report.py already wrote. Nothing is
recomputed from scores and no model runs: every number comes from a result
that exists, so the tables cannot disagree with it.

The one derived quantity is the frontier computed with the median of the
repeated p50 latencies (the amendment on repeated latency measurements)
instead of the single p50 of evaluate.py. ``frontier_changes`` lists every
detector whose status differs between the two.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.reporting import frontier_table

REFERENCE = "length"
PRIMARY_SAMPLE = "first"
NOT_AVAILABLE = "–"
REQUIRED_TABLES = ("metrics_summary", "auc_by_model", "latency_summary")
OPTIONAL_TABLES = ("truncation",)

DISPLAY_NAMES = {
    "lexical": "Lexical overlap",
    "distilbert": "DistilBERT (HaluEval)",
    "distilbert_transfer": "DistilBERT transfer",
    "distilbert_indomain": "DistilBERT in-domain",
    "nli": "NLI",
    "nli_chunked": "NLI chunked",
    "length": "Length (reference)",
}


def read_tables(tables_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the result tables; the truncation table is optional.

    Raises:
        FileNotFoundError: naming every required table that is missing.
    """
    missing = [name for name in REQUIRED_TABLES if not (tables_dir / f"{name}.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"missing tables in {tables_dir}: {', '.join(missing)}")

    tables = {name: pd.read_csv(tables_dir / f"{name}.csv") for name in REQUIRED_TABLES}
    for name in OPTIONAL_TABLES:
        if (tables_dir / f"{name}.csv").is_file():
            tables[name] = pd.read_csv(tables_dir / f"{name}.csv")
    return tables


def display_name(strategy: str) -> str:
    return DISPLAY_NAMES.get(strategy, strategy)


# --- number formatting -------------------------------------------------------


def fmt_number(value: Any, digits: int = 3) -> str:
    return NOT_AVAILABLE if pd.isna(value) else f"{value:.{digits}f}"


def fmt_interval(value: Any, lower: Any, upper: Any, digits: int = 3) -> str:
    if pd.isna(value) or pd.isna(lower) or pd.isna(upper):
        return NOT_AVAILABLE
    return f"{value:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def fmt_signed_interval(value: Any, lower: Any, upper: Any, digits: int = 3) -> str:
    if pd.isna(value) or pd.isna(lower) or pd.isna(upper):
        return NOT_AVAILABLE
    return f"{value:+.{digits}f} [{lower:+.{digits}f}, {upper:+.{digits}f}]"


def fmt_ms(value: Any) -> str:
    """Latency with three significant digits (0.0013, 0.134, 9.73, 163, 1630)."""
    if pd.isna(value):
        return NOT_AVAILABLE
    return f"{value:.0f}" if value >= 1000 else f"{value:.3g}"


def fmt_percent_change(value: Any) -> str:
    """Signed whole percent; a change that rounds to zero is "0%", never "-0%"."""
    if pd.isna(value):
        return NOT_AVAILABLE
    rounded = round(value)
    return f"{rounded:+d}%" if rounded else "0%"


def fmt_yes_no(value: Any) -> str:
    if pd.isna(value):
        return NOT_AVAILABLE
    return "yes" if bool(value) else "no"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# --- latency and frontier ----------------------------------------------------


def repeated_latency(
    latency_summary: pd.DataFrame, sample: str = PRIMARY_SAMPLE
) -> dict[tuple[str, str], dict[str, float]]:
    """Median and range of the repeated p50 (and the median p95 and p99) of
    every (dataset, strategy) measured on ``sample``.
    """
    rows = latency_summary[latency_summary["sample"] == sample]
    return {
        (row.dataset, row.strategy): {
            "p50": row.p50_median_ms,
            "p50_min": row.p50_min_ms,
            "p50_max": row.p50_max_ms,
            "p95": row.p95_median_ms,
            "p99": row.p99_median_ms,
            "runs": row.runs,
        }
        for row in rows.itertuples()
    }


def frontier_with_repeated_p50(
    metrics: pd.DataFrame,
    latency_summary: pd.DataFrame,
    reference_strategies: list[str],
    sample: str = PRIMARY_SAMPLE,
) -> list[dict[str, Any]]:
    """The frontier table with the median of the repeated p50 as latency.

    Raises:
        ValueError: if a (dataset, strategy) of ``metrics`` has no repeated
            measurement on ``sample``.
    """
    repeated = repeated_latency(latency_summary, sample)
    rows = metrics.astype(object).where(metrics.notna(), None).to_dict("records")
    for row in rows:
        key = (row["dataset"], row["strategy"])
        if key not in repeated:
            raise ValueError(f"no '{sample}' latency repetitions for {key[0]} / {key[1]}")
        row["p50_latency_ms"] = repeated[key]["p50"]
    return frontier_table(rows, reference_strategies)


def frontier_changes(
    metrics: pd.DataFrame, repeated_frontier: list[dict[str, Any]], reference_strategies: list[str]
) -> list[str]:
    """Detectors whose frontier status with the repeated latency differs from
    the one with the single p50 of evaluate.py."""
    rows = metrics.astype(object).where(metrics.notna(), None).to_dict("records")
    original = {
        (row["dataset"], row["strategy"]): (row["status"], row["dominated_by"])
        for row in frontier_table(rows, reference_strategies)
    }

    def describe(status: str, dominated_by: str) -> str:
        return f"{status} (by {dominated_by.replace(';', ', ')})" if dominated_by else status

    changes = []
    for row in repeated_frontier:
        key = (row["dataset"], row["strategy"])
        now = (row["status"], row["dominated_by"])
        if original[key] != now:
            changes.append(f"{key[0]} / {key[1]}: {describe(*original[key])} -> {describe(*now)}")
    return changes


# --- tables ------------------------------------------------------------------


def main_table_rows(
    metrics: pd.DataFrame,
    latency_summary: pd.DataFrame,
    frontier: list[dict[str, Any]],
    dataset: str,
    sample: str = PRIMARY_SAMPLE,
) -> list[dict[str, Any]]:
    """One row per detector of ``dataset``, the reference last."""
    repeated = repeated_latency(latency_summary, sample)
    status = {(r["dataset"], r["strategy"]): r for r in frontier}

    rows: list[dict[str, Any]] = []
    for record in metrics[metrics["dataset"] == dataset].to_dict("records"):
        key = (dataset, record["strategy"])
        latency = repeated[key]
        rows.append(
            {
                "strategy": record["strategy"],
                "auc": record["auc_roc"],
                "auc_lower": record["auc_lower"],
                "auc_upper": record["auc_upper"],
                "within_model_auc": record["within_model_auc"],
                "within_model_auc_lower": record["within_model_auc_lower"],
                "within_model_auc_upper": record["within_model_auc_upper"],
                "auc_minus_length": record["auc_minus_length"],
                "auc_minus_length_lower": record["auc_minus_length_lower"],
                "auc_minus_length_upper": record["auc_minus_length_upper"],
                "within_model_minus_length": record["within_model_minus_length"],
                "within_model_minus_length_lower": record["within_model_minus_length_lower"],
                "within_model_minus_length_upper": record["within_model_minus_length_upper"],
                "beats_length_within_model": record["beats_length_within_model"],
                "p50_ms": latency["p50"],
                "p50_min_ms": latency["p50_min"],
                "p50_max_ms": latency["p50_max"],
                "p95_ms": latency["p95"],
                "precision": record["precision"],
                "recall": record["recall"],
                "f1": record["f1_score"],
                "status": status[key]["status"],
                "dominated_by": status[key]["dominated_by"],
            }
        )
    return sorted(rows, key=lambda row: row["strategy"] == REFERENCE)


def main_table_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Detector",
        "AUC [95% CI]",
        "Within-model AUC [95% CI]",
        "AUC − length [95% CI]",
        "Within-model − length [95% CI]",
        "Beats length (within-model)",
        "p50 ms (range of runs)",
        "p95 ms",
        "Precision / recall / F1",
        "Frontier",
    ]
    body = []
    for row in rows:
        is_reference = row["status"] == "reference"
        dominators = ", ".join(
            display_name(name) for name in row["dominated_by"].split(";") if name
        )
        frontier = f"dominated by {dominators}" if dominators else row["status"]
        body.append(
            [
                display_name(row["strategy"]),
                fmt_interval(row["auc"], row["auc_lower"], row["auc_upper"]),
                fmt_interval(
                    row["within_model_auc"],
                    row["within_model_auc_lower"],
                    row["within_model_auc_upper"],
                ),
                NOT_AVAILABLE
                if is_reference
                else fmt_signed_interval(
                    row["auc_minus_length"],
                    row["auc_minus_length_lower"],
                    row["auc_minus_length_upper"],
                ),
                NOT_AVAILABLE
                if is_reference
                else fmt_signed_interval(
                    row["within_model_minus_length"],
                    row["within_model_minus_length_lower"],
                    row["within_model_minus_length_upper"],
                ),
                NOT_AVAILABLE if is_reference else fmt_yes_no(row["beats_length_within_model"]),
                f"{fmt_ms(row['p50_ms'])} "
                f"({fmt_ms(row['p50_min_ms'])}–{fmt_ms(row['p50_max_ms'])})",
                fmt_ms(row["p95_ms"]),
                f"{fmt_number(row['precision'], 2)} / {fmt_number(row['recall'], 2)}"
                f" / {fmt_number(row['f1'], 2)}",
                frontier,
            ]
        )
    return markdown_table(headers, body)


def latency_rows(latency_summary: pd.DataFrame) -> list[dict[str, Any]]:
    """Latency per (dataset, strategy): the protocol sample (``first``), the
    random sample as a sensitivity check, and the tail percentiles."""
    first = latency_summary[latency_summary["sample"] == PRIMARY_SAMPLE]
    random_sample = {
        (row.dataset, row.strategy): row.p50_median_ms
        for row in latency_summary[latency_summary["sample"] == "random"].itertuples()
    }
    rows = []
    for row in first.itertuples():
        random_p50 = random_sample.get((row.dataset, row.strategy), float("nan"))
        rows.append(
            {
                "dataset": row.dataset,
                "strategy": row.strategy,
                "runs": row.runs,
                "p50_first_ms": row.p50_median_ms,
                "p50_first_min_ms": row.p50_min_ms,
                "p50_first_max_ms": row.p50_max_ms,
                "p95_first_ms": row.p95_median_ms,
                "p99_first_ms": row.p99_median_ms,
                "p50_random_ms": random_p50,
                "random_vs_first_pct": (
                    100.0 * (random_p50 - row.p50_median_ms) / row.p50_median_ms
                ),
            }
        )
    return rows


def latency_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Dataset", "Detector", "Runs", "p50 ms, first 200 (range)", "p50 ms, random 200",
        "Random vs first", "p95 ms", "p99 ms",
    ]
    body = [
        [
            row["dataset"],
            display_name(row["strategy"]),
            str(row["runs"]),
            f"{fmt_ms(row['p50_first_ms'])} "
            f"({fmt_ms(row['p50_first_min_ms'])}–{fmt_ms(row['p50_first_max_ms'])})",
            fmt_ms(row["p50_random_ms"]),
            fmt_percent_change(row["random_vs_first_pct"]),
            fmt_ms(row["p95_first_ms"]),
            fmt_ms(row["p99_first_ms"]),
        ]
        for row in rows
    ]
    return markdown_table(headers, body)


def per_model_rows(auc_by_model: pd.DataFrame, dataset: str) -> list[dict[str, Any]]:
    """Per-model AUC rows of ``dataset`` for the models with enough positives
    to be interpreted (the flag of auc_by_model.csv)."""
    selected = auc_by_model[
        (auc_by_model["dataset"] == dataset) & (auc_by_model["interpretable"].astype(bool))
    ]
    return selected.to_dict("records")


def truncation_markdown(truncation: pd.DataFrame, split: str = "test") -> str:
    headers = [
        "Dataset", "Arm", "Group", "Pairs", "Truncated", "Context tokens median / p90",
        "Context kept (mean)",
    ]
    body = [
        [
            row.dataset,
            display_name(row.strategy),
            row.group,
            str(row.n),
            f"{row.truncated} ({row.truncated_pct:.1f}%)",
            f"{row.context_tokens_median:.0f} / {row.context_tokens_p90:.0f}",
            f"{row.context_kept_pct_mean:.1f}%",
        ]
        for row in truncation[truncation["split"] == split].itertuples()
    ]
    return markdown_table(headers, body)