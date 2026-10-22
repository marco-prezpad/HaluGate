"""
paper_data.py

Small result tables shaped like the CSV files of results/tables/, shared by
the tests of src/paper_tables.py and src/paper_figures.py.

"d1" has no within-model columns (like HaluEval); "d2" does (like RAGTruth).
On d2 the in-domain DistilBERT dominates NLI chunked (10 ms vs 160 ms, and
an interval entirely above), while lexical overlap, faster, stays on the
frontier.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

import pandas as pd

NAN = float("nan")


def _metrics_row(
    dataset: str, strategy: str, p50: float, auc: tuple[float, float, float],
    versus_length: tuple[float, float, float] | None,
    within: tuple[float, float, float] | None,
    within_versus_length: tuple[float, float, float] | None,
    beats: bool | None,
) -> dict:
    nan3 = (NAN, NAN, NAN)
    versus_length = versus_length or nan3
    within = within or nan3
    within_versus_length = within_versus_length or nan3
    return {
        "dataset": dataset, "strategy": strategy, "precision": 0.3, "recall": 0.75,
        "f1_score": 0.43, "auc_roc": auc[0], "auc_lower": auc[1], "auc_upper": auc[2],
        "auc_minus_length": versus_length[0], "auc_minus_length_lower": versus_length[1],
        "auc_minus_length_upper": versus_length[2], "within_model_auc": within[0],
        "within_model_auc_lower": within[1], "within_model_auc_upper": within[2],
        "within_model_minus_length": within_versus_length[0],
        "within_model_minus_length_lower": within_versus_length[1],
        "within_model_minus_length_upper": within_versus_length[2],
        "beats_length_within_model": NAN if beats is None else beats,
        "p50_latency_ms": p50,
    }


def metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _metrics_row("d1", "lexical", 0.02, (0.90, 0.88, 0.92), (-0.07, -0.09, -0.05),
                         None, None, None),
            _metrics_row("d1", "distilbert", 7.0, (0.99, 0.98, 1.00), (0.02, 0.01, 0.03),
                         None, None, None),
            _metrics_row("d1", "length", 0.001, (0.97, 0.96, 0.98), (0.0, 0.0, 0.0),
                         None, None, None),
            _metrics_row("d2", "lexical", 0.13, (0.70, 0.65, 0.75), (0.03, -0.02, 0.08),
                         (0.67, 0.62, 0.72), (0.04, -0.04, 0.12), False),
            _metrics_row("d2", "distilbert_indomain", 9.7, (0.80, 0.76, 0.84),
                         (0.08, 0.04, 0.12), (0.70, 0.64, 0.75), (0.07, 0.02, 0.12), True),
            _metrics_row("d2", "nli_chunked", 163.0, (0.66, 0.62, 0.70), (-0.05, -0.11, 0.02),
                         (0.59, 0.53, 0.64), (-0.04, -0.12, 0.04), False),
            _metrics_row("d2", "length", 0.016, (0.71, 0.66, 0.76), (0.0, 0.0, 0.0),
                         (0.63, 0.57, 0.68), (0.0, 0.0, 0.0), False),
        ]
    )


def _latency_row(dataset: str, strategy: str, sample: str, p50: float, spread: float) -> dict:
    return {
        "dataset": dataset, "strategy": strategy, "sample": sample, "runs": 5,
        "p50_median_ms": p50, "p50_min_ms": p50 * (1 - spread), "p50_max_ms": p50 * (1 + spread),
        "p50_spread_pct": 200 * spread, "p95_median_ms": p50 * 2, "p95_min_ms": p50,
        "p95_max_ms": p50 * 3, "p99_median_ms": p50 * 3,
    }


def latency_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _latency_row("d1", "lexical", "first", 0.022, 0.1),
            _latency_row("d1", "distilbert", "first", 7.6, 0.03),
            _latency_row("d1", "length", "first", 0.0013, 0.2),
            _latency_row("d2", "lexical", "first", 0.129, 0.2),
            _latency_row("d2", "distilbert_indomain", "first", 9.3, 0.1),
            _latency_row("d2", "nli_chunked", "first", 163.5, 0.01),
            _latency_row("d2", "length", "first", 0.007, 0.3),
            _latency_row("d2", "nli_chunked", "random", 126.0, 0.01),
            _latency_row("d2", "distilbert_indomain", "random", 8.9, 0.02),
        ]
    )


def auc_by_model_frame() -> pd.DataFrame:
    rows = []
    for model, positives, interpretable, shift in (
        ("model-a", 36, True, 0.0), ("model-b", 31, True, 0.05), ("model-c", 1, False, -0.3),
    ):
        for strategy, auc in (("lexical", 0.70), ("distilbert_indomain", 0.75),
                              ("nli_chunked", 0.60), ("length", 0.65)):
            rows.append({
                "dataset": "d2", "strategy": strategy, "model": model, "n": 150,
                "positives": positives, "auc": auc + shift, "auc_lower": auc + shift - 0.08,
                "auc_upper": auc + shift + 0.08, "interpretable": interpretable,
            })
    return pd.DataFrame(rows)


def truncation_frame() -> pd.DataFrame:
    base = {"response_over_limit": 0, "context_tokens_median": 291.0, "context_tokens_p90": 449.0,
            "context_kept_pct_mean": 92.7}
    return pd.DataFrame(
        [
            {"dataset": "d2", "strategy": "distilbert_indomain", "split": "test",
             "group": "all", "n": 900, "truncated": 284, "truncated_pct": 31.6, **base},
            {"dataset": "d2", "strategy": "distilbert_indomain", "split": "train",
             "group": "all", "n": 4284, "truncated": 1622, "truncated_pct": 37.9, **base},
        ]
    )