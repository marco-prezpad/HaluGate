#!/usr/bin/env python3
"""
make_paper_outputs.py

Builds the figures and tables of the paper from the result tables already
written under results/tables/ (no model runs, no score is read):

- results/tables/paper_main_<dataset>.csv   main table per dataset
- results/tables/paper_latency.csv          latency, protocol vs random sample
- results/tables/paper_frontier.csv         frontier with the median repeated p50
- results/tables/paper_tables.md            all tables formatted for the text
- results/figures/frontier.{png,pdf}        latency-AUC frontier, pooled + within-model
- results/figures/vs_length.{png,pdf}       AUC difference with the length baseline
- results/figures/per_model.{png,pdf}       AUC per generating model

Prints whether the frontier changes when the median of the repeated p50 is
used instead of the single p50 of evaluate.py.

Usage:
    python scripts/make_paper_outputs.py [--tables-dir results/tables]

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paper_figures import (  # noqa: E402
    plot_frontier,
    plot_per_model,
    plot_vs_length,
    save_figure,
)
from src.paper_tables import (  # noqa: E402
    REFERENCE,
    frontier_changes,
    frontier_with_repeated_p50,
    latency_markdown,
    latency_rows,
    main_table_markdown,
    main_table_rows,
    per_model_rows,
    read_tables,
    truncation_markdown,
)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Figures and tables of the paper.")
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--reference", default=REFERENCE)
    args = parser.parse_args()

    tables = read_tables(args.tables_dir)
    metrics, latency = tables["metrics_summary"], tables["latency_summary"]
    reference = [args.reference]

    frontier = frontier_with_repeated_p50(metrics, latency, reference)
    write_csv(frontier, args.tables_dir / "paper_frontier.csv")
    changes = frontier_changes(metrics, frontier, reference)
    print("Frontier with the median of the repeated p50 vs evaluate.py's single p50:")
    print("  unchanged" if not changes else "\n".join(f"  CHANGED {line}" for line in changes))

    sections = []
    for dataset in dict.fromkeys(metrics["dataset"]):
        rows = main_table_rows(metrics, latency, frontier, dataset)
        write_csv(rows, args.tables_dir / f"paper_main_{dataset}.csv")
        sections.append(f"## Main table: {dataset}\n\n{main_table_markdown(rows)}")

    latency_table = latency_rows(latency)
    write_csv(latency_table, args.tables_dir / "paper_latency.csv")
    sections.append(f"## Latency\n\n{latency_markdown(latency_table)}")
    if "truncation" in tables:
        truncation = truncation_markdown(tables["truncation"])
        sections.append(f"## Truncation (test split)\n\n{truncation}")
    markdown_path = args.tables_dir / "paper_tables.md"
    markdown_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"tables written to {markdown_path} and paper_*.csv")

    last_dataset = list(dict.fromkeys(metrics["dataset"]))[-1]
    figures = {
        "frontier": plot_frontier(metrics, latency, frontier, args.reference),
        "vs_length": plot_vs_length(metrics, args.reference),
        "per_model": plot_per_model(per_model_rows(tables["auc_by_model"], last_dataset),
                                    last_dataset),
    }
    for name, figure in figures.items():
        for path in save_figure(figure, args.figures_dir / name):
            print(f"figure written to {path}")


if __name__ == "__main__":
    main()