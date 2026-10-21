#!/usr/bin/env python3
"""
truncation_report.py

Share of (context, response) pairs that exceed the token limit of each
DistilBERT arm, for the train, validation and test split of every dataset,
in total and per class (see src/truncation.py). Lengths are measured with
the real tokenizer of each arm's checkpoint, and the datasets, splits and
limits are the ones evaluate.py uses.

Writes results/tables/truncation.csv and prints it. It reads no scores and
no model weights, only tokenizers, so it cannot influence any result.

Usage:
    python scripts/truncation_report.py [--datasets halueval ragtruth_qa]

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

from src.data_loading import get_dataset_loader  # noqa: E402
from src.evaluate import DEFAULT_SPLITS, load_config  # noqa: E402
from src.splitting import split_dataset  # noqa: E402
from src.strategies.distilbert_classifier import DEFAULT_MAX_SEQUENCE_LENGTH  # noqa: E402
from src.truncation import pair_lengths, truncation_rows  # noqa: E402
from src.utils import configure_logging  # noqa: E402

DISTILBERT_PREFIX = "distilbert"
SPLIT_NAMES = ("train", "validation", "test")


def load_tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path)


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: list[dict[str, Any]]) -> None:
    print(
        f"\n{'dataset':<12} {'arm':<20} {'split':<11} {'group':<9} {'n':>6} {'trunc':>6} "
        f"{'trunc %':>8} {'resp>lim':>8} {'ctx med':>8} {'ctx p90':>8} {'kept %':>7}"
    )
    for row in rows:
        print(
            f"{row['dataset']:<12} {row['strategy']:<20} {row['split']:<11} {row['group']:<9} "
            f"{row['n']:>6} {row['truncated']:>6} {row['truncated_pct']:>8.2f} "
            f"{row['response_over_limit']:>8} {row['context_tokens_median']:>8.0f} "
            f"{row['context_tokens_p90']:>8.0f} {row['context_kept_pct_mean']:>7.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Share of pairs truncated by each DistilBERT arm.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--datasets", nargs="+", default=None, help="default: from the config")
    args = parser.parse_args()

    config = load_config(args.config)
    results_root = Path(config["results_root"])
    random_seed = config.get("random_seed", 42)
    logger = configure_logging(results_root / "logs" / "truncation_report.log")

    rows: list[dict[str, Any]] = []
    for dataset_name in args.datasets or config["datasets"]:
        samples = get_dataset_loader(dataset_name, Path(config["data_root"])).load()
        splits = split_dataset(samples, config.get("splits", DEFAULT_SPLITS), random_seed)
        partitions = dict(zip(SPLIT_NAMES, (splits.train, splits.validation, splits.test)))

        strategies = config.get("strategies_by_dataset", {}).get(dataset_name, config["strategies"])
        arms = [name for name in strategies if name.startswith(DISTILBERT_PREFIX)]
        for arm in arms:
            params = config.get("strategy_params", {}).get(arm, {})
            model_path = params["model_path"]
            max_length = params.get("max_sequence_length", DEFAULT_MAX_SEQUENCE_LENGTH)
            tokenizer = load_tokenizer(model_path)
            special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
            logger.info(
                "%s / %s: tokenizer of %s, limit %d (%d special tokens)",
                dataset_name, arm, model_path, max_length, special_tokens,
            )

            for split_name, partition in partitions.items():
                lengths = pair_lengths(
                    tokenizer, [s.context for s in partition], [s.response for s in partition]
                )
                rows.extend(
                    truncation_rows(
                        dataset_name, arm, split_name, lengths,
                        [int(s.is_hallucination) for s in partition], special_tokens, max_length,
                    )
                )

    if not rows:
        logger.warning("No DistilBERT arm in the selected datasets; nothing to report.")
        return

    output_path = results_root / "tables" / "truncation.csv"
    write_rows(rows, output_path)
    print_rows(rows)
    logger.info("Truncation table written to %s", output_path)


if __name__ == "__main__":
    main()