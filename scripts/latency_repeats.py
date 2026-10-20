#!/usr/bin/env python3
"""
latency_repeats.py

Repeats the protocol's latency measurement several times per detector (see
src/latency_repeats.py) and writes, under results/:

- tables/latency_repeats.csv: one row per (detector, repetition), appended
  on every invocation, with the GPU state around each timed pass;
- tables/latency_summary.csv: median and range of the per-repetition
  percentiles, computed over EVERY row of the file above;
- tables/latency_environment_<run_id>.json: versions, GPU, driver, commit;
- latency/repeats_raw_<run_id>.parquet: every timed response.

The detectors, parameters, splits and the sample (the first
``latency_sample_size`` test responses in file order) are the ones
evaluate.py uses, so the numbers are comparable with its p50.

For a measurement worth reporting on a laptop: plug it in, use the
high-performance power plan, close everything else that uses the GPU, and
run the script two or three times at different moments (each invocation
adds its repetitions to the summary). A large ``p50_spread_pct`` means the
detector's latency is not stable and its p50 must be reported as a range.

Usage:
    python scripts/latency_repeats.py [--repetitions 5] [--burn-in 1]
    python scripts/latency_repeats.py --datasets ragtruth_qa --sample random

Author: Marco Pérez Padilla
Date:   20-10-2026
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import get_dataset_loader  # noqa: E402
from src.evaluate import DEFAULT_SPLITS, load_config  # noqa: E402
from src.factory import get_detector  # noqa: E402
from src.latency_repeats import (  # noqa: E402
    DEFAULT_BURN_IN,
    DEFAULT_REPETITIONS,
    SAMPLE_MODES,
    append_repetition_rows,
    collect_environment,
    read_repetition_rows,
    run_repetitions,
    select_sample,
    summarize_repetitions,
)
from src.splitting import split_dataset  # noqa: E402
from src.utils import configure_logging, set_global_seed  # noqa: E402


def write_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    print(
        f"\n{'dataset':<12} {'strategy':<20} {'sample':<7} {'runs':>4} "
        f"{'p50 median':>11} {'p50 min':>9} {'p50 max':>9} {'spread %':>9} {'p95 median':>11}"
    )
    for row in rows:
        print(
            f"{row['dataset']:<12} {row['strategy']:<20} {row['sample']:<7} {row['runs']:>4} "
            f"{row['p50_median_ms']:>11.3f} {row['p50_min_ms']:>9.3f} {row['p50_max_ms']:>9.3f} "
            f"{row['p50_spread_pct']:>9.1f} {row['p95_median_ms']:>11.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated, controlled latency measurement.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--datasets", nargs="+", default=None, help="default: from the config")
    parser.add_argument("--strategies", nargs="+", default=None, help="default: from the config")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--burn-in", type=int, default=DEFAULT_BURN_IN)
    parser.add_argument("--sample", choices=SAMPLE_MODES, default="first")
    parser.add_argument("--sample-size", type=int, default=None, help="default: from the config")
    parser.add_argument("--run-id", default=None, help="default: current date and time")
    args = parser.parse_args()

    config = load_config(args.config)
    results_root = Path(config["results_root"])
    data_root = Path(config["data_root"])
    random_seed = config.get("random_seed", 42)
    logger = configure_logging(results_root / "logs" / "latency_repeats.log")
    set_global_seed(random_seed)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    sample_size = args.sample_size or config.get("latency_sample_size", 200)
    tables_dir = results_root / "tables"

    environment = collect_environment()
    environment["run_id"] = run_id
    environment["arguments"] = {
        "repetitions": args.repetitions,
        "burn_in": args.burn_in,
        "sample": args.sample,
        "sample_size": sample_size,
    }
    environment_path = tables_dir / f"latency_environment_{run_id}.json"
    environment_path.parent.mkdir(parents=True, exist_ok=True)
    environment_path.write_text(json.dumps(environment, indent=2, default=str), encoding="utf-8")
    logger.info("Run %s | environment written to %s", run_id, environment_path)

    raw_frames: list[pd.DataFrame] = []
    for dataset_name in args.datasets or config["datasets"]:
        samples = get_dataset_loader(dataset_name, data_root).load()
        splits = split_dataset(samples, config.get("splits", DEFAULT_SPLITS), random_seed)
        contexts, responses = select_sample(
            [sample.context for sample in splits.test],
            [sample.response for sample in splits.test],
            sample_size,
            args.sample,
            random_seed,
        )

        strategies = args.strategies or config.get("strategies_by_dataset", {}).get(
            dataset_name, config["strategies"]
        )
        params = config.get("strategy_params", {})
        detectors = {
            name: get_detector(name, dict(params.get(name, {}))) for name in strategies
        }
        logger.info(
            "%s: %d responses (%s), %d detectors, %d repetitions after %d burn-in",
            dataset_name, len(contexts), args.sample, len(detectors), args.repetitions,
            args.burn_in,
        )

        rows, raw = run_repetitions(
            detectors, contexts, responses, dataset_name, args.sample, run_id,
            args.repetitions, args.burn_in, random_seed,
        )
        append_repetition_rows(rows, tables_dir / "latency_repeats.csv")
        raw_frames.append(pd.DataFrame(raw))

    raw_path = results_root / "latency" / f"repeats_raw_{run_id}.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(raw_frames, ignore_index=True).to_parquet(raw_path, index=False)

    summary = summarize_repetitions(read_repetition_rows(tables_dir / "latency_repeats.csv"))
    write_summary(summary, tables_dir / "latency_summary.csv")
    print_summary(summary)
    logger.info("Latency summary written to %s", tables_dir / "latency_summary.csv")


if __name__ == "__main__":
    main()