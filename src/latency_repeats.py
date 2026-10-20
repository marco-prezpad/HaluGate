"""
latency_repeats.py

Controlled latency measurement. evaluate.py times every detector once,
right after scoring it, and the same model has given 8 to 25 ms in
different runs, so a single p50 is not a number to build the frontier on.
This module repeats the protocol's measurement (one response at a time,
warm-up calls, the first N test responses) several times per detector and
summarises the percentiles of each repetition by their median and range.

What it controls:

- repetitions are interleaved across detectors, in a seeded random order
  per repetition, so a slow drift (a laptop GPU heating up and throttling)
  is spread over all detectors instead of landing on the last one;
- a burn-in pass per detector is run and discarded, so clocks and caches
  are in steady state before anything is recorded;
- GPU temperature, SM clock and power are read before and after every timed
  pass, outside the timed region, and the environment (versions, GPU,
  driver, git commit) is recorded next to the results.

Everything that touches the system (nvidia-smi, git) goes through an
injectable ``run_command``, so the logic is testable without a GPU.

Author: Marco Pérez Padilla
Date:   20-10-2026
"""

from __future__ import annotations

import csv
import os
import platform
import random
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from statistics import median
from typing import Any

from src.latency_bench import measure_latency
from src.strategies.base import HallucinationDetector

DEFAULT_REPETITIONS = 5
DEFAULT_BURN_IN = 1
SAMPLE_MODES = ("first", "random")

GPU_FIELDS = ("temp_c", "sm_clock_mhz", "power_w")
NVIDIA_SMI_STATE_QUERY = [
    "nvidia-smi",
    "--query-gpu=temperature.gpu,clocks.sm,power.draw",
    "--format=csv,noheader,nounits",
]
NVIDIA_SMI_STATIC_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,driver_version,clocks.max.sm,power.limit",
    "--format=csv,noheader,nounits",
]
COMMAND_TIMEOUT_SECONDS = 10

TEXT_FIELDS = ("run_id", "dataset", "strategy", "sample")
INTEGER_FIELDS = ("repetition", "n")
ROW_FIELDS = (
    *TEXT_FIELDS,
    *INTEGER_FIELDS,
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "mean_ms",
    "std_ms",
    *(f"gpu_{field}_{moment}" for field in GPU_FIELDS for moment in ("before", "after")),
)

RunCommand = Callable[[list[str]], str]


def run_command(command: list[str]) -> str:
    """Run a command and return its stdout; raises on any failure."""
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS
    )
    return completed.stdout


def parse_gpu_state(text: str) -> dict[str, float | None]:
    """Parse the output of ``NVIDIA_SMI_STATE_QUERY`` (first GPU only).

    A field the driver does not report (``[N/A]``) or an empty output gives ``None``.
    """
    state: dict[str, float | None] = {field: None for field in GPU_FIELDS}
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return state

    for field, value in zip(GPU_FIELDS, (part.strip() for part in lines[0].split(","))):
        try:
            state[field] = float(value)
        except ValueError:
            pass
    return state


def read_gpu_state(runner: RunCommand = run_command) -> dict[str, float | None]:
    """Temperature, SM clock and power of the first GPU, ``None`` when unavailable."""
    try:
        return parse_gpu_state(runner(NVIDIA_SMI_STATE_QUERY))
    except (OSError, subprocess.SubprocessError):
        return {field: None for field in GPU_FIELDS}


def repetition_order(strategies: list[str], repetition: int, random_seed: int) -> list[str]:
    """Order in which the detectors are timed in one repetition (seeded shuffle)."""
    order = list(strategies)
    random.Random(random_seed + repetition).shuffle(order)
    return order


def select_sample(
    contexts: list[str],
    responses: list[str],
    size: int,
    mode: str = "first",
    random_seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Choose the responses to time.

    ``first`` is the protocol's sample: the first ``size`` in file order.
    ``random`` is a seeded random subset, kept in file order, used to check
    that the first responses are not unrepresentative (they come from only a
    few sources).

    Raises:
        ValueError: on mismatched lengths, a non-positive size or an unknown mode.
    """
    if len(contexts) != len(responses):
        raise ValueError(
            f"contexts and responses must have equal length: {len(contexts)} != {len(responses)}"
        )
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    if mode not in SAMPLE_MODES:
        raise ValueError(f"unknown sample mode '{mode}', expected one of {SAMPLE_MODES}")

    count = min(size, len(contexts))
    if mode == "first":
        indices = list(range(count))
    else:
        indices = sorted(random.Random(random_seed).sample(range(len(contexts)), count))
    return [contexts[i] for i in indices], [responses[i] for i in indices]


def run_repetitions(
    detectors: dict[str, HallucinationDetector],
    contexts: list[str],
    responses: list[str],
    dataset_name: str,
    sample_mode: str,
    run_id: str,
    repetitions: int = DEFAULT_REPETITIONS,
    burn_in: int = DEFAULT_BURN_IN,
    random_seed: int = 42,
    read_gpu: Callable[[], dict[str, float | None]] = read_gpu_state,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Time every detector ``repetitions`` times, interleaved.

    Returns ``(rows, raw)``: one row per (detector, repetition) with the
    percentiles and the GPU state around the pass, and one raw record per
    timed response. The burn-in passes are executed and not recorded.

    Raises:
        ValueError: if there are no detectors, ``repetitions`` < 1 or ``burn_in`` < 0.
    """
    if not detectors:
        raise ValueError("At least one detector is needed.")
    if repetitions < 1:
        raise ValueError(f"repetitions must be >= 1, got {repetitions}")
    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in}")

    names = list(detectors)
    for _ in range(burn_in):
        for name in names:
            measure_latency(detectors[name], contexts, responses)

    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        for name in repetition_order(names, repetition, random_seed):
            before = read_gpu()
            records, stats = measure_latency(detectors[name], contexts, responses)
            after = read_gpu()

            row: dict[str, Any] = {
                "run_id": run_id,
                "dataset": dataset_name,
                "strategy": name,
                "sample": sample_mode,
                "repetition": repetition,
                "n": stats.sample_count,
                "p50_ms": stats.p50_ms,
                "p95_ms": stats.p95_ms,
                "p99_ms": stats.p99_ms,
                "mean_ms": stats.mean_ms,
                "std_ms": stats.std_ms,
            }
            for field in GPU_FIELDS:
                row[f"gpu_{field}_before"] = before.get(field)
                row[f"gpu_{field}_after"] = after.get(field)
            rows.append(row)

            raw.extend(
                {
                    "run_id": run_id,
                    "dataset": dataset_name,
                    "strategy": name,
                    "sample": sample_mode,
                    "repetition": repetition,
                    "example_index": record.example_index,
                    "latency_ms": record.latency_ms,
                }
                for record in records
            )
    return rows, raw


def summarize_repetitions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Median and range of the per-repetition percentiles of each
    (dataset, detector, sample), in order of first appearance.

    ``p50_spread_pct`` is (max - min) / median of the p50, in percent: how
    much the same detector moved between repetitions.
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["strategy"], row["sample"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for (dataset_name, strategy_name, sample_mode), group in groups.items():
        p50 = [row["p50_ms"] for row in group]
        p95 = [row["p95_ms"] for row in group]
        centre = median(p50)
        summary.append(
            {
                "dataset": dataset_name,
                "strategy": strategy_name,
                "sample": sample_mode,
                "runs": len(group),
                "p50_median_ms": centre,
                "p50_min_ms": min(p50),
                "p50_max_ms": max(p50),
                "p50_spread_pct": 100.0 * (max(p50) - min(p50)) / centre if centre > 0 else 0.0,
                "p95_median_ms": median(p95),
                "p95_min_ms": min(p95),
                "p95_max_ms": max(p95),
                "p99_median_ms": median(row["p99_ms"] for row in group),
            }
        )
    return summary


def append_repetition_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Append rows to the cumulative CSV, writing the header only once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ROW_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def read_repetition_rows(path: Path) -> list[dict[str, Any]]:
    """Read the cumulative CSV back, restoring numbers and empty GPU fields."""
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            row: dict[str, Any] = {}
            for key, value in raw_row.items():
                if key in TEXT_FIELDS:
                    row[key] = value
                elif value == "":
                    row[key] = None
                elif key in INTEGER_FIELDS:
                    row[key] = int(float(value))
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _query_static_gpu(runner: RunCommand) -> dict[str, Any]:
    keys = ("name", "driver_version", "max_sm_clock_mhz", "power_limit_w")
    values: dict[str, Any] = {key: None for key in keys}
    try:
        lines = [line for line in runner(NVIDIA_SMI_STATIC_QUERY).splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return values
    if lines:
        for key, value in zip(keys, (part.strip() for part in lines[0].split(","))):
            values[key] = value
    return values


def _git_state(runner: RunCommand) -> dict[str, Any]:
    try:
        commit = runner(["git", "rev-parse", "HEAD"]).strip()
        dirty = bool(runner(["git", "status", "--porcelain"]).strip())
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def collect_environment(runner: RunCommand = run_command) -> dict[str, Any]:
    """Versions, hardware and code state to store next to the measurements."""
    environment: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": {
            package: _package_version(package)
            for package in ("numpy", "pandas", "scikit-learn", "torch", "transformers")
        },
        "torch_cuda": None,
        "cuda_available": None,
        "torch_gpu_name": None,
        "nvidia_smi": _query_static_gpu(runner),
        **_git_state(runner),
    }
    try:
        import torch

        environment["torch_cuda"] = torch.version.cuda
        environment["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            environment["torch_gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return environment