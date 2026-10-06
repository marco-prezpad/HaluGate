"""
latency_bench.py

Measures per-sample inference latency for any HallucinationDetector,
including a warm-up phase so one-off costs (CUDA context init, first-call
JIT/cache effects) do not bias the measured distribution.

Returns both the raw per-sample measurements (tagged with the index of the
example they came from, so they can be persisted and re-analysed later
without re-running inference) and the summary statistics.

Author: Marco Pérez Padilla
Date:   04-10-2026
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.strategies.base import HallucinationDetector

DEFAULT_WARMUP_SAMPLES = 3
MILLISECONDS_PER_SECOND = 1000.0


@dataclass(frozen=True)
class LatencyRecord:
    """A single timed measurement, tagged with the example it came from."""

    example_index: int
    latency_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return {"example_index": self.example_index, "latency_ms": self.latency_ms}


@dataclass(frozen=True)
class LatencyStats:
    """Summary statistics over a set of per-sample latency measurements."""

    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    sample_count: int


def summarize_latencies(latencies_ms: list[float]) -> LatencyStats:
    """Reduce raw millisecond measurements to the reported summary statistics.

    Raises:
        ValueError: if ``latencies_ms`` is empty.
    """
    if not latencies_ms:
        raise ValueError("Cannot summarize an empty list of latencies.")

    latency_array = np.asarray(latencies_ms, dtype=float)
    return LatencyStats(
        mean_ms=float(latency_array.mean()),
        std_ms=float(latency_array.std()),
        p50_ms=float(np.percentile(latency_array, 50)),
        p95_ms=float(np.percentile(latency_array, 95)),
        p99_ms=float(np.percentile(latency_array, 99)),
        sample_count=len(latencies_ms),
    )


def measure_latency(
    detector: HallucinationDetector,
    contexts: list[str],
    responses: list[str],
    warmup_samples: int = DEFAULT_WARMUP_SAMPLES,
) -> tuple[list[LatencyRecord], LatencyStats]:
    """Time ``detector.predict`` over each (context, response) pair individually.

    Single-sample timing (rather than batch timing) is intentional: in
    production, a guardrail typically scores one LLM response at a time as
    it streams back, so per-sample latency is the metric that matters.

    Raises:
        ValueError: if ``contexts`` and ``responses`` differ in length or are empty.
    """
    if len(contexts) != len(responses):
        raise ValueError(
            f"contexts and responses must have equal length: {len(contexts)} != {len(responses)}"
        )
    if len(contexts) == 0:
        raise ValueError("Cannot measure latency over an empty sample set.")

    warmup_count = min(warmup_samples, len(contexts))
    for warmup_index in range(warmup_count):
        detector.predict(contexts[warmup_index], responses[warmup_index])

    records: list[LatencyRecord] = []
    for example_index, (context, response) in enumerate(zip(contexts, responses)):
        start_time = time.perf_counter()
        detector.predict(context, response)
        elapsed_ms = (time.perf_counter() - start_time) * MILLISECONDS_PER_SECOND
        records.append(LatencyRecord(example_index=example_index, latency_ms=elapsed_ms))

    stats = summarize_latencies([record.latency_ms for record in records])
    return records, stats