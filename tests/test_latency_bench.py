"""
test_latency_bench.py

Uses a detector with an artificial sleep to verify warm-up exclusion,
correct example indices, millisecond units, and error handling.

Author: Marco Pérez Padilla
Date:   04-10-2026
"""

from __future__ import annotations

import time

import pytest

from src.latency_bench import LatencyRecord, measure_latency, summarize_latencies
from src.strategies.base import HallucinationDetector


class _SleepyDetector(HallucinationDetector):
    name = "sleepy"

    def __init__(self, sleep_seconds: float = 0.001) -> None:
        self.sleep_seconds = sleep_seconds
        self.call_count = 0

    def predict_batch(self, contexts, responses):
        self.call_count += len(contexts)
        time.sleep(self.sleep_seconds * len(contexts))
        return [False] * len(contexts), [0.0] * len(contexts)


class TestMeasureLatency:
    def test_returns_one_record_per_sample(self) -> None:
        records, _stats = measure_latency(
            _SleepyDetector(), ["c1", "c2", "c3"], ["r1", "r2", "r3"], warmup_samples=1
        )
        assert len(records) == 3
        assert all(isinstance(r, LatencyRecord) and r.latency_ms > 0 for r in records)

    def test_records_carry_sequential_example_indices(self) -> None:
        records, _stats = measure_latency(
            _SleepyDetector(), ["c1", "c2", "c3"], ["r1", "r2", "r3"], warmup_samples=0
        )
        assert [r.example_index for r in records] == [0, 1, 2]

    def test_warmup_calls_are_excluded_from_timed_records(self) -> None:
        detector = _SleepyDetector()
        measure_latency(detector, ["c1", "c2", "c3"], ["r1", "r2", "r3"], warmup_samples=2)
        assert detector.call_count == 5  # 2 warmup + 3 timed

    def test_raises_on_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            measure_latency(_SleepyDetector(), ["c1", "c2"], ["r1"])

    def test_raises_on_empty_input(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            measure_latency(_SleepyDetector(), [], [])


class TestSummarizeLatencies:
    def test_computes_expected_statistics(self) -> None:
        stats = summarize_latencies([1.0, 2.0, 3.0, 4.0])
        assert stats.mean_ms == pytest.approx(2.5)
        assert stats.sample_count == 4

    def test_raises_on_empty_list(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            summarize_latencies([])