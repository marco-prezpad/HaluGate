"""
test_latency_repeats.py

Covers the sample selection, the interleaved order of the repetitions, the
burn-in (executed, not recorded), the summary (median and range), the
cumulative CSV, and the parsing of nvidia-smi and the environment, all with
fake detectors and fake commands so no GPU is needed.

Author: Marco Pérez Padilla
Date:   20-10-2026
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.latency_repeats import (
    GPU_FIELDS,
    ROW_FIELDS,
    append_repetition_rows,
    collect_environment,
    parse_gpu_state,
    read_gpu_state,
    read_repetition_rows,
    repetition_order,
    run_repetitions,
    select_sample,
    summarize_repetitions,
)
from src.strategies.base import HallucinationDetector


class _LoggingDetector(HallucinationDetector):
    """Records every call in a shared log, so the order can be checked."""

    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    def predict_batch(self, contexts, responses):
        self.log.append(self.name)
        return [False] * len(contexts), [0.0] * len(contexts)


CONTEXTS = [f"c{i}" for i in range(10)]
RESPONSES = [f"r{i}" for i in range(10)]
NO_GPU = {field: None for field in GPU_FIELDS}


class TestSelectSample:
    def test_first_mode_keeps_the_first_responses_in_file_order(self) -> None:
        contexts, responses = select_sample(CONTEXTS, RESPONSES, 3, "first")

        assert contexts == ["c0", "c1", "c2"] and responses == ["r0", "r1", "r2"]

    def test_random_mode_is_seeded_sorted_and_not_just_the_first(self) -> None:
        first = select_sample(CONTEXTS, RESPONSES, 4, "random", random_seed=1)
        again = select_sample(CONTEXTS, RESPONSES, 4, "random", random_seed=1)

        assert first == again
        assert len(set(first[0])) == 4
        assert first[0] == sorted(first[0], key=lambda text: int(text[1:]))
        assert first[0] != CONTEXTS[:4]

    def test_contexts_and_responses_stay_paired(self) -> None:
        contexts, responses = select_sample(CONTEXTS, RESPONSES, 5, "random", random_seed=7)

        assert [c[1:] for c in contexts] == [r[1:] for r in responses]

    def test_a_size_larger_than_the_data_returns_everything(self) -> None:
        assert select_sample(CONTEXTS, RESPONSES, 99, "first")[0] == CONTEXTS
        assert select_sample(CONTEXTS, RESPONSES, 99, "random")[0] == CONTEXTS

    @pytest.mark.parametrize(
        "contexts, responses, size, mode",
        [
            (CONTEXTS, RESPONSES[:-1], 3, "first"),
            (CONTEXTS, RESPONSES, 0, "first"),
            (CONTEXTS, RESPONSES, 3, "middle"),
        ],
    )
    def test_rejects_invalid_arguments(self, contexts, responses, size, mode) -> None:
        with pytest.raises(ValueError):
            select_sample(contexts, responses, size, mode)


class TestRepetitionOrder:
    def test_every_detector_appears_exactly_once(self) -> None:
        names = ["a", "b", "c", "d"]

        assert sorted(repetition_order(names, 1, 42)) == names

    def test_the_same_seed_and_repetition_give_the_same_order(self) -> None:
        names = ["a", "b", "c", "d"]

        assert repetition_order(names, 3, 42) == repetition_order(names, 3, 42)

    def test_the_order_changes_between_repetitions(self) -> None:
        names = ["a", "b", "c", "d", "e"]

        orders = {tuple(repetition_order(names, repetition, 42)) for repetition in range(1, 9)}

        assert len(orders) > 1

    def test_the_input_list_is_not_modified(self) -> None:
        names = ["a", "b", "c", "d"]
        repetition_order(names, 2, 42)

        assert names == ["a", "b", "c", "d"]


class TestRunRepetitions:
    def _run(self, repetitions: int = 3, burn_in: int = 1, **kwargs):
        log: list[str] = []
        detectors = {name: _LoggingDetector(name, log) for name in ("a", "b", "c")}
        result = run_repetitions(
            detectors, CONTEXTS[:4], RESPONSES[:4], "d", "first", "run1",
            repetitions=repetitions, burn_in=burn_in, read_gpu=lambda: NO_GPU, **kwargs,
        )
        return result, log

    def test_one_row_per_detector_and_repetition(self) -> None:
        (rows, _), _ = self._run(repetitions=3)

        assert len(rows) == 3 * 3
        assert {(row["strategy"], row["repetition"]) for row in rows} == {
            (name, repetition) for name in "abc" for repetition in (1, 2, 3)
        }

    def test_one_raw_record_per_timed_response(self) -> None:
        (_, raw), _ = self._run(repetitions=2)

        assert len(raw) == 2 * 3 * 4
        assert {record["example_index"] for record in raw} == {0, 1, 2, 3}

    def test_burn_in_runs_but_is_not_recorded(self) -> None:
        (rows, raw), log = self._run(repetitions=2, burn_in=2)

        # per detector: (burn_in + repetitions) passes x (3 warm-up + 4 timed) calls
        assert log.count("a") == (2 + 2) * (3 + 4)
        assert len(rows) == 2 * 3 and {row["repetition"] for row in rows} == {1, 2}

    def test_detectors_are_interleaved_inside_each_repetition(self) -> None:
        (rows, _), _ = self._run(repetitions=4, burn_in=0)

        for repetition in range(1, 5):
            order = [row["strategy"] for row in rows if row["repetition"] == repetition]
            assert order == repetition_order(["a", "b", "c"], repetition, 42)

    def test_rows_carry_the_gpu_state_before_and_after_each_pass(self) -> None:
        states = iter(range(1, 1000))
        log: list[str] = []

        def _read() -> dict[str, float | None]:
            value = float(next(states))
            return {"temp_c": value, "sm_clock_mhz": value * 10, "power_w": None}

        (rows, _) = run_repetitions(
            {"a": _LoggingDetector("a", log)}, CONTEXTS[:2], RESPONSES[:2], "d", "first", "r",
            repetitions=2, burn_in=0, read_gpu=_read,
        )

        assert [(r["gpu_temp_c_before"], r["gpu_temp_c_after"]) for r in rows] == [(1, 2), (3, 4)]
        assert rows[0]["gpu_sm_clock_mhz_after"] == 20 and rows[0]["gpu_power_w_after"] is None

    def test_rows_hold_every_declared_field_and_positive_latencies(self) -> None:
        (rows, _), _ = self._run(repetitions=1)

        for row in rows:
            assert set(row) == set(ROW_FIELDS)
            assert row["n"] == 4 and 0 < row["p50_ms"] <= row["p95_ms"] <= row["p99_ms"]

    @pytest.mark.parametrize("kwargs", [{"repetitions": 0}, {"burn_in": -1}])
    def test_rejects_invalid_counts(self, kwargs) -> None:
        with pytest.raises(ValueError):
            self._run(**kwargs)

    def test_rejects_an_empty_detector_set(self) -> None:
        with pytest.raises(ValueError, match="detector"):
            run_repetitions({}, CONTEXTS, RESPONSES, "d", "first", "r")


def _row(strategy: str, p50: float, p95: float = 10.0, dataset: str = "d", sample: str = "first"):
    return {
        "dataset": dataset, "strategy": strategy, "sample": sample,
        "p50_ms": p50, "p95_ms": p95, "p99_ms": p95 * 2,
    }


class TestSummarizeRepetitions:
    def test_median_and_range_of_the_p50(self) -> None:
        rows = [_row("a", 10.0), _row("a", 14.0), _row("a", 12.0), _row("a", 25.0)]

        (summary,) = summarize_repetitions(rows)

        assert summary["runs"] == 4
        assert summary["p50_median_ms"] == pytest.approx(13.0)
        assert (summary["p50_min_ms"], summary["p50_max_ms"]) == (10.0, 25.0)

    def test_spread_is_the_range_relative_to_the_median(self) -> None:
        (summary,) = summarize_repetitions([_row("a", 8.0), _row("a", 10.0), _row("a", 12.0)])

        assert summary["p50_spread_pct"] == pytest.approx(100 * (12 - 8) / 10)

    def test_a_perfectly_stable_detector_has_zero_spread(self) -> None:
        (summary,) = summarize_repetitions([_row("a", 5.0)] * 3)

        assert summary["p50_spread_pct"] == 0.0

    def test_detectors_datasets_and_samples_are_kept_apart_in_order(self) -> None:
        rows = [
            _row("b", 1.0), _row("a", 2.0), _row("b", 3.0),
            _row("a", 9.0, dataset="other"), _row("a", 7.0, sample="random"),
        ]

        summary = summarize_repetitions(rows)

        assert [(s["dataset"], s["strategy"], s["sample"], s["runs"]) for s in summary] == [
            ("d", "b", "first", 2),
            ("d", "a", "first", 1),
            ("other", "a", "first", 1),
            ("d", "a", "random", 1),
        ]

    def test_percentiles_95_and_99_are_summarised_too(self) -> None:
        (summary,) = summarize_repetitions([_row("a", 1.0, p95=4.0), _row("a", 1.0, p95=8.0)])

        assert summary["p95_median_ms"] == pytest.approx(6.0)
        assert (summary["p95_min_ms"], summary["p95_max_ms"]) == (4.0, 8.0)
        assert summary["p99_median_ms"] == pytest.approx(12.0)

    def test_empty_input_gives_an_empty_summary(self) -> None:
        assert summarize_repetitions([]) == []


class TestCumulativeCsv:
    def _rows(self, run_id: str, repetitions: int = 2):
        log: list[str] = []
        rows, _ = run_repetitions(
            {"a": _LoggingDetector("a", log)}, CONTEXTS[:3], RESPONSES[:3], "d", "first", run_id,
            repetitions=repetitions, burn_in=0, read_gpu=lambda: NO_GPU,
        )
        return rows

    def test_invocations_accumulate_with_a_single_header(self, tmp_path: Path) -> None:
        path = tmp_path / "tables" / "latency_repeats.csv"

        append_repetition_rows(self._rows("one"), path)
        append_repetition_rows(self._rows("two"), path)

        assert path.read_text(encoding="utf-8").count("run_id,") == 1
        assert [row["run_id"] for row in read_repetition_rows(path)] == ["one"] * 2 + ["two"] * 2

    def test_values_survive_the_round_trip_with_their_types(self, tmp_path: Path) -> None:
        path = tmp_path / "latency_repeats.csv"
        written = self._rows("one", repetitions=1)

        append_repetition_rows(written, path)
        (read,) = read_repetition_rows(path)

        assert isinstance(read["repetition"], int) and isinstance(read["n"], int)
        assert read["p50_ms"] == pytest.approx(written[0]["p50_ms"])
        assert read["gpu_temp_c_before"] is None

    def test_a_summary_can_be_built_from_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "latency_repeats.csv"
        append_repetition_rows(self._rows("one"), path)
        append_repetition_rows(self._rows("two"), path)

        (summary,) = summarize_repetitions(read_repetition_rows(path))

        assert summary["runs"] == 4


class TestGpuState:
    def test_parses_a_normal_line(self) -> None:
        assert parse_gpu_state("45, 2100, 62.35\n") == {
            "temp_c": 45.0, "sm_clock_mhz": 2100.0, "power_w": 62.35,
        }

    def test_a_field_the_driver_does_not_report_is_none(self) -> None:
        state = parse_gpu_state("45, 2100, [N/A]")

        assert state["power_w"] is None and state["temp_c"] == 45.0

    @pytest.mark.parametrize("text", ["", "\n", "   "])
    def test_empty_output_gives_all_none(self, text: str) -> None:
        assert parse_gpu_state(text) == {field: None for field in GPU_FIELDS}

    def test_only_the_first_gpu_is_read(self) -> None:
        assert parse_gpu_state("40, 1000, 10\n99, 9999, 99")["temp_c"] == 40.0

    def test_a_missing_nvidia_smi_gives_all_none_instead_of_failing(self) -> None:
        def _missing(_command: list[str]) -> str:
            raise FileNotFoundError("nvidia-smi")

        assert read_gpu_state(_missing) == {field: None for field in GPU_FIELDS}

    def test_a_failing_nvidia_smi_gives_all_none_instead_of_failing(self) -> None:
        def _fails(command: list[str]) -> str:
            raise subprocess.CalledProcessError(1, command)

        assert read_gpu_state(_fails) == {field: None for field in GPU_FIELDS}

    def test_the_command_output_is_parsed(self) -> None:
        assert read_gpu_state(lambda _command: "50, 1500, 30")["sm_clock_mhz"] == 1500.0


class TestCollectEnvironment:
    @staticmethod
    def _runner(command: list[str]) -> str:
        if command[0] == "nvidia-smi":
            return "NVIDIA GeForce RTX 5060 Laptop GPU, 576.02, 2850, 80.00\n"
        if command[:2] == ["git", "rev-parse"]:
            return "abc123\n"
        return " M src/evaluate.py\n"

    def test_records_versions_hardware_and_code_state(self) -> None:
        environment = collect_environment(self._runner)

        assert environment["nvidia_smi"]["name"] == "NVIDIA GeForce RTX 5060 Laptop GPU"
        assert environment["nvidia_smi"]["driver_version"] == "576.02"
        assert environment["git_commit"] == "abc123" and environment["git_dirty"] is True
        assert environment["python"] and environment["packages"]["numpy"]

    def test_a_clean_working_tree_is_not_dirty(self) -> None:
        def _clean(command: list[str]) -> str:
            return "" if command[:2] == ["git", "status"] else self._runner(command)

        assert collect_environment(_clean)["git_dirty"] is False

    def test_missing_tools_leave_none_instead_of_failing(self) -> None:
        def _nothing(_command: list[str]) -> str:
            raise FileNotFoundError("not installed")

        environment = collect_environment(_nothing)

        assert environment["git_commit"] is None and environment["git_dirty"] is None
        assert environment["nvidia_smi"]["name"] is None