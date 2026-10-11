"""
test_data_loading.py

Covers HaluEval loader parsing against the real schema (including that the
question is folded into the stored context -- see HaluEvalLoader's
docstring for why), the processed-cache layer, and error handling for
missing files and malformed records.

Author: Marco Pérez Padilla
Date:   06-10-2026
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data_loading import HaluEvalLoader, get_dataset_loader
from src.errors import DatasetFormatError, DatasetNotFoundError, UnknownStrategyError


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record) + "\n")


class TestHaluEvalLoader:
    def test_load_produces_two_samples_per_record(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw" / "halueval_qa.jsonl"
        _write_jsonl(
            raw_path,
            [
                {
                    "knowledge": "Paris is the capital of France.",
                    "question": "What is the capital of France?",
                    "right_answer": "Paris",
                    "hallucinated_answer": "Lyon",
                }
            ],
        )
        loader = HaluEvalLoader(raw_path=raw_path, processed_dir=tmp_path / "processed")

        samples = loader.load()

        assert len(samples) == 2
        assert any(s.response == "Paris" and s.is_hallucination is False for s in samples)
        assert any(s.response == "Lyon" and s.is_hallucination is True for s in samples)

    def test_context_is_the_knowledge_passage_only(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw" / "halueval_qa.jsonl"
        _write_jsonl(
            raw_path,
            [
                {
                    "knowledge": "Delhi is the capital of India.",
                    "question": "What is the capital of India?",
                    "right_answer": "Delhi",
                    "hallucinated_answer": "Mumbai",
                }
            ],
        )
        loader = HaluEvalLoader(raw_path=raw_path, processed_dir=tmp_path / "processed")

        samples = loader.load()

        for sample in samples:
            assert sample.context == "Delhi is the capital of India."
            assert "What is the capital" not in sample.context

    def test_load_raises_when_raw_file_missing(self, tmp_path: Path) -> None:
        loader = HaluEvalLoader(
            raw_path=tmp_path / "raw" / "missing.jsonl", processed_dir=tmp_path / "processed"
        )
        with pytest.raises(DatasetNotFoundError):
            loader.load()

    def test_load_raises_on_missing_question_field(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw" / "halueval_qa.jsonl"
        _write_jsonl(
            raw_path,
            [{"knowledge": "x", "right_answer": "y", "hallucinated_answer": "z"}],
        )  # missing "question"
        loader = HaluEvalLoader(raw_path=raw_path, processed_dir=tmp_path / "processed")

        with pytest.raises(DatasetFormatError):
            loader.load()

    def test_load_raises_on_missing_required_field(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw" / "halueval_qa.jsonl"
        _write_jsonl(raw_path, [{"knowledge": "x", "question": "q", "right_answer": "y"}])
        loader = HaluEvalLoader(raw_path=raw_path, processed_dir=tmp_path / "processed")

        with pytest.raises(DatasetFormatError):
            loader.load()

    def test_load_raises_on_malformed_json_line(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw" / "halueval_qa.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text("{not valid json\n", encoding="utf-8")
        loader = HaluEvalLoader(raw_path=raw_path, processed_dir=tmp_path / "processed")

        with pytest.raises(DatasetFormatError):
            loader.load()

    def test_load_skips_blank_lines(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw" / "halueval_qa.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        record = json.dumps(
            {"knowledge": "k", "question": "q", "right_answer": "r", "hallucinated_answer": "h"}
        )
        raw_path.write_text(f"{record}\n\n{record}\n", encoding="utf-8")
        loader = HaluEvalLoader(raw_path=raw_path, processed_dir=tmp_path / "processed")

        samples = loader.load()
        assert len(samples) == 4

    def test_second_load_uses_cache_not_raw_file(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw" / "halueval_qa.jsonl"
        _write_jsonl(
            raw_path,
            [{"knowledge": "k", "question": "q", "right_answer": "r", "hallucinated_answer": "h"}],
        )
        loader = HaluEvalLoader(raw_path=raw_path, processed_dir=tmp_path / "processed")
        first_load = loader.load()

        raw_path.unlink()

        second_load = loader.load()
        assert first_load == second_load

    def test_cache_file_is_created_after_first_load(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw" / "halueval_qa.jsonl"
        _write_jsonl(
            raw_path,
            [{"knowledge": "k", "question": "q", "right_answer": "r", "hallucinated_answer": "h"}],
        )
        processed_dir = tmp_path / "processed"
        loader = HaluEvalLoader(raw_path=raw_path, processed_dir=processed_dir)
        loader.load()

        assert (processed_dir / "halueval.jsonl").exists()


class TestGetDatasetLoader:
    def test_returns_halueval_loader(self, tmp_path: Path) -> None:
        loader = get_dataset_loader("halueval", tmp_path)
        assert isinstance(loader, HaluEvalLoader)

    def test_raises_on_unknown_dataset_name(self, tmp_path: Path) -> None:
        with pytest.raises(UnknownStrategyError):
            get_dataset_loader("made_up_dataset", tmp_path)