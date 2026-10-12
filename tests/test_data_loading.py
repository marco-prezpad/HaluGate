"""
test_data_loading.py

Covers the HaluEval loader (context is the knowledge passage only), the
RAGTruth QA loader (task filtering, labels, grouping, official split,
error handling), the processed-cache layer, and the dataset factory.

Author: Marco Pérez Padilla
Date:   11-10-2026
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.data_loading import (
    HaluEvalLoader,
    HallucinationSample,
    RagTruthLoader,
    get_dataset_loader,
)
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

    def test_cache_written_before_group_fields_existed_still_loads(
        self, tmp_path: Path
    ) -> None:
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir(parents=True)
        old_line = {
            "context": "k",
            "response": "r",
            "is_hallucination": False,
            "source_dataset": "halueval",
        }
        (processed_dir / "halueval.jsonl").write_text(
            json.dumps(old_line) + "\n", encoding="utf-8"
        )
        loader = HaluEvalLoader(raw_path=tmp_path / "unused", processed_dir=processed_dir)

        samples = loader.load()

        assert samples == [HallucinationSample("k", "r", False, "halueval", "", "")]


def _source(source_id: str, task_type: str, source_info: Any) -> dict:
    return {
        "source_id": source_id,
        "task_type": task_type,
        "source": "MARCO",
        "source_info": source_info,
        "prompt": "unused",
    }


def _response(source_id: str, split: str = "train", labels: list | None = None) -> dict:
    return {
        "id": "0",
        "source_id": source_id,
        "model": "gpt-4-0613",
        "temperature": 0.7,
        "labels": labels if labels is not None else [],
        "split": split,
        "quality": "good",
        "response": f"answer for {source_id}",
    }


SPAN = {"start": 0, "end": 5, "text": "x", "label_type": "Evident Baseless Info"}


def _write_ragtruth(
    directory: Path, sources: list[dict], responses: list[dict]
) -> RagTruthLoader:
    directory.mkdir(parents=True, exist_ok=True)
    _write_jsonl(directory / "source_info.jsonl", sources)
    _write_jsonl(directory / "response.jsonl", responses)
    return RagTruthLoader(raw_path=directory, processed_dir=directory.parent / "processed")


QA_SOURCE = _source("1", "QA", {"question": "what is x?", "passages": "passage 1:x is y.\n\n"})


class TestRagTruthLoader:
    def test_builds_samples_with_context_label_group_and_split(self, tmp_path: Path) -> None:
        loader = _write_ragtruth(
            tmp_path / "ragtruth",
            [QA_SOURCE],
            [_response("1", "train", []), _response("1", "test", [SPAN])],
        )

        samples = loader.load()

        assert samples == [
            HallucinationSample("passage 1:x is y.", "answer for 1", False, "ragtruth_qa", "1", "train"),
            HallucinationSample("passage 1:x is y.", "answer for 1", True, "ragtruth_qa", "1", "test"),
        ]

    def test_context_is_the_passages_without_the_question(self, tmp_path: Path) -> None:
        loader = _write_ragtruth(tmp_path / "ragtruth", [QA_SOURCE], [_response("1")])

        sample = loader.load()[0]

        assert "what is x?" not in sample.context
        assert "x is y" in sample.context

    def test_responses_of_other_tasks_are_skipped(self, tmp_path: Path) -> None:
        summary_source = _source("2", "Summary", "some long article")
        loader = _write_ragtruth(
            tmp_path / "ragtruth",
            [QA_SOURCE, summary_source],
            [_response("1"), _response("2")],
        )

        samples = loader.load()

        assert [s.group_id for s in samples] == ["1"]

    def test_quality_flag_does_not_filter_anything(self, tmp_path: Path) -> None:
        refused = {**_response("1"), "quality": "incorrect_refusal"}
        truncated = {**_response("1"), "quality": "truncated"}
        loader = _write_ragtruth(tmp_path / "ragtruth", [QA_SOURCE], [refused, truncated])

        assert len(loader.load()) == 2

    def test_all_responses_to_one_source_share_the_group_id(self, tmp_path: Path) -> None:
        loader = _write_ragtruth(
            tmp_path / "ragtruth", [QA_SOURCE], [_response("1") for _ in range(6)]
        )

        assert {s.group_id for s in loader.load()} == {"1"}

    def test_blank_lines_are_ignored(self, tmp_path: Path) -> None:
        loader = _write_ragtruth(tmp_path / "ragtruth", [QA_SOURCE], [_response("1")])
        response_path = tmp_path / "ragtruth" / "response.jsonl"
        response_path.write_text("\n" + response_path.read_text(encoding="utf-8") + "\n")

        assert len(loader.load()) == 1

    def test_missing_directory_raises_not_found(self, tmp_path: Path) -> None:
        loader = RagTruthLoader(raw_path=tmp_path / "nope", processed_dir=tmp_path / "processed")

        with pytest.raises(DatasetNotFoundError):
            loader.load()

    @pytest.mark.parametrize("missing_file", ["response.jsonl", "source_info.jsonl"])
    def test_missing_file_raises_not_found(self, tmp_path: Path, missing_file: str) -> None:
        loader = _write_ragtruth(tmp_path / "ragtruth", [QA_SOURCE], [_response("1")])
        (tmp_path / "ragtruth" / missing_file).unlink()

        with pytest.raises(DatasetNotFoundError):
            loader.load()

    def test_malformed_json_line_raises_format_error(self, tmp_path: Path) -> None:
        loader = _write_ragtruth(tmp_path / "ragtruth", [QA_SOURCE], [_response("1")])
        (tmp_path / "ragtruth" / "response.jsonl").write_text("{broken\n", encoding="utf-8")

        with pytest.raises(DatasetFormatError, match="line 1"):
            loader.load()

    def test_response_with_unknown_source_raises_format_error(self, tmp_path: Path) -> None:
        loader = _write_ragtruth(tmp_path / "ragtruth", [QA_SOURCE], [_response("999")])

        with pytest.raises(DatasetFormatError, match="999"):
            loader.load()

    def test_qa_source_without_passages_raises_format_error(self, tmp_path: Path) -> None:
        loader = _write_ragtruth(
            tmp_path / "ragtruth",
            [_source("1", "QA", {"question": "only a question"})],
            [_response("1")],
        )

        with pytest.raises(DatasetFormatError, match="passages"):
            loader.load()

    def test_qa_source_info_that_is_not_a_dict_raises_format_error(self, tmp_path: Path) -> None:
        loader = _write_ragtruth(
            tmp_path / "ragtruth", [_source("1", "QA", "plain text")], [_response("1")]
        )

        with pytest.raises(DatasetFormatError):
            loader.load()

    def test_non_string_passages_raise_format_error(self, tmp_path: Path) -> None:
        loader = _write_ragtruth(
            tmp_path / "ragtruth",
            [_source("1", "QA", {"question": "q", "passages": ["a", "b"]})],
            [_response("1")],
        )

        with pytest.raises(DatasetFormatError, match="not a string"):
            loader.load()

    def test_unknown_split_raises_format_error(self, tmp_path: Path) -> None:
        loader = _write_ragtruth(
            tmp_path / "ragtruth", [QA_SOURCE], [_response("1", split="dev")]
        )

        with pytest.raises(DatasetFormatError, match="split"):
            loader.load()

    def test_labels_that_are_not_a_list_raise_format_error(self, tmp_path: Path) -> None:
        bad = {**_response("1"), "labels": "none"}
        loader = _write_ragtruth(tmp_path / "ragtruth", [QA_SOURCE], [bad])

        with pytest.raises(DatasetFormatError, match="labels"):
            loader.load()

    @pytest.mark.parametrize("missing", ["response", "labels", "split", "source_id"])
    def test_response_missing_a_field_raises_format_error(
        self, tmp_path: Path, missing: str
    ) -> None:
        record = _response("1")
        del record[missing]
        loader = _write_ragtruth(tmp_path / "ragtruth", [QA_SOURCE], [record])

        with pytest.raises(DatasetFormatError, match=missing):
            loader.load()

    def test_source_missing_task_type_raises_format_error(self, tmp_path: Path) -> None:
        broken = {"source_id": "1", "source_info": {"passages": "p"}}
        loader = _write_ragtruth(tmp_path / "ragtruth", [broken], [_response("1")])

        with pytest.raises(DatasetFormatError, match="task_type"):
            loader.load()

    def test_no_qa_samples_raises_format_error(self, tmp_path: Path) -> None:
        loader = _write_ragtruth(
            tmp_path / "ragtruth", [_source("2", "Summary", "text")], [_response("2")]
        )

        with pytest.raises(DatasetFormatError, match="zero"):
            loader.load()

    def test_cache_round_trip_keeps_group_and_split(self, tmp_path: Path) -> None:
        loader = _write_ragtruth(
            tmp_path / "ragtruth", [QA_SOURCE], [_response("1", "test", [SPAN])]
        )
        first_load = loader.load()

        for name in ("response.jsonl", "source_info.jsonl"):
            (tmp_path / "ragtruth" / name).unlink()

        assert loader.load() == first_load
        assert first_load[0].group_id == "1" and first_load[0].official_split == "test"


class TestGetDatasetLoader:
    def test_returns_halueval_loader(self, tmp_path: Path) -> None:
        assert isinstance(get_dataset_loader("halueval", tmp_path), HaluEvalLoader)

    def test_returns_ragtruth_loader(self, tmp_path: Path) -> None:
        loader = get_dataset_loader("ragtruth_qa", tmp_path)
        assert isinstance(loader, RagTruthLoader)
        assert loader.raw_path == tmp_path / "raw" / "ragtruth"

    def test_raises_on_unknown_dataset_name(self, tmp_path: Path) -> None:
        with pytest.raises(UnknownStrategyError):
            get_dataset_loader("made_up_dataset", tmp_path)