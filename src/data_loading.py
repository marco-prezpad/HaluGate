"""
data_loading.py

Loaders for the hallucination-detection datasets used by HaluGate:
HaluEval QA (primary, synthetic hallucinations) and RAGTruth QA
(secondary, organic hallucinations). Each loader implements the same
interface (Strategy pattern, mirroring src/strategies/) so evaluate.py can
iterate over datasets exactly like it iterates over detectors. Processed
samples are cached to data/processed/ as JSON Lines: re-running the
pipeline after an interruption reuses the cache instead of re-parsing the
raw files from scratch.

Author: Marco Pérez Padilla
Date:   11-10-2026
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.errors import DatasetFormatError, DatasetNotFoundError

RAGTRUTH_RESPONSE_FILE = "response.jsonl"
RAGTRUTH_SOURCE_FILE = "source_info.jsonl"
RAGTRUTH_QA_TASK = "QA"
RAGTRUTH_SPLITS = ("train", "test")


@dataclass(frozen=True)
class HallucinationSample:
    """A single (context, response, label) triple used for training/evaluation.

    ``group_id`` ties together samples that must never be split apart (for
    RAGTruth, the source document); empty means "group by context text".
    ``official_split`` carries a dataset's own train/test partition when it
    has one; empty means the dataset defines none. ``model`` names the system
    that wrote the response when the dataset records it (RAGTruth); empty
    otherwise.
    """

    context: str
    response: str
    is_hallucination: bool
    source_dataset: str
    group_id: str = ""
    official_split: str = ""
    model: str = ""


class DatasetLoader(ABC):
    """Common interface every dataset loader must implement."""

    name: str

    @abstractmethod
    def load(self) -> list[HallucinationSample]:
        """Return the full list of samples for this dataset."""

    @abstractmethod
    def _read_raw(self) -> list[HallucinationSample]:
        """Parse the dataset's raw, on-disk format into samples."""


class CachingDatasetLoader(DatasetLoader):
    """Adds a processed-cache layer on top of a subclass's raw parsing logic.

    Subclasses only need to implement ``_read_raw``; caching, cache
    invalidation, and resumability are handled here so each concrete loader
    stays focused on its own file format (DRY).
    """

    def __init__(self, raw_path: Path, processed_dir: Path) -> None:
        self.raw_path = raw_path
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _cache_path(self) -> Path:
        return self.processed_dir / f"{self.name}.jsonl"

    def load(self) -> list[HallucinationSample]:
        if self._cache_path.exists():
            return self._read_cache()

        if not self.raw_path.exists():
            raise DatasetNotFoundError(str(self.raw_path))

        samples = self._read_raw()
        self._write_cache(samples)
        return samples

    def _read_cache(self) -> list[HallucinationSample]:
        samples = []
        with self._cache_path.open("r", encoding="utf-8") as cache_file:
            for line_number, line in enumerate(cache_file, start=1):
                try:
                    record = json.loads(line)
                    samples.append(HallucinationSample(**record))
                except (json.JSONDecodeError, TypeError) as exc:
                    raise DatasetFormatError(
                        self.name,
                        f"corrupted cache line {line_number} in {self._cache_path}: {exc}",
                    ) from exc
        return samples

    def _write_cache(self, samples: list[HallucinationSample]) -> None:
        tmp_path = self._cache_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as cache_file:
            for sample in samples:
                cache_file.write(json.dumps(sample.__dict__) + "\n")
        tmp_path.replace(self._cache_path)


class HaluEvalLoader(CachingDatasetLoader):
    """Loader for the HaluEval QA split (context, right/hallucinated answer pairs).

    The stored ``context`` is the ``knowledge`` passage only. The question
    is validated to be present (schema check) but deliberately NOT folded
    into the context: prepending it was tried and measured -- it did not
    rescue the zero-shot NLI detector (AUC stayed ~0.5 with or without it)
    and it let hallucinated answers that merely rephrase the question
    look supported. Keeping the context as pure evidence also matches how
    the RAGTruth loader defines it, so both datasets are scored under
    the same rule.
    """

    name = "halueval"

    def _read_raw(self) -> list[HallucinationSample]:
        samples: list[HallucinationSample] = []
        with self.raw_path.open("r", encoding="utf-8") as raw_file:
            for line_number, line in enumerate(raw_file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record["question"]  # schema check only, see class docstring
                    knowledge = record["knowledge"]
                    right_answer = record["right_answer"]
                    hallucinated_answer = record["hallucinated_answer"]
                except (json.JSONDecodeError, KeyError) as exc:
                    raise DatasetFormatError(
                        self.name, f"line {line_number} missing expected fields: {exc}"
                    ) from exc

                context = knowledge.strip()

                samples.append(
                    HallucinationSample(context, right_answer, False, self.name)
                )
                samples.append(
                    HallucinationSample(context, hallucinated_answer, True, self.name)
                )

        if not samples:
            raise DatasetFormatError(self.name, "raw file parsed to zero samples")
        return samples


class RagTruthLoader(CachingDatasetLoader):
    """Loader for the QA task of RAGTruth (organic, span-annotated hallucinations).

    ``raw_path`` is the directory holding the two official files:
    ``response.jsonl`` (one generated answer per line, with its annotated
    hallucination spans, quality flag and official train/test split) and
    ``source_info.jsonl`` (one source per line; for QA, the question and
    the retrieved passages).

    A response is positive when at least one hallucinated span is annotated.
    The context is ``source_info["passages"]`` verbatim, without the
    question. Responses are not filtered by their ``quality`` flag. The
    ``source_id`` is stored as ``group_id`` so all responses to one source
    stay on the same side of any split.
    """

    name = "ragtruth_qa"

    def _read_jsonl(self, path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
        if not path.exists():
            raise DatasetNotFoundError(str(path))
        with path.open("r", encoding="utf-8") as file_handle:
            for line_number, line in enumerate(file_handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetFormatError(
                        self.name, f"{path.name} line {line_number}: invalid JSON ({exc})"
                    ) from exc

    def _read_sources(self) -> tuple[dict[str, str], set[str]]:
        """Return ({source_id: passages} for QA sources, ids of other tasks)."""
        passages_by_source: dict[str, str] = {}
        other_task_ids: set[str] = set()

        for line_number, record in self._read_jsonl(self.raw_path / RAGTRUTH_SOURCE_FILE):
            try:
                source_id = str(record["source_id"])
                task_type = record["task_type"]
            except KeyError as exc:
                raise DatasetFormatError(
                    self.name, f"source_info line {line_number} missing field: {exc}"
                ) from exc

            if task_type != RAGTRUTH_QA_TASK:
                other_task_ids.add(source_id)
                continue

            try:
                passages = record["source_info"]["passages"]
            except (KeyError, TypeError) as exc:
                raise DatasetFormatError(
                    self.name, f"source_info line {line_number}: no QA passages ({exc!r})"
                ) from exc
            if not isinstance(passages, str):
                raise DatasetFormatError(
                    self.name, f"source_info line {line_number}: passages is not a string"
                )
            passages_by_source[source_id] = passages.strip()

        return passages_by_source, other_task_ids

    def _read_raw(self) -> list[HallucinationSample]:
        passages_by_source, other_task_ids = self._read_sources()

        samples: list[HallucinationSample] = []
        for line_number, record in self._read_jsonl(self.raw_path / RAGTRUTH_RESPONSE_FILE):
            try:
                source_id = str(record["source_id"])
                response = record["response"]
                labels = record["labels"]
                split = record["split"]
                model = record["model"]
            except KeyError as exc:
                raise DatasetFormatError(
                    self.name, f"response line {line_number} missing field: {exc}"
                ) from exc

            if source_id in other_task_ids:
                continue
            if source_id not in passages_by_source:
                raise DatasetFormatError(
                    self.name,
                    f"response line {line_number}: source_id {source_id!r} not in source_info",
                )
            if not isinstance(labels, list):
                raise DatasetFormatError(
                    self.name, f"response line {line_number}: labels is not a list"
                )
            if split not in RAGTRUTH_SPLITS:
                raise DatasetFormatError(
                    self.name, f"response line {line_number}: unknown split {split!r}"
                )

            samples.append(
                HallucinationSample(
                    context=passages_by_source[source_id],
                    response=response,
                    is_hallucination=bool(labels),
                    source_dataset=self.name,
                    group_id=source_id,
                    official_split=split,
                    model=str(model),
                )
            )

        if not samples:
            raise DatasetFormatError(self.name, "raw files parsed to zero QA samples")
        return samples


def get_dataset_loader(name: str, data_root: Path) -> DatasetLoader:
    """Factory mirroring src/factory.py, so datasets are selected the same
    declarative way as detectors, from config.yaml.
    """
    registry: dict[str, DatasetLoader] = {
        "halueval": HaluEvalLoader(
            raw_path=data_root / "raw" / "halueval_qa.jsonl",
            processed_dir=data_root / "processed",
        ),
        "ragtruth_qa": RagTruthLoader(
            raw_path=data_root / "raw" / "ragtruth",
            processed_dir=data_root / "processed",
        ),
    }
    if name not in registry:
        from src.errors import UnknownStrategyError

        raise UnknownStrategyError(name, list(registry.keys()))
    return registry[name]