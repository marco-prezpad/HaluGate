"""
data_loading.py

Loaders for the hallucination-detection datasets used by HaluGate. Each
loader implements the same interface (Strategy pattern, mirroring
src/strategies/) so evaluate.py can iterate over datasets exactly like it
iterates over detectors. Processed samples are cached to data/processed/ as
JSON Lines: re-running the pipeline after an interruption reuses the cache
instead of re-tokenizing and re-cleaning the raw files from scratch.

Author: Marco Pérez Padilla
Date:   04-10-2026
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from src.errors import DatasetFormatError, DatasetNotFoundError


@dataclass(frozen=True)
class HallucinationSample:
    """A single (context, response, label) triple used for training/evaluation."""

    context: str
    response: str
    is_hallucination: bool
    source_dataset: str


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
    """Loader for the HaluEval QA split (context, right/hallucinated answer pairs)."""

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
                    context = record["knowledge"]
                    right_answer = record["right_answer"]
                    hallucinated_answer = record["hallucinated_answer"]
                except (json.JSONDecodeError, KeyError) as exc:
                    raise DatasetFormatError(
                        self.name, f"line {line_number} missing expected fields: {exc}"
                    ) from exc

                samples.append(
                    HallucinationSample(context, right_answer, False, self.name)
                )
                samples.append(
                    HallucinationSample(context, hallucinated_answer, True, self.name)
                )

        if not samples:
            raise DatasetFormatError(self.name, "raw file parsed to zero samples")
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
    }
    if name not in registry:
        from src.errors import UnknownStrategyError

        raise UnknownStrategyError(name, list(registry.keys()))
    return registry[name]