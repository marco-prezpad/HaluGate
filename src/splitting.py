"""
splitting.py

Deterministic train/validation/test splits for HallucinationSample lists,
always by group: samples that share a group are never separated.

Two modes:

- ``split_samples``: ratio-based split (HaluEval). The group is the
  ``group_id`` when set, otherwise the context text: HaluEval produces two
  samples per record (grounded + hallucinated) that share the same
  context, and if one landed in train and its sibling in test the test set
  would leak information the model has effectively already seen.
- ``split_with_official_test``: the dataset's own test partition is kept as
  the test set (RAGTruth), and a validation set is carved out of the rest
  by group. For RAGTruth the group is the source document: all six
  responses to one source stay together.

``split_dataset`` picks the mode from the samples themselves.

Author: Marco Pérez Padilla
Date:   11-10-2026
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.data_loading import HallucinationSample
from src.errors import HaluGateError

OFFICIAL_TEST_SPLIT = "test"


class SplitRatioError(HaluGateError):
    """Raised when train/val/test ratios don't sum to 1.0."""

    def __init__(self, train: float, validation: float, test: float) -> None:
        self.train = train
        self.validation = validation
        self.test = test
        total = train + validation + test
        super().__init__(
            f"Split ratios must sum to 1.0, got {train} + {validation} + {test} = {total}"
        )


@dataclass(frozen=True)
class DatasetSplits:
    """The three partitions produced by the split functions."""

    train: list[HallucinationSample]
    validation: list[HallucinationSample]
    test: list[HallucinationSample]


def _group_key(sample: HallucinationSample) -> str:
    return sample.group_id or sample.context


def _group_in_order(samples: list[HallucinationSample]) -> dict[str, list[HallucinationSample]]:
    """Group samples by key, keeping first-appearance order of the groups
    and the original order of the samples inside each group.
    """
    groups: dict[str, list[HallucinationSample]] = {}
    for sample in samples:
        groups.setdefault(_group_key(sample), []).append(sample)
    return groups


def _flatten(
    groups: dict[str, list[HallucinationSample]], keys: list[str]
) -> list[HallucinationSample]:
    flattened: list[HallucinationSample] = []
    for key in keys:
        flattened.extend(groups[key])
    return flattened


def split_samples(
    samples: list[HallucinationSample],
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> DatasetSplits:
    """Split samples into train/validation/test by ratio, grouped.

    Raises:
        SplitRatioError: if the three ratios don't sum to 1.0 (within float
            tolerance).
        ValueError: if ``samples`` is empty.
    """
    if abs((train_ratio + validation_ratio + test_ratio) - 1.0) > 1e-9:
        raise SplitRatioError(train_ratio, validation_ratio, test_ratio)
    if not samples:
        raise ValueError("Cannot split an empty sample list.")

    groups = _group_in_order(samples)
    shuffled_keys = list(groups)
    random.Random(random_seed).shuffle(shuffled_keys)

    total_groups = len(shuffled_keys)
    train_cutoff = int(total_groups * train_ratio)
    validation_cutoff = train_cutoff + int(total_groups * validation_ratio)

    return DatasetSplits(
        train=_flatten(groups, shuffled_keys[:train_cutoff]),
        validation=_flatten(groups, shuffled_keys[train_cutoff:validation_cutoff]),
        test=_flatten(groups, shuffled_keys[validation_cutoff:]),
    )


def split_with_official_test(
    samples: list[HallucinationSample],
    validation_ratio: float = 0.15,
    random_seed: int = 42,
) -> DatasetSplits:
    """Keep the official test partition as test; carve validation out of the rest.

    ``validation_ratio`` is a fraction of the groups that are NOT in the
    official test (RAGTruth protocol: 15% of the official train, by
    source). The test set keeps the order of ``samples``.

    Raises:
        ValueError: if ``validation_ratio`` is not in (0, 1), if no sample is
            marked as official test or all of them are, or if there are too
            few groups outside the test to form a validation set.
    """
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(f"validation_ratio must be in (0, 1), got {validation_ratio}")

    test = [s for s in samples if s.official_split == OFFICIAL_TEST_SPLIT]
    remainder = [s for s in samples if s.official_split != OFFICIAL_TEST_SPLIT]
    if not test:
        raise ValueError("No sample is marked as official test.")
    if not remainder:
        raise ValueError("Every sample is marked as official test; nothing left to train on.")

    groups = _group_in_order(remainder)
    shuffled_keys = list(groups)
    random.Random(random_seed).shuffle(shuffled_keys)

    validation_group_count = int(len(shuffled_keys) * validation_ratio)
    if validation_group_count < 1 or validation_group_count >= len(shuffled_keys):
        raise ValueError(
            f"{len(shuffled_keys)} groups outside the test cannot form a validation "
            f"set at ratio {validation_ratio}."
        )

    return DatasetSplits(
        train=_flatten(groups, shuffled_keys[validation_group_count:]),
        validation=_flatten(groups, shuffled_keys[:validation_group_count]),
        test=test,
    )


def split_dataset(
    samples: list[HallucinationSample],
    split_ratios: dict[str, float],
    random_seed: int = 42,
) -> DatasetSplits:
    """Split ``samples`` using the dataset's official partition when it has one.

    If every sample carries an ``official_split``, the official test is kept
    and ``split_ratios["validation"]`` is read as a fraction of the rest.
    If none does, all three ratios are used as in ``split_samples``.

    Raises:
        ValueError: if ``samples`` is empty or only some samples carry an
            official split.
    """
    if not samples:
        raise ValueError("Cannot split an empty sample list.")

    with_official_split = sum(1 for s in samples if s.official_split)
    if with_official_split == len(samples):
        return split_with_official_test(samples, split_ratios["validation"], random_seed)
    if with_official_split > 0:
        raise ValueError("Only some samples carry an official split; the data is inconsistent.")

    return split_samples(
        samples,
        train_ratio=split_ratios["train"],
        validation_ratio=split_ratios["validation"],
        test_ratio=split_ratios["test"],
        random_seed=random_seed,
    )