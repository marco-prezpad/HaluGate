"""
splitting.py

Deterministic train/validation/test split for HallucinationSample lists.
Splitting by (context, response) pairs rather than by record index matters
here: HaluEval produces two samples per record (grounded + hallucinated)
that share the same context. If one ends up in train and its sibling in
test, the test set leaks information the model has effectively already
seen. Splitting by the shared context keeps both siblings on the same side.

Author: Marco Pérez Padilla
Date:   05-10-2026
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.data_loading import HallucinationSample
from src.errors import HaluGateError


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
    """The three partitions produced by split_samples."""

    train: list[HallucinationSample]
    validation: list[HallucinationSample]
    test: list[HallucinationSample]


def split_samples(
    samples: list[HallucinationSample],
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> DatasetSplits:
    """Split samples into train/validation/test, grouped by shared context.

    Raises:
        SplitRatioError: if the three ratios don't sum to 1.0 (within float
            tolerance).
        ValueError: if ``samples`` is empty.
    """
    if abs((train_ratio + validation_ratio + test_ratio) - 1.0) > 1e-9:
        raise SplitRatioError(train_ratio, validation_ratio, test_ratio)
    if not samples:
        raise ValueError("Cannot split an empty sample list.")

    contexts_in_order: list[str] = []
    samples_by_context: dict[str, list[HallucinationSample]] = {}
    for sample in samples:
        if sample.context not in samples_by_context:
            contexts_in_order.append(sample.context)
            samples_by_context[sample.context] = []
        samples_by_context[sample.context].append(sample)

    shuffled_contexts = list(contexts_in_order)
    random.Random(random_seed).shuffle(shuffled_contexts)

    total_contexts = len(shuffled_contexts)
    train_cutoff = int(total_contexts * train_ratio)
    validation_cutoff = train_cutoff + int(total_contexts * validation_ratio)

    train_contexts = shuffled_contexts[:train_cutoff]
    validation_contexts = shuffled_contexts[train_cutoff:validation_cutoff]
    test_contexts = shuffled_contexts[validation_cutoff:]

    def _flatten(contexts: list[str]) -> list[HallucinationSample]:
        flattened: list[HallucinationSample] = []
        for context in contexts:
            flattened.extend(samples_by_context[context])
        return flattened

    return DatasetSplits(
        train=_flatten(train_contexts),
        validation=_flatten(validation_contexts),
        test=_flatten(test_contexts),
    )