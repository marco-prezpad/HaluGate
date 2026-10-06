"""
test_splitting.py

Covers ratio validation, context-grouping (siblings never split across
partitions), determinism, and the empty-input error.

Author: Marco Pérez Padilla
Date:   05-10-2026
"""

from __future__ import annotations

import pytest

from src.data_loading import HallucinationSample
from src.splitting import SplitRatioError, split_samples


def _make_samples(num_contexts: int) -> list[HallucinationSample]:
    samples = []
    for i in range(num_contexts):
        context = f"context {i}"
        samples.append(HallucinationSample(context, f"right {i}", False, "halueval"))
        samples.append(HallucinationSample(context, f"wrong {i}", True, "halueval"))
    return samples


def test_raises_on_ratios_not_summing_to_one() -> None:
    with pytest.raises(SplitRatioError):
        split_samples(_make_samples(10), train_ratio=0.5, validation_ratio=0.3, test_ratio=0.3)


def test_raises_on_empty_samples() -> None:
    with pytest.raises(ValueError, match="empty"):
        split_samples([])


def test_sibling_samples_stay_in_the_same_split() -> None:
    samples = _make_samples(100)
    splits = split_samples(samples, random_seed=42)

    for partition in (splits.train, splits.validation, splits.test):
        contexts_in_partition = {sample.context for sample in partition}
        for context in contexts_in_partition:
            matching = [s for s in partition if s.context == context]
            assert len(matching) == 2


def test_split_sizes_are_roughly_proportional() -> None:
    samples = _make_samples(1000)
    splits = split_samples(samples, train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15)

    total = len(splits.train) + len(splits.validation) + len(splits.test)
    assert total == len(samples)
    assert 0.65 < len(splits.train) / total < 0.75


def test_same_seed_produces_same_split() -> None:
    samples = _make_samples(50)
    first = split_samples(samples, random_seed=7)
    second = split_samples(samples, random_seed=7)
    assert first.train == second.train
    assert first.test == second.test


def test_different_seeds_produce_different_splits() -> None:
    samples = _make_samples(200)
    first = split_samples(samples, random_seed=1)
    second = split_samples(samples, random_seed=2)
    assert first.train != second.train