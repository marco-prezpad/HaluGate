"""
test_splitting.py

Covers ratio validation, grouping (siblings never split across
partitions), determinism, the official-test mode used for RAGTruth, and
the dispatcher that chooses between the two.

Author: Marco Pérez Padilla
Date:   11-10-2026
"""

from __future__ import annotations

import pytest

from src.data_loading import HallucinationSample
from src.splitting import (
    SplitRatioError,
    split_dataset,
    split_samples,
    split_with_official_test,
)

RATIOS = {"train": 0.7, "validation": 0.15, "test": 0.15}


def _make_samples(num_contexts: int) -> list[HallucinationSample]:
    samples = []
    for i in range(num_contexts):
        context = f"context {i}"
        samples.append(HallucinationSample(context, f"right {i}", False, "halueval"))
        samples.append(HallucinationSample(context, f"wrong {i}", True, "halueval"))
    return samples


def _make_official_samples(
    train_sources: int, test_sources: int, per_source: int = 6
) -> list[HallucinationSample]:
    samples = []
    for source in range(train_sources + test_sources):
        split = "train" if source < train_sources else "test"
        for response in range(per_source):
            samples.append(
                HallucinationSample(
                    context=f"passages {source}",
                    response=f"response {source}-{response}",
                    is_hallucination=response % 2 == 0,
                    source_dataset="ragtruth_qa",
                    group_id=f"s{source}",
                    official_split=split,
                )
            )
    return samples


class TestSplitSamples:
    def test_raises_on_ratios_not_summing_to_one(self) -> None:
        with pytest.raises(SplitRatioError):
            split_samples(_make_samples(10), train_ratio=0.5, validation_ratio=0.3, test_ratio=0.3)

    def test_raises_on_empty_samples(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            split_samples([])

    def test_sibling_samples_stay_in_the_same_split(self) -> None:
        splits = split_samples(_make_samples(100), random_seed=42)

        for partition in (splits.train, splits.validation, splits.test):
            for context in {sample.context for sample in partition}:
                assert len([s for s in partition if s.context == context]) == 2

    def test_split_sizes_are_roughly_proportional(self) -> None:
        samples = _make_samples(1000)
        splits = split_samples(samples)

        total = len(splits.train) + len(splits.validation) + len(splits.test)
        assert total == len(samples)
        assert 0.65 < len(splits.train) / total < 0.75

    def test_same_seed_produces_same_split(self) -> None:
        samples = _make_samples(50)
        first = split_samples(samples, random_seed=7)
        second = split_samples(samples, random_seed=7)
        assert first.train == second.train
        assert first.test == second.test

    def test_different_seeds_produce_different_splits(self) -> None:
        samples = _make_samples(200)
        assert split_samples(samples, random_seed=1).train != split_samples(samples, random_seed=2).train

    def test_explicit_group_id_groups_samples_with_different_contexts(self) -> None:
        samples = [
            HallucinationSample(f"context {i}-{j}", "r", bool(j % 2), "x", group_id=f"g{i}")
            for i in range(60)
            for j in range(4)
        ]

        splits = split_samples(samples)

        for partition in (splits.train, splits.validation, splits.test):
            for group in {s.group_id for s in partition}:
                assert len([s for s in partition if s.group_id == group]) == 4

    def test_empty_group_id_behaves_like_grouping_by_context(self) -> None:
        by_context = _make_samples(120)
        by_group_id = [
            HallucinationSample(s.context, s.response, s.is_hallucination, s.source_dataset, group_id=s.context)
            for s in by_context
        ]

        first = split_samples(by_context, random_seed=9)
        second = split_samples(by_group_id, random_seed=9)

        assert [s.response for s in first.train] == [s.response for s in second.train]
        assert [s.response for s in first.test] == [s.response for s in second.test]


class TestSplitWithOfficialTest:
    def test_official_test_is_kept_untouched_and_in_order(self) -> None:
        samples = _make_official_samples(train_sources=40, test_sources=10)

        splits = split_with_official_test(samples)

        assert splits.test == [s for s in samples if s.official_split == "test"]

    def test_nothing_from_the_official_test_leaks_into_train_or_validation(self) -> None:
        splits = split_with_official_test(_make_official_samples(40, 10))

        for partition in (splits.train, splits.validation):
            assert all(s.official_split == "train" for s in partition)

    def test_validation_is_a_fraction_of_the_groups_outside_the_test(self) -> None:
        splits = split_with_official_test(_make_official_samples(100, 10), validation_ratio=0.15)

        assert len({s.group_id for s in splits.validation}) == 15
        assert len({s.group_id for s in splits.train}) == 85

    def test_every_source_stays_on_one_side(self) -> None:
        splits = split_with_official_test(_make_official_samples(60, 10))

        train_groups = {s.group_id for s in splits.train}
        validation_groups = {s.group_id for s in splits.validation}
        assert train_groups.isdisjoint(validation_groups)
        assert len(splits.train) == len(train_groups) * 6
        assert len(splits.validation) == len(validation_groups) * 6

    def test_no_sample_is_lost_or_duplicated(self) -> None:
        samples = _make_official_samples(60, 10)
        splits = split_with_official_test(samples)

        assert len(splits.train) + len(splits.validation) + len(splits.test) == len(samples)

    def test_same_seed_gives_the_same_validation_set(self) -> None:
        samples = _make_official_samples(60, 10)
        assert split_with_official_test(samples, random_seed=3).validation == (
            split_with_official_test(samples, random_seed=3).validation
        )

    def test_different_seed_gives_a_different_validation_set(self) -> None:
        samples = _make_official_samples(60, 10)
        assert split_with_official_test(samples, random_seed=1).validation != (
            split_with_official_test(samples, random_seed=2).validation
        )

    @pytest.mark.parametrize("invalid_ratio", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_invalid_validation_ratio(self, invalid_ratio: float) -> None:
        with pytest.raises(ValueError, match="validation_ratio"):
            split_with_official_test(_make_official_samples(40, 10), validation_ratio=invalid_ratio)

    def test_raises_when_there_is_no_official_test(self) -> None:
        with pytest.raises(ValueError, match="official test"):
            split_with_official_test(_make_official_samples(40, 0))

    def test_raises_when_everything_is_official_test(self) -> None:
        with pytest.raises(ValueError, match="nothing left"):
            split_with_official_test(_make_official_samples(0, 10))

    def test_raises_when_too_few_groups_for_a_validation_set(self) -> None:
        with pytest.raises(ValueError, match="cannot form a validation"):
            split_with_official_test(_make_official_samples(3, 5), validation_ratio=0.15)


class TestSplitDataset:
    def test_uses_the_official_partition_when_every_sample_has_one(self) -> None:
        samples = _make_official_samples(60, 10)

        splits = split_dataset(samples, RATIOS)

        assert splits.test == [s for s in samples if s.official_split == "test"]

    def test_validation_ratio_is_read_as_a_fraction_of_the_non_test_groups(self) -> None:
        splits = split_dataset(_make_official_samples(100, 10), RATIOS)

        assert len({s.group_id for s in splits.validation}) == 15

    def test_uses_ratios_when_no_sample_has_an_official_split(self) -> None:
        samples = _make_samples(200)

        splits = split_dataset(samples, RATIOS, random_seed=5)

        expected = split_samples(samples, 0.7, 0.15, 0.15, random_seed=5)
        assert splits.test == expected.test

    def test_raises_on_empty_input(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            split_dataset([], RATIOS)

    def test_raises_when_only_some_samples_have_an_official_split(self) -> None:
        mixed = _make_official_samples(40, 10) + _make_samples(5)

        with pytest.raises(ValueError, match="inconsistent"):
            split_dataset(mixed, RATIOS)