"""
test_training_data.py

Covers correct extraction and index-alignment of the three parallel lists,
and the empty-input error.

Author: Marco Pérez Padilla
Date:   16-09-2026
"""

from __future__ import annotations

import pytest

from src.data_loading import HallucinationSample
from src.training_data import samples_to_training_arrays


def test_extracts_parallel_lists_in_order() -> None:
    samples = [
        HallucinationSample("context A", "right A", False, "halueval"),
        HallucinationSample("context A", "wrong A", True, "halueval"),
    ]
    contexts, responses, labels = samples_to_training_arrays(samples)
    assert contexts == ["context A", "context A"]
    assert responses == ["right A", "wrong A"]
    assert labels == [0, 1]


def test_labels_are_plain_ints_not_bools() -> None:
    samples = [HallucinationSample("c", "r", True, "halueval")]
    _contexts, _responses, labels = samples_to_training_arrays(samples)
    assert labels == [1]
    assert isinstance(labels[0], int)
    assert not isinstance(labels[0], bool)


def test_raises_on_empty_sample_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        samples_to_training_arrays([])


def test_output_lists_stay_index_aligned() -> None:
    samples = [
        HallucinationSample(f"context {i}", f"response {i}", bool(i % 2), "halueval")
        for i in range(10)
    ]
    contexts, responses, labels = samples_to_training_arrays(samples)
    for i in range(10):
        assert contexts[i] == f"context {i}"
        assert responses[i] == f"response {i}"
        assert labels[i] == i % 2