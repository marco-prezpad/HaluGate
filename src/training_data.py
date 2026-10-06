"""
training_data.py

Extracts the parallel (contexts, responses, labels) lists that
scripts/train_distilbert.py needs from a list of HallucinationSample.

Author: Marco Pérez Padilla
Date:   06-10-2026
"""

from __future__ import annotations

from src.data_loading import HallucinationSample


def samples_to_training_arrays(
    samples: list[HallucinationSample],
) -> tuple[list[str], list[str], list[int]]:
    """Return parallel (contexts, responses, labels) lists for training/eval.

    Raises:
        ValueError: if ``samples`` is empty.
    """
    if not samples:
        raise ValueError("Cannot build training data from an empty sample list.")

    contexts = [sample.context for sample in samples]
    responses = [sample.response for sample in samples]
    labels = [int(sample.is_hallucination) for sample in samples]
    return contexts, responses, labels