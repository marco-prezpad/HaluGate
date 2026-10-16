"""
response_length.py

Reference baseline, not a detector: the score is just how long the
response is (in whitespace-separated words), squashed to [0, 1) with
n / (n + 1), which keeps the order of the lengths and so does not change
any AUC. It has no parameters and never looks at the context.

If a real detector does not clearly beat this baseline, its results may
come from the way the responses are written rather than from checking
them against the evidence. It is reported next to the detectors and is
excluded from the frontier rule (paper/protocol.md).

Author: Marco Pérez Padilla
Date:   14-10-2026
"""

from __future__ import annotations

from src.strategies.base import HallucinationDetector


class ResponseLengthDetector(HallucinationDetector):
    """Scores a response by its word count; longer means more suspicious."""

    name = "length"

    def predict_batch(
        self, contexts: list[str], responses: list[str]
    ) -> tuple[list[bool], list[float]]:
        self._validate_batch(contexts, responses)

        word_counts = [len(response.split()) for response in responses]
        scores = [count / (count + 1) for count in word_counts]
        # evaluate.py calibrates its own threshold; these labels only keep
        # the interface complete.
        labels = [score >= 0.5 for score in scores]
        return labels, scores