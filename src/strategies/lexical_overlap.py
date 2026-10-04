"""
lexical_overlap.py

The cheapest strategy in the latency-accuracy frontier: flags a response as
a hallucination when its n-gram overlap with the context falls below a
threshold.

Author: Marco Pérez Padilla
Date:   04-10-2026
"""

from __future__ import annotations

import re

from src.strategies.base import HallucinationDetector


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _n_grams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


class LexicalOverlapDetector(HallucinationDetector):
    """Flags low n-gram overlap between context and response as a hallucination.

    Args:
        n_gram_size: size of the n-grams compared (default: trigrams).
        overlap_threshold: minimum fraction of response n-grams that must
            appear in the context for the response to be considered grounded.
            Higher threshold = stricter requirement = more flags.
    """

    name = "lexical"

    def __init__(self, n_gram_size: int = 3, overlap_threshold: float = 0.15) -> None:
        if n_gram_size < 1:
            raise ValueError(f"n_gram_size must be >= 1, got {n_gram_size}")
        if not 0.0 <= overlap_threshold <= 1.0:
            raise ValueError(f"overlap_threshold must be in [0, 1], got {overlap_threshold}")
        self.n_gram_size = n_gram_size
        self.overlap_threshold = overlap_threshold

    def _overlap_score(self, context: str, response: str) -> float:
        context_n_grams = _n_grams(_tokenize(context), self.n_gram_size)
        response_n_grams = _n_grams(_tokenize(response), self.n_gram_size)

        if not response_n_grams:
            # A response too short to form an n-gram is treated as ungrounded
            # rather than silently passing it as grounded.
            return 1.0

        matched = response_n_grams & context_n_grams
        grounded_fraction = len(matched) / len(response_n_grams)
        return 1.0 - grounded_fraction

    def predict_batch(
        self, contexts: list[str], responses: list[str]
    ) -> tuple[list[bool], list[float]]:
        self._validate_batch(contexts, responses)

        scores = [
            self._overlap_score(context, response)
            for context, response in zip(contexts, responses)
        ]
        labels = [score >= (1.0 - self.overlap_threshold) for score in scores]
        return labels, scores