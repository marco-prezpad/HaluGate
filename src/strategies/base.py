"""
base.py

Abstract interface shared by every hallucination-detection strategy.
Subclasses only implement predict_batch; predict() and score() are derived
from it here.

Author: Marco Pérez Padilla
Date:   04-10-2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.errors import EmptyBatchError


class HallucinationDetector(ABC):
    """Common interface for all hallucination-detection strategies."""

    name: str

    @abstractmethod
    def predict_batch(
        self, contexts: list[str], responses: list[str]
    ) -> tuple[list[bool], list[float]]:
        """Score a batch of (context, response) pairs.

        Args:
            contexts: the context/knowledge given to the LLM, one per sample.
            responses: the LLM's generated response, one per sample.

        Returns:
            A tuple ``(predicted_labels, predicted_scores)`` where
            ``predicted_labels[i]`` is ``True`` if sample ``i`` is judged a
            hallucination, and ``predicted_scores[i]`` is a continuous score
            in [0, 1] (higher = more likely hallucinated) suitable for AUC-ROC.

        Raises:
            EmptyBatchError: if ``contexts`` and ``responses`` are empty.
            ValueError: if ``contexts`` and ``responses`` have different lengths.
        """

    def predict(self, context: str, response: str) -> bool:
        """Return the binary decision for a single (context, response) pair."""
        labels, _ = self.predict_batch([context], [response])
        return labels[0]

    def score(self, context: str, response: str) -> float:
        """Return the continuous hallucination score (in [0, 1]) for a single pair.

        Delegates to predict_batch rather than having its own code path, so
        it can never drift out of sync with the batched implementation. For
        scoring many samples, call predict_batch directly instead of looping
        over this method -- it's the version that batches and optimizes.
        """
        _, scores = self.predict_batch([context], [response])
        return scores[0]

    def _validate_batch(self, contexts: list[str], responses: list[str]) -> None:
        if len(contexts) != len(responses):
            raise ValueError(
                f"contexts and responses must have equal length: "
                f"{len(contexts)} != {len(responses)}"
            )
        if len(contexts) == 0:
            raise EmptyBatchError(self.name)