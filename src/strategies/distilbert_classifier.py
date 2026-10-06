"""
distilbert_classifier.py

Mid-cost strategy in the latency-accuracy frontier: a DistilBERT model
fine-tuned as a binary sequence-pair classifier over (context, response).
Context and response are passed to the tokenizer as two separate
sequences -- the standard Hugging Face sentence-pair convention, which
inserts the real [CLS]/[SEP] special tokens and produces token_type_ids --
rather than concatenated into one string. This matches both how
scripts/train_distilbert.py tokenizes during fine-tuning and how
NLIDetector tokenizes its premise/hypothesis pairs, so training and
inference see identical input shapes.

Batching and length-bucketing keep inference latency reasonable without
resorting to anything exotic.

Author: Marco Pérez Padilla
Date:   06-10-2026
"""

from __future__ import annotations

from pathlib import Path

from src.errors import ModelLoadError
from src.strategies.base import HallucinationDetector
from src.utils import sort_indices_by_text_length

DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_DECISION_THRESHOLD = 0.5


class DistilBERTDetector(HallucinationDetector):
    """Binary hallucination classifier built on a fine-tuned DistilBERT model.

    Args:
        model_path: path or Hugging Face Hub identifier for the fine-tuned
            sequence-classification checkpoint.
        batch_size: number of samples per forward pass.
        max_sequence_length: truncation length for the tokenizer.
        decision_threshold: probability above which a sample is labeled a
            hallucination. Unused by evaluate.py, which calibrates its own
            threshold against the validation split -- kept for standalone
            use of this detector outside the evaluation pipeline.
        device: torch device string; auto-detected (cuda if available) when
            left as ``None``.
    """

    name = "distilbert"

    def __init__(
        self,
        model_path: str | Path,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
        device: str | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.decision_threshold = decision_threshold
        self._device_name = device
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first use so importing this module (e.g. for
        tests that mock predict_batch) never requires a network call or GPU.
        """
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device_name = self._device_name or ("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self._model.to(device_name)
            self._model.eval()
            self._device_name = device_name
        except Exception as exc:  # noqa: BLE001 - re-raised with project context
            raise ModelLoadError(self.model_path, exc) from exc

    def predict_batch(
        self, contexts: list[str], responses: list[str]
    ) -> tuple[list[bool], list[float]]:
        self._validate_batch(contexts, responses)
        self._ensure_loaded()

        import torch

        scores = [0.0] * len(contexts)

        # Sort by combined length before batching so each mini-batch has
        # similar-length sequences: less padding, fewer wasted FLOPs, no
        # exotic tricks required. Results are scattered back to original order.
        order = sort_indices_by_text_length(
            [f"{context} {response}" for context, response in zip(contexts, responses)]
        )

        with torch.no_grad():
            for batch_start in range(0, len(order), self.batch_size):
                batch_indices = order[batch_start : batch_start + self.batch_size]
                batch_contexts = [contexts[i] for i in batch_indices]
                batch_responses = [responses[i] for i in batch_indices]

                encoded = self._tokenizer(
                    batch_contexts,
                    batch_responses,
                    padding=True,
                    truncation=True,
                    max_length=self.max_sequence_length,
                    return_tensors="pt",
                ).to(self._device_name)

                use_amp = self._device_name == "cuda"
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    logits = self._model(**encoded).logits

                batch_scores = torch.softmax(logits, dim=-1)[:, 1].float().cpu().tolist()
                for original_index, score in zip(batch_indices, batch_scores):
                    scores[original_index] = score

        labels = [score >= self.decision_threshold for score in scores]
        return labels, scores