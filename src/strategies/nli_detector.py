"""
nli_detector.py

Zero-shot NLI strategy: a pretrained NLI cross-encoder scores whether the
response is entailed by the context, and 1 - P(entailment) is used as the
hallucination score.

Measured behaviour on HaluEval QA: AUC ~0.5 (no usable signal), before and
after several fixes (label-index check, fp16 vs fp32, padding, hypothesis
format). HaluEval responses are mostly short answer spans, which is not the
premise/hypothesis pairing NLI models are trained on. This detector is kept
as a documented negative result there; its fate on RAGTruth (full-sentence
responses) is decided by the protocol defined before running it.

Author: Marco Pérez Padilla
Date:   06-10-2026
"""

from __future__ import annotations

from src.errors import ModelLoadError
from src.strategies.base import HallucinationDetector
from src.utils import sort_indices_by_text_length

DEFAULT_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_CONTRADICTION_THRESHOLD = 0.5


class NLIDetector(HallucinationDetector):
    """Flags a response as a hallucination when an NLI model finds it is not
    entailed by (or is contradicted by) the given context.

    Args:
        model_name: Hugging Face Hub identifier for the NLI cross-encoder.
        batch_size: number of (premise, hypothesis) pairs per forward pass.
        max_sequence_length: truncation length for the tokenizer.
        contradiction_threshold: combined (contradiction + neutral)
            probability above which a response is flagged as unsupported.
            Unused by evaluate.py, which calibrates its own threshold.
        device: torch device string; auto-detected when left as ``None``.
        revision: commit of the Hub repository to load (tokenizer and model);
            ``None`` loads whatever ``main`` points to at that moment.
    """

    name = "nli"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
        device: str | None = None,
        revision: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.contradiction_threshold = contradiction_threshold
        self._device_name = device
        self._model = None
        self._tokenizer = None
        self._entailment_label_index: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device_name = self._device_name or ("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision=self.revision
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, revision=self.revision
            )
            self._model.to(device_name)
            self._model.eval()
            self._device_name = device_name

            label_to_index = {
                label.lower(): index
                for index, label in self._model.config.id2label.items()
            }
            self._entailment_label_index = label_to_index.get("entailment", 0)
        except Exception as exc:  # noqa: BLE001 - re-raised with project context
            raise ModelLoadError(self.model_name, exc) from exc

    def predict_batch(
        self, contexts: list[str], responses: list[str]
    ) -> tuple[list[bool], list[float]]:
        self._validate_batch(contexts, responses)
        self._ensure_loaded()

        import torch

        hallucination_scores = [0.0] * len(contexts)
        order = sort_indices_by_text_length(
            [f"{c} {r}" for c, r in zip(contexts, responses)]
        )

        with torch.no_grad():
            for batch_start in range(0, len(order), self.batch_size):
                batch_indices = order[batch_start : batch_start + self.batch_size]
                premises = [contexts[i] for i in batch_indices]
                hypotheses = [responses[i] for i in batch_indices]

                encoded = self._tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation="only_first",
                    max_length=self.max_sequence_length,
                    return_tensors="pt",
                ).to(self._device_name)

                use_amp = self._device_name == "cuda"
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    logits = self._model(**encoded).logits

                probabilities = torch.softmax(logits, dim=-1)
                entailment_probabilities = probabilities[:, self._entailment_label_index]
                batch_scores = (1.0 - entailment_probabilities).float().cpu().tolist()

                for original_index, score in zip(batch_indices, batch_scores):
                    hallucination_scores[original_index] = score

        labels = [score >= self.contradiction_threshold for score in hallucination_scores]
        return labels, hallucination_scores