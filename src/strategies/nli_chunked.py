"""
nli_chunked.py

NLI scored per sentence against passage windows, the single configuration
fixed in paper/protocol.md. The plain NLIDetector gives the whole response
as one hypothesis and truncates the evidence, which is not the
premise/hypothesis pairing NLI models are trained on. Here:

1. the response is split into sentences (src/text_chunking.py);
2. the context is split into windows of at most ``max_window_tokens``;
3. every (window, sentence) pair is scored by the NLI model;
4. the support of a sentence is the maximum P(entailment) over the windows;
5. the response score is the maximum over its sentences of (1 - support).

A response with no sentences, or a context with no text, scores 1.0.

The model runs in fp32 (no autocast), as the protocol requires; the plain
NLIDetector uses autocast on CUDA, so this class does not reuse its
forward pass, only its loading code.

Author: Marco Pérez Padilla
Date:   20-10-2026
"""

from __future__ import annotations

from src.strategies.nli_detector import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTRADICTION_THRESHOLD,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_MODEL_NAME,
    NLIDetector,
)
from src.text_chunking import DEFAULT_WINDOW_TOKENS, split_sentences, window_text
from src.tokenization import encode_pairs
from src.utils import sort_indices_by_text_length

NO_EVIDENCE_SCORE = 1.0


class NLIChunkedDetector(NLIDetector):
    """Flags a response when some sentence is not entailed by any passage window.

    Args:
        model_name: Hugging Face Hub identifier for the NLI cross-encoder.
        batch_size: number of (window, sentence) pairs per forward pass.
        max_window_tokens: maximum tokens per passage window.
        max_sequence_length: truncation length of a (window, sentence) pair.
            With the default window of 350 only an unusually long sentence
            reaches it.
        contradiction_threshold: score above which a response is flagged.
            Unused by evaluate.py, which calibrates its own threshold.
        device: torch device string; auto-detected when left as ``None``.
    """

    name = "nli_chunked"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_window_tokens: int = DEFAULT_WINDOW_TOKENS,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
        device: str | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            max_sequence_length=max_sequence_length,
            contradiction_threshold=contradiction_threshold,
            device=device,
        )
        self.max_window_tokens = max_window_tokens

    def _count_tokens(self, texts: list[str]) -> list[int]:
        """Token count of each text, without special tokens."""
        if not texts:
            return []
        encoded = self._tokenizer(texts, add_special_tokens=False)
        return [len(token_ids) for token_ids in encoded["input_ids"]]

    def _build_pairs(
        self, contexts: list[str], responses: list[str]
    ) -> tuple[list[tuple[int, int, str, str]], dict[int, list[str]]]:
        """All (sample, sentence index, window, sentence) pairs of a batch.

        Also returns the sentences of every sample that has both sentences
        and windows; the others keep ``NO_EVIDENCE_SCORE``.
        """
        windows_by_context: dict[str, list[str]] = {}
        pairs: list[tuple[int, int, str, str]] = []
        sentences_by_sample: dict[int, list[str]] = {}

        for sample_index, (context, response) in enumerate(zip(contexts, responses)):
            if context not in windows_by_context:
                windows_by_context[context] = window_text(
                    context, self._count_tokens, self.max_window_tokens
                )
            windows = windows_by_context[context]
            sentences = split_sentences(response)
            if not windows or not sentences:
                continue

            sentences_by_sample[sample_index] = sentences
            for sentence_index, sentence in enumerate(sentences):
                for window in windows:
                    pairs.append((sample_index, sentence_index, window, sentence))
        return pairs, sentences_by_sample

    def _sentence_support(
        self, contexts: list[str], responses: list[str]
    ) -> dict[int, list[tuple[str, float]]]:
        """For each sample with evidence, every sentence with its support
        (maximum P(entailment) over the windows).
        """
        import torch

        pairs, sentences_by_sample = self._build_pairs(contexts, responses)

        support = {
            (sample_index, sentence_index): 0.0
            for sample_index, sentences in sentences_by_sample.items()
            for sentence_index in range(len(sentences))
        }

        order = sort_indices_by_text_length([f"{pair[2]} {pair[3]}" for pair in pairs])

        with torch.no_grad():
            for batch_start in range(0, len(order), self.batch_size):
                batch = [pairs[i] for i in order[batch_start : batch_start + self.batch_size]]

                encoded = encode_pairs(
                    self._tokenizer,
                    [pair[2] for pair in batch],
                    [pair[3] for pair in batch],
                    self.max_sequence_length,
                    return_tensors="pt",
                ).to(self._device_name)

                logits = self._model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1)
                entailment = probabilities[:, self._entailment_label_index].float().cpu().tolist()

                for (sample_index, sentence_index, _, _), probability in zip(batch, entailment):
                    key = (sample_index, sentence_index)
                    support[key] = max(support[key], probability)

        return {
            sample_index: [
                (sentence, support[(sample_index, sentence_index)])
                for sentence_index, sentence in enumerate(sentences)
            ]
            for sample_index, sentences in sentences_by_sample.items()
        }

    def explain(self, context: str, response: str) -> list[tuple[str, float]]:
        """Each sentence of ``response`` with its support, for inspection.

        Empty when the response has no sentences or the context no text. The
        response score is ``max(1 - support)`` over these sentences.
        """
        self._ensure_loaded()
        return self._sentence_support([context], [response]).get(0, [])

    def predict_batch(
        self, contexts: list[str], responses: list[str]
    ) -> tuple[list[bool], list[float]]:
        self._validate_batch(contexts, responses)
        self._ensure_loaded()

        supports = self._sentence_support(contexts, responses)

        scores = [NO_EVIDENCE_SCORE] * len(contexts)
        for sample_index, sentence_supports in supports.items():
            scores[sample_index] = max(1.0 - support for _, support in sentence_supports)

        labels = [score >= self.contradiction_threshold for score in scores]
        return labels, scores