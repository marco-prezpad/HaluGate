"""
test_distilbert_classifier.py

The real model requires a network download and a GPU/CPU forward pass we
don't want in unit tests, so the tokenizer and model are replaced with
lightweight fakes that mimic the Hugging Face sentence-pair interface.

Author: Marco Pérez Padilla
Date:   06-10-2026
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.errors import EmptyBatchError, ModelLoadError
from src.strategies.distilbert_classifier import DistilBERTDetector


class _FakeEncoding(dict):
    def to(self, _device: str) -> "_FakeEncoding":
        return self


class _FakeTokenizer:
    """Mimics the Hugging Face sentence-pair call signature: two parallel
    lists (contexts, responses), not one list of pre-joined strings.
    """

    def __call__(
        self, contexts: list[str], responses: list[str], **_kwargs: Any
    ) -> _FakeEncoding:
        assert len(contexts) == len(responses)
        return _FakeEncoding({"input_ids": torch.zeros((len(contexts), 4), dtype=torch.long)})


class _FakeModelOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeModel:
    """Produces the same strong hallucination-class logits for every item."""

    def __call__(self, input_ids: torch.Tensor, **_kwargs: Any) -> _FakeModelOutput:
        batch_size = input_ids.shape[0]
        logits = torch.zeros((batch_size, 2))
        logits[:, 1] = 5.0
        return _FakeModelOutput(logits)

    def eval(self) -> "_FakeModel":
        return self

    def to(self, _device: str) -> "_FakeModel":
        return self


@pytest.fixture()
def detector_with_fake_model() -> DistilBERTDetector:
    detector = DistilBERTDetector(model_path="fake/path", batch_size=2, device="cpu")
    detector._model = _FakeModel()
    detector._tokenizer = _FakeTokenizer()
    return detector


def test_predict_batch_returns_one_label_and_score_per_sample(
    detector_with_fake_model: DistilBERTDetector,
) -> None:
    labels, scores = detector_with_fake_model.predict_batch(
        ["context a", "context b", "a much longer context here for sorting"],
        ["response a", "response b", "response c"],
    )
    assert len(labels) == 3
    assert len(scores) == 3
    assert all(isinstance(label, bool) for label in labels)


def test_predict_batch_respects_decision_threshold() -> None:
    detector = DistilBERTDetector(
        model_path="fake/path", batch_size=2, device="cpu", decision_threshold=0.999
    )
    detector._model = _FakeModel()
    detector._tokenizer = _FakeTokenizer()

    labels, scores = detector.predict_batch(["ctx"], ["resp"])
    assert labels == [False]
    assert scores[0] < 0.999


def test_predict_batch_raises_on_empty_batch(detector_with_fake_model: DistilBERTDetector) -> None:
    with pytest.raises(EmptyBatchError):
        detector_with_fake_model.predict_batch([], [])


def test_ensure_loaded_wraps_failures_in_model_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = DistilBERTDetector(model_path="nonexistent/model", device="cpu")

    def _raise_os_error(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("model not found on hub")

    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(_raise_os_error)
    )

    with pytest.raises(ModelLoadError):
        detector.predict("context", "response")


def test_tokenizer_receives_context_and_response_as_separate_sequences() -> None:
    """Regression test: context and response must be passed as two parallel
    lists (sentence-pair tokenization), not pre-joined into one string with
    a literal "[SEP]" -- that was the original (buggy) implementation, and
    it must match how scripts/train_distilbert.py tokenizes during training.
    """
    received_args: dict[str, list[str]] = {}

    class _RecordingTokenizer(_FakeTokenizer):
        def __call__(self, contexts: list[str], responses: list[str], **kwargs: Any) -> _FakeEncoding:
            received_args["contexts"] = contexts
            received_args["responses"] = responses
            return super().__call__(contexts, responses, **kwargs)

    detector = DistilBERTDetector(model_path="fake/path", batch_size=10, device="cpu")
    detector._model = _FakeModel()
    detector._tokenizer = _RecordingTokenizer()

    detector.predict_batch(["the context"], ["the response"])

    assert received_args["contexts"] == ["the context"]
    assert received_args["responses"] == ["the response"]


def test_batching_processes_all_samples_across_multiple_batches() -> None:
    detector = DistilBERTDetector(model_path="fake/path", batch_size=2, device="cpu")
    detector._model = _FakeModel()
    detector._tokenizer = _FakeTokenizer()

    contexts = [f"context {i}" for i in range(5)]
    responses = [f"response {i}" for i in range(5)]
    labels, scores = detector.predict_batch(contexts, responses)

    assert len(labels) == 5
    assert len(scores) == 5


def test_score_returns_hallucinated_class_probability(
    detector_with_fake_model: DistilBERTDetector,
) -> None:
    assert detector_with_fake_model.score("ctx", "resp") == pytest.approx(0.9933, abs=1e-3)


def test_score_lies_within_unit_interval(
    detector_with_fake_model: DistilBERTDetector,
) -> None:
    score_value = detector_with_fake_model.score("ctx", "resp")
    assert 0.0 <= score_value <= 1.0


def test_only_the_context_is_truncated() -> None:
    seen: dict[str, Any] = {}

    class _KwargTokenizer(_FakeTokenizer):
        def __call__(self, contexts: list[str], responses: list[str], **kwargs: Any):
            seen.update(kwargs)
            return super().__call__(contexts, responses, **kwargs)

    detector = DistilBERTDetector(
        model_path="fake/path", batch_size=10, device="cpu", max_sequence_length=128
    )
    detector._model = _FakeModel()
    detector._tokenizer = _KwargTokenizer()

    detector.predict_batch(["context"], ["response"])

    assert seen["truncation"] == "only_first"
    assert seen["max_length"] == 128