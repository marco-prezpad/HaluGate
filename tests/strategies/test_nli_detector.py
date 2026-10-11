"""
test_nli_detector.py

Mirrors test_distilbert_classifier.py's mocking approach, adapted for the
NLI detector's premise/hypothesis input signature and entailment-label
lookup logic.

Author: Marco Pérez Padilla
Date:   06-10-2026
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.errors import EmptyBatchError, ModelLoadError
from src.strategies.nli_detector import NLIDetector


class _FakeEncoding(dict):
    def to(self, _device: str) -> "_FakeEncoding":
        return self


class _FakeTokenizer:
    def __call__(
        self, premises: list[str], hypotheses: list[str], **_kwargs: Any
    ) -> _FakeEncoding:
        batch_size = len(premises)
        assert len(hypotheses) == batch_size
        return _FakeEncoding({"input_ids": torch.zeros((batch_size, 4), dtype=torch.long)})


class _FakeModelOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeConfig:
    id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}


class _FakeModel:
    """Always predicts strong entailment (index 2) -- i.e. never a hallucination."""

    config = _FakeConfig()

    def __call__(self, input_ids: torch.Tensor, **_kwargs: Any) -> _FakeModelOutput:
        batch_size = input_ids.shape[0]
        logits = torch.zeros((batch_size, 3))
        logits[:, 2] = 5.0
        return _FakeModelOutput(logits)

    def eval(self) -> "_FakeModel":
        return self

    def to(self, _device: str) -> "_FakeModel":
        return self


@pytest.fixture()
def detector_with_fake_model() -> NLIDetector:
    detector = NLIDetector(batch_size=2, device="cpu")
    detector._model = _FakeModel()
    detector._tokenizer = _FakeTokenizer()
    detector._entailment_label_index = 2
    return detector


def test_strong_entailment_is_not_flagged_as_hallucination(
    detector_with_fake_model: NLIDetector,
) -> None:
    labels, scores = detector_with_fake_model.predict_batch(["context"], ["response"])
    assert labels == [False]
    assert scores[0] < 0.5


def test_predict_batch_returns_one_result_per_sample(
    detector_with_fake_model: NLIDetector,
) -> None:
    contexts = ["ctx a", "a much longer context to force a length-sort reorder", "ctx c"]
    responses = ["resp a", "resp b", "resp c"]
    labels, scores = detector_with_fake_model.predict_batch(contexts, responses)
    assert len(labels) == 3
    assert len(scores) == 3


def test_predict_batch_raises_on_empty_batch(detector_with_fake_model: NLIDetector) -> None:
    with pytest.raises(EmptyBatchError):
        detector_with_fake_model.predict_batch([], [])


def test_ensure_loaded_wraps_failures_in_model_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = NLIDetector(device="cpu")

    def _raise_os_error(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("model not found on hub")

    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(_raise_os_error)
    )

    with pytest.raises(ModelLoadError):
        detector.predict("context", "response")


def test_entailment_label_index_resolved_case_insensitively() -> None:
    """Some checkpoints use 'ENTAILMENT' or mixed case in id2label."""
    detector = NLIDetector(device="cpu")

    class _MixedCaseConfig:
        id2label = {0: "CONTRADICTION", 1: "Neutral", 2: "Entailment"}

    fake_model = _FakeModel()
    fake_model.config = _MixedCaseConfig()

    import transformers

    class _FromPretrainedStub:
        @staticmethod
        def from_pretrained(*_args: Any, **_kwargs: Any) -> Any:
            return fake_model

    detector._model = None
    original_tokenizer_cls = transformers.AutoTokenizer
    original_model_cls = transformers.AutoModelForSequenceClassification
    try:
        transformers.AutoTokenizer = type(
            "FakeTokenizerCls", (), {"from_pretrained": staticmethod(lambda *_a, **_k: _FakeTokenizer())}
        )
        transformers.AutoModelForSequenceClassification = _FromPretrainedStub
        detector._ensure_loaded()
    finally:
        transformers.AutoTokenizer = original_tokenizer_cls
        transformers.AutoModelForSequenceClassification = original_model_cls

    assert detector._entailment_label_index == 2


def test_score_is_one_minus_entailment_probability(
    detector_with_fake_model: NLIDetector,
) -> None:
    # The fake model emits logits [0, 0, 5] over (contradiction, neutral,
    # entailment); entailment probability ~ 0.9866, so the score ~ 0.0134.
    assert detector_with_fake_model.score("ctx", "resp") == pytest.approx(0.0134, abs=1e-3)


def test_score_lies_within_unit_interval(detector_with_fake_model: NLIDetector) -> None:
    score_value = detector_with_fake_model.score("ctx", "resp")
    assert 0.0 <= score_value <= 1.0