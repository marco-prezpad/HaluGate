"""
test_nli_chunked.py

The real model needs a download and a forward pass, so the tokenizer and
the model are replaced by fakes with one rule: a (window, sentence) pair is
"entailed" when every word of the sentence appears in the window. That makes
the aggregation (max over windows, max over sentences of 1 - support)
testable with readable texts. Tokens are whitespace-separated words.

Author: Marco Pérez Padilla
Date:   20-10-2026
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import torch

from src.errors import EmptyBatchError
from src.factory import get_detector
from src.strategies.nli_chunked import NO_EVIDENCE_SCORE, NLIChunkedDetector


class _FakeEncoding(dict):
    def to(self, _device: str) -> "_FakeEncoding":
        return self


def _vocabulary(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


class _FakeTokenizer:
    """One call, two modes: with a single list it counts words (window
    sizing); with two lists it encodes whether each hypothesis is covered by
    its premise as ``input_ids[:, 0]`` (1 = covered).
    """

    def __init__(self) -> None:
        self.scored_pairs: list[tuple[str, str]] = []

    def __call__(
        self, first: list[str], second: list[str] | None = None, **_kwargs: Any
    ) -> _FakeEncoding:
        if second is None:
            return _FakeEncoding({"input_ids": [text.split() for text in first]})

        assert len(first) == len(second)
        flags = []
        for premise, hypothesis in zip(first, second):
            self.scored_pairs.append((premise, hypothesis))
            flags.append(1 if _vocabulary(hypothesis) <= _vocabulary(premise) else 0)
        return _FakeEncoding({"input_ids": torch.tensor(flags, dtype=torch.long).reshape(-1, 1)})


class _FakeModelOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeConfig:
    id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}


class _FakeModel:
    """Strong entailment when the flag is 1, strong contradiction otherwise."""

    config = _FakeConfig()

    def __call__(self, input_ids: torch.Tensor, **_kwargs: Any) -> _FakeModelOutput:
        covered = input_ids[:, 0] == 1
        logits = torch.zeros((input_ids.shape[0], 3))
        logits[covered, 2] = 8.0
        logits[~covered, 0] = 8.0
        return _FakeModelOutput(logits)

    def eval(self) -> "_FakeModel":
        return self

    def to(self, _device: str) -> "_FakeModel":
        return self


def _detector(max_window_tokens: int = 8, batch_size: int = 3) -> NLIChunkedDetector:
    detector = NLIChunkedDetector(
        batch_size=batch_size, max_window_tokens=max_window_tokens, device="cpu"
    )
    detector._model = _FakeModel()
    detector._tokenizer = _FakeTokenizer()
    detector._entailment_label_index = 2
    return detector


CONTEXT = "Paris is in France. Berlin is in Germany. Rome is in Italy."


class TestScoring:
    def test_a_supported_response_scores_low(self) -> None:
        _, scores = _detector().predict_batch([CONTEXT], ["Paris is in France."])

        assert scores[0] < 0.01

    def test_an_unsupported_response_scores_high(self) -> None:
        _, scores = _detector().predict_batch([CONTEXT], ["Madrid is in Spain."])

        assert scores[0] > 0.99

    def test_one_unsupported_sentence_makes_the_whole_response_suspicious(self) -> None:
        response = "Paris is in France. Madrid is in Spain. Rome is in Italy."

        _, scores = _detector().predict_batch([CONTEXT], [response])

        assert scores[0] > 0.99

    def test_support_is_the_maximum_over_windows(self) -> None:
        # With 4-word windows the evidence for each sentence sits in a
        # different window: the score must still be low.
        detector = _detector(max_window_tokens=4)

        _, scores = detector.predict_batch([CONTEXT], ["Rome is in Italy. Paris is in France."])

        assert scores[0] < 0.01

    def test_labels_follow_the_score(self) -> None:
        labels, _ = _detector().predict_batch(
            [CONTEXT, CONTEXT], ["Paris is in France.", "Madrid is in Spain."]
        )

        assert labels == [False, True]


class TestChunking:
    def test_each_sentence_is_scored_against_every_window(self) -> None:
        detector = _detector(max_window_tokens=4)

        detector.predict_batch([CONTEXT], ["Paris is in France. Rome is in Italy."])

        pairs = detector._tokenizer.scored_pairs
        windows = {premise for premise, _ in pairs}
        sentences = {hypothesis for _, hypothesis in pairs}
        assert len(windows) == 3 and sentences == {"Paris is in France.", "Rome is in Italy."}
        assert len(pairs) == 3 * 2

    def test_the_premise_is_a_window_and_never_the_whole_context_when_it_is_long(self) -> None:
        detector = _detector(max_window_tokens=4)

        detector.predict_batch([CONTEXT], ["Paris is in France."])

        assert all(len(premise.split()) <= 4 for premise, _ in detector._tokenizer.scored_pairs)

    def test_a_repeated_context_is_windowed_once(self) -> None:
        detector = _detector()
        calls: list[int] = []
        original = detector._count_tokens

        def _counting(texts: list[str]) -> list[int]:
            calls.append(len(texts))
            return original(texts)

        detector._count_tokens = _counting  # type: ignore[method-assign]

        detector.predict_batch([CONTEXT] * 4, ["Paris is in France."] * 4)

        assert len(calls) == 1


class TestExplain:
    def test_lists_every_sentence_with_its_support(self) -> None:
        detector = _detector()

        explained = detector.explain(CONTEXT, "Paris is in France. Madrid is in Spain.")

        assert [sentence for sentence, _ in explained] == [
            "Paris is in France.",
            "Madrid is in Spain.",
        ]
        assert explained[0][1] > 0.99 and explained[1][1] < 0.01

    def test_the_score_is_one_minus_the_weakest_support(self) -> None:
        detector = _detector()
        response = "Paris is in France. Madrid is in Spain."

        weakest = min(support for _, support in detector.explain(CONTEXT, response))
        _, scores = detector.predict_batch([CONTEXT], [response])

        assert scores[0] == pytest.approx(1.0 - weakest)

    def test_is_empty_without_sentences_or_without_context(self) -> None:
        detector = _detector()

        assert detector.explain(CONTEXT, " \n ") == []
        assert detector.explain("  ", "Paris is in France.") == []


class TestEdgeCases:
    def test_a_response_without_sentences_scores_one(self) -> None:
        _, scores = _detector().predict_batch([CONTEXT], ["  \n "])

        assert scores == [NO_EVIDENCE_SCORE]

    def test_a_context_without_text_scores_one(self) -> None:
        _, scores = _detector().predict_batch(["   "], ["Paris is in France."])

        assert scores == [NO_EVIDENCE_SCORE]

    def test_samples_without_evidence_do_not_disturb_the_others(self) -> None:
        detector = _detector()

        _, scores = detector.predict_batch(
            [CONTEXT, "  ", CONTEXT], ["Paris is in France.", "Anything.", "Madrid is in Spain."]
        )

        assert scores[0] < 0.01 and scores[1] == NO_EVIDENCE_SCORE and scores[2] > 0.99

    def test_results_come_back_in_input_order_whatever_the_lengths(self) -> None:
        contexts = [CONTEXT, "Short one here.", CONTEXT]
        responses = ["Madrid is in Spain.", "Short one here.", "Paris is in France."]

        _, scores = _detector(batch_size=2).predict_batch(contexts, responses)

        assert [score > 0.5 for score in scores] == [True, False, False]

    def test_batch_size_does_not_change_the_scores(self) -> None:
        contexts = [CONTEXT] * 3
        responses = [
            "Paris is in France.",
            "Madrid is in Spain.",
            "Rome is in Italy. Oslo is cold.",
        ]

        small = _detector(batch_size=1).predict_batch(contexts, responses)[1]
        large = _detector(batch_size=50).predict_batch(contexts, responses)[1]

        assert small == pytest.approx(large)

    def test_raises_on_an_empty_batch(self) -> None:
        with pytest.raises(EmptyBatchError):
            _detector().predict_batch([], [])

    def test_raises_on_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError):
            _detector().predict_batch([CONTEXT], ["a", "b"])

    def test_score_lies_within_the_unit_interval(self) -> None:
        _, scores = _detector().predict_batch([CONTEXT] * 2, ["Paris is in France.", "Oslo."])

        assert all(0.0 <= score <= 1.0 for score in scores)


class TestConfiguration:
    def test_the_factory_builds_it_with_the_fixed_parameters(self) -> None:
        detector = get_detector(
            "nli_chunked", {"max_window_tokens": 350, "batch_size": 4, "max_sequence_length": 512}
        )

        assert isinstance(detector, NLIChunkedDetector)
        assert detector.name == "nli_chunked"
        assert (detector.max_window_tokens, detector.batch_size) == (350, 4)

    def test_defaults_match_the_protocol(self) -> None:
        detector = NLIChunkedDetector()

        assert detector.model_name == "cross-encoder/nli-deberta-v3-base"
        assert detector.max_window_tokens == 350

    def test_it_never_runs_the_forward_pass_under_autocast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The protocol fixes fp32; the plain NLIDetector uses autocast on CUDA.
        def _forbidden(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("nli_chunked must run in fp32, without autocast")

        monkeypatch.setattr(torch, "autocast", _forbidden)
        detector = _detector()
        detector._device_name = "cuda"  # would enable autocast in the plain detector

        # The fake model and encoding ignore the device, so this only checks the call path.
        _, scores = detector.predict_batch([CONTEXT], ["Paris is in France."])

        assert scores[0] < 0.01