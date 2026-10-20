"""
test_text_chunking.py

Covers sentence splitting (punctuation, newlines, text used as provided)
and passage windowing (packing, order, no overlap, the token limit, and a
sentence that alone exceeds it). Tokens are whitespace-separated words, so
the counts in every test can be read by eye.

Author: Marco Pérez Padilla
Date:   20-10-2026
"""

from __future__ import annotations

import pytest

from src.text_chunking import split_sentences, window_text


def _words(texts: list[str]) -> list[int]:
    return [len(text.split()) for text in texts]


class TestSplitSentences:
    def test_splits_after_terminal_punctuation(self) -> None:
        assert split_sentences("One. Two! Three? Four.") == ["One.", "Two!", "Three?", "Four."]

    def test_splits_on_newlines_even_without_punctuation(self) -> None:
        assert split_sentences("passage 1:\nfirst line\n\nsecond line") == [
            "passage 1:",
            "first line",
            "second line",
        ]

    def test_does_not_split_inside_a_number_or_a_word_without_spaces(self) -> None:
        assert split_sentences("It costs 3.5 euros. Done.Next") == [
            "It costs 3.5 euros.",
            "Done.Next",
        ]

    def test_strips_whitespace_and_drops_empty_pieces(self) -> None:
        assert split_sentences("  Hello.   \n  World.  ") == ["Hello.", "World."]

    @pytest.mark.parametrize("text", ["", "   ", "\n\n", " \n "])
    def test_empty_text_has_no_sentences(self, text: str) -> None:
        assert split_sentences(text) == []

    def test_a_text_without_punctuation_is_one_sentence(self) -> None:
        assert split_sentences("just some words") == ["just some words"]


class TestWindowText:
    def test_short_text_is_a_single_window(self) -> None:
        assert window_text("a b. c d.", _words, max_tokens=10) == ["a b. c d."]

    def test_sentences_are_packed_in_order_without_overlap(self) -> None:
        text = "a b c. d e f. g h i. j k l."

        windows = window_text(text, _words, max_tokens=6)

        assert windows == ["a b c. d e f.", "g h i. j k l."]

    def test_a_sentence_is_never_split_across_windows_when_it_fits(self) -> None:
        windows = window_text("a b c d. e f g h.", _words, max_tokens=5)

        assert windows == ["a b c d.", "e f g h."]

    def test_a_window_may_hold_exactly_the_limit_but_never_one_more(self) -> None:
        text = "a b c d. e f g h."

        assert window_text(text, _words, max_tokens=8) == ["a b c d. e f g h."]
        assert window_text(text, _words, max_tokens=7) == ["a b c d.", "e f g h."]

    def test_no_window_exceeds_the_limit(self) -> None:
        text = " ".join(f"w{i} w{i} w{i}." for i in range(40))

        windows = window_text(text, _words, max_tokens=7)

        assert all(len(window.split()) <= 7 for window in windows)

    def test_every_word_appears_exactly_once_and_in_order(self) -> None:
        text = " ".join(f"s{i} x y." for i in range(25))

        windows = window_text(text, _words, max_tokens=8)

        assert " ".join(windows).split() == text.split()

    def test_a_sentence_longer_than_the_limit_is_cut_on_word_boundaries(self) -> None:
        sentence = " ".join(f"w{i}" for i in range(10)) + "."

        windows = window_text(f"short one. {sentence} tail.", _words, max_tokens=4)

        assert windows[0] == "short one."
        assert all(len(window.split()) <= 4 for window in windows)
        assert " ".join(windows).split() == f"short one. {sentence} tail.".split()

    def test_a_single_word_above_the_limit_keeps_its_own_window(self) -> None:
        def _heavy(texts: list[str]) -> list[int]:
            return [10 if text.startswith("huge") else len(text.split()) for text in texts]

        windows = window_text("huge a b", _heavy, max_tokens=4)

        assert windows == ["huge", "a b"]

    def test_empty_text_has_no_windows(self) -> None:
        assert window_text("  \n ", _words, max_tokens=5) == []

    @pytest.mark.parametrize("max_tokens", [0, -3])
    def test_rejects_a_non_positive_limit(self, max_tokens: int) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            window_text("a b c.", _words, max_tokens=max_tokens)