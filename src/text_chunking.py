"""
text_chunking.py

Splits a response into sentences and a context into passage windows, as
fixed in paper/protocol.md for the nli_chunked detector:

- sentences: a regex on terminal punctuation (. ! ?) followed by whitespace,
  and on newlines. Texts are used as provided, so words with missing
  spaces ("end.Next") are not split;
- windows: the context is split into sentences with the same rule and the
  sentences are packed, in order and without overlap, into windows of at
  most ``max_tokens`` tokens. A single sentence longer than the limit is
  cut on word boundaries so that no window exceeds it.

Token counts come from a ``count_tokens`` callable (list of texts -> list of
counts), so this module does not depend on any tokenizer and the real one
is only plugged in by the detector.

Author: Marco Pérez Padilla
Date:   20-10-2026
"""

from __future__ import annotations

import re
from collections.abc import Callable

DEFAULT_WINDOW_TOKENS = 350

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")

TokenCounter = Callable[[list[str]], list[int]]


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into stripped, non-empty sentences.

    A boundary is whitespace after ``.``, ``!`` or ``?``, or any newline.
    Returns an empty list for empty or whitespace-only text.
    """
    return [piece.strip() for piece in _SENTENCE_BOUNDARY.split(text) if piece.strip()]


def _split_long_sentence(sentence: str, max_tokens: int, count_tokens: TokenCounter) -> list[str]:
    """Cut a sentence that exceeds ``max_tokens`` into word groups that fit.

    A word that alone exceeds the limit still gets its own group: it cannot
    be cut further without splitting inside a word, and the tokenizer
    truncates it later.
    """
    words = sentence.split()
    word_counts = count_tokens(words)

    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word, word_tokens in zip(words, word_counts):
        if current and current_tokens + word_tokens > max_tokens:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(word)
        current_tokens += word_tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def window_text(
    text: str,
    count_tokens: TokenCounter,
    max_tokens: int = DEFAULT_WINDOW_TOKENS,
) -> list[str]:
    """Pack the sentences of ``text`` into windows of at most ``max_tokens`` tokens.

    Windows keep the original order and never overlap. Returns an empty list
    when ``text`` has no sentences.

    Raises:
        ValueError: if ``max_tokens`` is not positive.
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")

    sentences = split_sentences(text)
    if not sentences:
        return []

    sentence_counts = count_tokens(sentences)

    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence, sentence_tokens in zip(sentences, sentence_counts):
        if sentence_tokens > max_tokens:
            if current:
                windows.append(" ".join(current))
                current, current_tokens = [], 0
            windows.extend(_split_long_sentence(sentence, max_tokens, count_tokens))
            continue
        if current and current_tokens + sentence_tokens > max_tokens:
            windows.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        windows.append(" ".join(current))
    return windows