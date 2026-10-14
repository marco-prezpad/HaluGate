"""
tokenization.py

Sentence-pair encoding shared by DistilBERT training and inference, so
both see exactly the same input.

Truncation rule (paper/protocol.md): when a (context, response) pair
exceeds the model's limit, the context is truncated and the response is
never touched ("only_first"). Hugging Face raises a bare ``Exception``
("Truncation error: Sequence to truncate too short...") when the response
alone is longer than the limit; that pair, and only that pair, falls back
to "longest_first" so a single oversize response cannot break a batch or
change how the others in it are truncated.

Author: Marco Pérez Padilla
Date:   12-10-2026
"""

from __future__ import annotations

from typing import Any

PRIMARY_TRUNCATION = "only_first"
FALLBACK_TRUNCATION = "longest_first"


def _is_truncation_error(error: Exception) -> bool:
    return "truncat" in str(error).lower()


def _encode_single(tokenizer: Any, context: str, response: str, max_length: int) -> Any:
    try:
        return tokenizer(context, response, truncation=PRIMARY_TRUNCATION, max_length=max_length)
    except Exception as error:  # noqa: BLE001 - Hugging Face raises a bare Exception
        if not _is_truncation_error(error):
            raise
        return tokenizer(context, response, truncation=FALLBACK_TRUNCATION, max_length=max_length)


def encode_pairs(
    tokenizer: Any,
    contexts: list[str],
    responses: list[str],
    max_length: int,
    return_tensors: str | None = None,
) -> Any:
    """Encode (context, response) pairs, truncating only the context.

    Args:
        tokenizer: a Hugging Face tokenizer.
        contexts: first sequences (the evidence), truncated if needed.
        responses: second sequences, kept whole unless one alone exceeds
            ``max_length``.
        max_length: token limit including special tokens.
        return_tensors: ``"pt"`` returns a padded batch of tensors (inference);
            ``None`` returns unpadded lists (training, where a collator pads).

    Raises:
        ValueError: if the inputs are empty or of different lengths.
    """
    if len(contexts) != len(responses):
        raise ValueError(
            f"contexts and responses must have equal length: {len(contexts)} != {len(responses)}"
        )
    if not contexts:
        raise ValueError("Cannot encode an empty batch.")

    padding = return_tensors is not None
    try:
        return tokenizer(
            contexts,
            responses,
            truncation=PRIMARY_TRUNCATION,
            max_length=max_length,
            padding=padding,
            return_tensors=return_tensors,
        )
    except Exception as error:  # noqa: BLE001 - Hugging Face raises a bare Exception
        if not _is_truncation_error(error):
            raise

    encodings = [
        _encode_single(tokenizer, context, response, max_length)
        for context, response in zip(contexts, responses)
    ]
    merged = {key: [encoding[key] for encoding in encodings] for key in encodings[0].keys()}
    if return_tensors is None:
        return merged
    return tokenizer.pad(merged, return_tensors=return_tensors)