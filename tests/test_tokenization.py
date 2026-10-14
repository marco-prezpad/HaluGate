"""
test_tokenization.py

Uses a real Hugging Face fast tokenizer built offline from a tiny
vocabulary (one word = one token, BERT-style [CLS] A [SEP] B [SEP] pair
template), so the truncation behaviour tested here is the library's actual
behaviour, not a mock of it. It is built with the ``tokenizers`` library
directly because the ``vocab_file`` constructor of the BERT tokenizer
classes changed between transformers versions.

Author: Marco Pérez Padilla
Date:   12-10-2026
"""

from __future__ import annotations

from typing import Any

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from src.tokenization import encode_pairs

VOCABULARY = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "alpha", "beta", "gamma", "delta"]
MAX_LENGTH = 16


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {token: index for index, token in enumerate(VOCABULARY)}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocabulary["[CLS]"]), ("[SEP]", vocabulary["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def _tokens(tokenizer: PreTrainedTokenizerFast, ids: list[int]) -> list[str]:
    return tokenizer.convert_ids_to_tokens(ids)


LONG_CONTEXT = " ".join(["alpha"] * 50)
SHORT_RESPONSE = "beta gamma delta"
LONG_RESPONSE = " ".join(["delta"] * 30)


def test_long_context_is_truncated_and_the_response_is_kept_whole(tokenizer) -> None:
    encoded = encode_pairs(tokenizer, [LONG_CONTEXT], [SHORT_RESPONSE], MAX_LENGTH)

    ids = encoded["input_ids"][0]
    assert len(ids) == MAX_LENGTH
    assert _tokens(tokenizer, ids)[-5:] == ["[SEP]", "beta", "gamma", "delta", "[SEP]"]


def test_a_pair_that_fits_is_left_untouched(tokenizer) -> None:
    encoded = encode_pairs(tokenizer, ["alpha alpha"], ["beta"], MAX_LENGTH)

    assert _tokens(tokenizer, encoded["input_ids"][0]) == [
        "[CLS]", "alpha", "alpha", "[SEP]", "beta", "[SEP]",
    ]


def test_a_response_longer_than_the_limit_falls_back_instead_of_raising(tokenizer) -> None:
    encoded = encode_pairs(tokenizer, [LONG_CONTEXT], [LONG_RESPONSE], MAX_LENGTH)

    assert len(encoded["input_ids"][0]) <= MAX_LENGTH


def test_one_oversize_response_does_not_change_how_the_others_are_truncated(tokenizer) -> None:
    encoded = encode_pairs(
        tokenizer,
        [LONG_CONTEXT, LONG_CONTEXT],
        [SHORT_RESPONSE, LONG_RESPONSE],
        MAX_LENGTH,
        return_tensors="pt",
    )

    assert tuple(encoded["input_ids"].shape) == (2, MAX_LENGTH)
    first_row = _tokens(tokenizer, encoded["input_ids"][0].tolist())
    assert first_row[-5:] == ["[SEP]", "beta", "gamma", "delta", "[SEP]"]


def test_tensor_output_is_padded_to_the_longest_pair_in_the_batch(tokenizer) -> None:
    encoded = encode_pairs(
        tokenizer, ["alpha", "alpha alpha alpha"], ["beta", "beta"], MAX_LENGTH, return_tensors="pt"
    )

    assert tuple(encoded["input_ids"].shape) == (2, 7)
    assert encoded["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 0, 0]
    assert encoded["attention_mask"][1].tolist() == [1] * 7


def test_without_tensors_the_output_is_unpadded_lists(tokenizer) -> None:
    encoded = encode_pairs(tokenizer, ["alpha", "alpha alpha alpha"], ["beta", "beta"], MAX_LENGTH)

    assert [len(ids) for ids in encoded["input_ids"]] == [5, 7]


def test_fallback_output_without_tensors_is_also_unpadded_lists(tokenizer) -> None:
    encoded = encode_pairs(
        tokenizer, [LONG_CONTEXT, "alpha"], [LONG_RESPONSE, "beta"], MAX_LENGTH
    )

    assert [len(ids) for ids in encoded["input_ids"]] == [MAX_LENGTH, 5]


def test_unrelated_errors_are_not_swallowed() -> None:
    class _BrokenTokenizer:
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        encode_pairs(_BrokenTokenizer(), ["c"], ["r"], MAX_LENGTH)


def test_rejects_mismatched_lengths(tokenizer) -> None:
    with pytest.raises(ValueError, match="equal length"):
        encode_pairs(tokenizer, ["a", "b"], ["c"], MAX_LENGTH)


def test_rejects_empty_input(tokenizer) -> None:
    with pytest.raises(ValueError, match="empty"):
        encode_pairs(tokenizer, [], [], MAX_LENGTH)