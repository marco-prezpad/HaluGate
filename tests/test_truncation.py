"""
test_truncation.py

Covers the token counting (each unique context once), the statistics per
split and class, and, with a real Hugging Face tokenizer built locally, that
the counted truncation agrees with what src/tokenization.py does to the
same pairs.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from src.truncation import PairLength, count_tokens, pair_lengths, truncation_rows


class _WordTokenizer:
    """Counts whitespace-separated words and records the texts it is given."""

    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, list[list[str]]]:
        assert kwargs == {"add_special_tokens": False, "verbose": False}
        self.seen.append(list(texts))
        return {"input_ids": [text.split() for text in texts]}


class TestPairLengths:
    def test_counts_context_and_response_tokens(self) -> None:
        lengths = pair_lengths(_WordTokenizer(), ["a b c", "d e"], ["x", "y z w v"])

        assert lengths == [PairLength(3, 1), PairLength(2, 4)]

    def test_each_unique_context_is_tokenized_once(self) -> None:
        tokenizer = _WordTokenizer()

        lengths = pair_lengths(tokenizer, ["a b", "a b", "c", "a b"], ["x", "y", "z", "w"])

        assert tokenizer.seen[0] == ["a b", "c"]
        assert [length.context_tokens for length in lengths] == [2, 2, 1, 2]

    def test_empty_input_gives_no_lengths(self) -> None:
        assert pair_lengths(_WordTokenizer(), [], []) == []
        assert count_tokens(_WordTokenizer(), []) == []

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            pair_lengths(_WordTokenizer(), ["a"], ["x", "y"])


def _row(rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    return next(row for row in rows if row["group"] == group)


class TestTruncationRows:
    # max_length 10 with 3 special tokens: a budget of 7 tokens for the pair
    LENGTHS = [
        PairLength(2, 3),   # 5 <= 7: fits
        PairLength(6, 3),   # 9 > 7: truncated, the context keeps 4 of 6
        PairLength(5, 9),   # the response alone exceeds the budget
        PairLength(4, 3),   # 7 <= 7: fits exactly
    ]
    LABELS = [0, 1, 1, 0]

    def _rows(self) -> list[dict[str, Any]]:
        return truncation_rows("d", "arm", "test", self.LENGTHS, self.LABELS, 3, 10)

    def test_counts_truncated_pairs_and_the_fallback_case(self) -> None:
        row = _row(self._rows(), "all")

        assert (row["n"], row["truncated"], row["response_over_limit"]) == (4, 2, 1)
        assert row["truncated_pct"] == pytest.approx(50.0)

    def test_a_pair_that_fits_exactly_is_not_truncated(self) -> None:
        rows = truncation_rows("d", "arm", "test", [PairLength(4, 3)], [0], 3, 10)

        assert _row(rows, "all")["truncated"] == 0

    def test_the_share_of_context_kept_is_averaged_over_all_pairs(self) -> None:
        # kept: 1.0, 4/6, 0.0 (response fills the budget), 1.0
        expected = 100 * (1.0 + 4 / 6 + 0.0 + 1.0) / 4

        assert _row(self._rows(), "all")["context_kept_pct_mean"] == pytest.approx(expected)

    def test_classes_are_reported_apart(self) -> None:
        rows = self._rows()

        assert (_row(rows, "positive")["n"], _row(rows, "positive")["truncated"]) == (2, 2)
        assert (_row(rows, "negative")["n"], _row(rows, "negative")["truncated"]) == (2, 0)
        assert _row(rows, "positive")["truncated_pct"] == pytest.approx(100.0)

    def test_context_length_statistics(self) -> None:
        row = _row(self._rows(), "all")

        assert row["context_tokens_median"] == pytest.approx(4.5)
        assert 5 < row["context_tokens_p90"] <= 6

    def test_an_empty_context_loses_nothing(self) -> None:
        rows = truncation_rows("d", "arm", "test", [PairLength(0, 3)], [1], 3, 10)

        assert _row(rows, "all")["context_kept_pct_mean"] == pytest.approx(100.0)

    def test_a_group_without_pairs_is_omitted(self) -> None:
        rows = truncation_rows("d", "arm", "test", [PairLength(2, 3)], [0], 3, 10)

        assert {row["group"] for row in rows} == {"all", "negative"}

    def test_rows_identify_dataset_arm_and_split(self) -> None:
        rows = self._rows()

        assert {(r["dataset"], r["strategy"], r["split"]) for r in rows} == {("d", "arm", "test")}

    def test_no_pairs_gives_no_rows(self) -> None:
        assert truncation_rows("d", "arm", "test", [], [], 3, 10) == []

    @pytest.mark.parametrize(
        "lengths, labels, special_tokens, max_length",
        [
            ([PairLength(1, 1)], [0, 1], 3, 10),
            ([PairLength(1, 1)], [0], 10, 10),
            ([PairLength(1, 1)], [0], 12, 10),
        ],
    )
    def test_rejects_invalid_arguments(self, lengths, labels, special_tokens, max_length) -> None:
        with pytest.raises(ValueError):
            truncation_rows("d", "arm", "test", lengths, labels, special_tokens, max_length)


def _local_bert_like_tokenizer() -> Any:
    """A real Hugging Face fast tokenizer built locally (no download): one
    token per word, [CLS] a [SEP] b [SEP] for pairs."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}, "[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
        cls_token="[CLS]", sep_token="[SEP]",
    )


class TestAgreementWithTheRealTruncation:
    MAX_LENGTH = 40

    def _pairs(self) -> list[tuple[str, str]]:
        generator = random.Random(3)
        return [
            (
                " ".join("w" for _ in range(generator.randint(0, 60))),
                " ".join("r" for _ in range(generator.randint(1, 50))),
            )
            for _ in range(120)
        ]

    def test_special_tokens_of_a_pair_are_three(self) -> None:
        assert _local_bert_like_tokenizer().num_special_tokens_to_add(pair=True) == 3

    def test_counted_lengths_match_the_tokenizer(self) -> None:
        tokenizer = _local_bert_like_tokenizer()
        pairs = self._pairs()

        lengths = pair_lengths(tokenizer, [c for c, _ in pairs], [r for _, r in pairs])

        assert [(p.context_tokens, p.response_tokens) for p in lengths] == [
            (len(c.split()), len(r.split())) for c, r in pairs
        ]

    def test_truncated_and_kept_context_agree_with_encode_pairs(self) -> None:
        from src.tokenization import encode_pairs

        tokenizer = _local_bert_like_tokenizer()
        specials = tokenizer.num_special_tokens_to_add(pair=True)
        budget = self.MAX_LENGTH - specials

        checked_truncated = checked_fitting = 0
        for context, response in self._pairs():
            (length,) = pair_lengths(tokenizer, [context], [response])
            row = _row(
                truncation_rows("d", "a", "s", [length], [1], specials, self.MAX_LENGTH), "all"
            )
            encoded = encode_pairs(tokenizer, [context], [response], self.MAX_LENGTH)
            total = len(encoded["input_ids"][0])

            assert (row["truncated"] == 1) == (
                length.context_tokens + length.response_tokens > budget
            )
            if length.response_tokens > budget:
                assert total <= self.MAX_LENGTH  # fallback case: both texts are cut
                continue

            kept_tokens = total - length.response_tokens - specials
            assert total == min(length.context_tokens + length.response_tokens + specials,
                                self.MAX_LENGTH)
            if length.context_tokens:
                assert row["context_kept_pct_mean"] == pytest.approx(
                    100 * kept_tokens / length.context_tokens
                )
            checked_truncated += row["truncated"]
            checked_fitting += 1 - row["truncated"]

        assert checked_truncated > 10 and checked_fitting > 10