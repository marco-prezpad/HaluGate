"""
truncation.py

How many (context, response) pairs a model with a token limit cannot read
in full. paper/protocol.md declares that the DistilBERT arms lose evidence
to truncation while nli_chunked does not, and asks for the share of test
pairs that exceed the limit to be reported together with that limitation.

The rule is the one of src/tokenization.py: the context is truncated and
the response never is ("only_first"); the budget for the context is
``max_length`` minus the special tokens minus the response tokens. A pair
whose response alone exceeds the budget is the fallback case (both texts
are cut) and is counted apart.

Lengths are measured with the real tokenizer of each model, counting each
unique context once (RAGTruth repeats every context in six responses).

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

GROUPS = (("all", None), ("positive", 1), ("negative", 0))


@dataclass(frozen=True)
class PairLength:
    """Token counts of one pair, without special tokens."""

    context_tokens: int
    response_tokens: int


def count_tokens(tokenizer: Any, texts: list[str]) -> list[int]:
    """Token count of each text, without special tokens and without the
    "sequence longer than the maximum" warning (nothing is truncated here).
    """
    if not texts:
        return []
    encoded = tokenizer(texts, add_special_tokens=False, verbose=False)
    return [len(token_ids) for token_ids in encoded["input_ids"]]


def pair_lengths(tokenizer: Any, contexts: list[str], responses: list[str]) -> list[PairLength]:
    """Token counts of every (context, response) pair.

    Raises:
        ValueError: if ``contexts`` and ``responses`` differ in length.
    """
    if len(contexts) != len(responses):
        raise ValueError(
            f"contexts and responses must have equal length: {len(contexts)} != {len(responses)}"
        )

    unique_contexts = list(dict.fromkeys(contexts))
    context_counts = dict(zip(unique_contexts, count_tokens(tokenizer, unique_contexts)))
    response_counts = count_tokens(tokenizer, responses)
    return [
        PairLength(context_counts[context], response_count)
        for context, response_count in zip(contexts, response_counts)
    ]


def truncation_rows(
    dataset_name: str,
    strategy_name: str,
    split_name: str,
    lengths: list[PairLength],
    labels: list[int],
    special_tokens: int,
    max_length: int,
) -> list[dict[str, Any]]:
    """Truncation statistics of one split, for all pairs and per class.

    ``context_kept_pct_mean`` is the average share of the context that the
    model sees under the "only_first" rule (100 for pairs that fit, 0 when
    the response alone fills the budget). Groups without pairs are omitted.

    Raises:
        ValueError: if ``lengths`` and ``labels`` differ in length, or if
            ``max_length`` leaves no room after the special tokens.
    """
    if len(lengths) != len(labels):
        raise ValueError(
            f"lengths and labels must have equal length: {len(lengths)} != {len(labels)}"
        )
    budget = max_length - special_tokens
    if budget < 1:
        raise ValueError(
            f"max_length {max_length} leaves no room after {special_tokens} special tokens"
        )

    rows: list[dict[str, Any]] = []
    for group_name, wanted_label in GROUPS:
        selected = [
            length
            for length, label in zip(lengths, labels)
            if wanted_label is None or label == wanted_label
        ]
        if not selected:
            continue

        truncated = sum(1 for p in selected if p.context_tokens + p.response_tokens > budget)
        response_over_limit = sum(1 for p in selected if p.response_tokens > budget)
        kept = [
            1.0
            if p.context_tokens == 0
            else min(1.0, max(0, budget - p.response_tokens) / p.context_tokens)
            for p in selected
        ]
        context_tokens = [p.context_tokens for p in selected]
        rows.append(
            {
                "dataset": dataset_name,
                "strategy": strategy_name,
                "split": split_name,
                "group": group_name,
                "n": len(selected),
                "truncated": truncated,
                "truncated_pct": 100.0 * truncated / len(selected),
                "response_over_limit": response_over_limit,
                "context_tokens_median": float(np.median(context_tokens)),
                "context_tokens_p90": float(np.percentile(context_tokens, 90)),
                "context_kept_pct_mean": 100.0 * float(np.mean(kept)),
            }
        )
    return rows