#!/usr/bin/env python3
"""
nli_contact.py

First contact of nli_chunked with the real model, before the protocol run.
It answers three questions and nothing else:

1. Does it load and run (device, fp32, versions)?
2. Do the windows respect the token limit with the real tokenizer, and does
   the evidence at the end of a long context still reach the model?
3. Are the scores on a few hand-written cases in the expected direction?

Optionally it also scores a small random sample of the official RAGTruth
TRAIN partition, to check that the pipeline runs on real inputs (counts of
sentences and windows, mean score per class, time per response). It never
reads the test partition and it does not compute an AUC on purpose: nothing
seen here may change the configuration fixed in paper/protocol.md.

The expected-direction checks use loose bounds (low < 0.3, high > 0.7). They
are a smoke test of the plumbing, not a measure of quality.

Usage:
    python scripts/nli_contact.py [--ragtruth-dir data/raw/ragtruth] [--n-train 40]
    python scripts/nli_contact.py --n-train 0        # hand-written cases only

Author: Marco Pérez Padilla
Date:   20-10-2026
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategies.nli_chunked import NLIChunkedDetector  # noqa: E402
from src.strategies.nli_detector import NLIDetector  # noqa: E402
from src.text_chunking import split_sentences, window_text  # noqa: E402

LOW_BOUND = 0.3
HIGH_BOUND = 0.7
SEED = 42

PARIS = (
    "passage 1:\nParis is the capital and most populous city of France. "
    "The city is located on the river Seine, in the north of the country. "
    "The Eiffel Tower, completed in 1889, is one of its best known landmarks."
)

FILLER = (
    "The regional council met on several occasions during the year to review "
    "routine administrative matters, budget lines and maintenance schedules. "
)
EVIDENCE = "The new bridge over the river Aldan was opened to traffic on 14 March 2021."


def build_long_context(filler_repeats: int = 60, with_evidence: bool = True) -> str:
    """A context far beyond 512 tokens whose only relevant fact is at the end."""
    return "passage 1:\n" + FILLER * filler_repeats + (EVIDENCE if with_evidence else "")


# (name, context, response, expectation)
CASES: list[tuple[str, str, str, str]] = [
    (
        "supported",
        PARIS,
        "Paris is the capital of France. Paris is located on the river Seine.",
        "low",
    ),
    (
        "fabricated",
        PARIS,
        "Paris is the capital of Spain. The Eiffel Tower was completed in 1950.",
        "high",
    ),
    (
        "one bad sentence",
        PARIS,
        "Paris is the capital of France. The Eiffel Tower was completed in 1950.",
        "high",
    ),
    (
        "evidence at the end of a long context",
        build_long_context(),
        "The bridge over the river Aldan opened on 14 March 2021.",
        "low",
    ),
    ("empty response", PARIS, "   ", "one"),
    # Informative, no expectation: a sentence that depends on a pronoun whose
    # antecedent is in another sentence. Sentence-level NLI cannot see it.
    ("unresolved pronoun (informative)", PARIS, "It is located on the river Seine.", "any"),
]


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if line.strip():
                yield json.loads(line)


def meets(expectation: str, score: float) -> bool:
    if expectation == "any":
        return True
    if expectation == "low":
        return score < LOW_BOUND
    if expectation == "high":
        return score > HIGH_BOUND
    return score == 1.0


def print_environment(detector: NLIChunkedDetector) -> None:
    import torch
    import transformers

    detector._ensure_loaded()
    print("--- environment ---")
    print(f"torch {torch.__version__} | transformers {transformers.__version__}")
    print(f"device: {detector._device_name}", end="")
    if detector._device_name == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})", end="")
    print(f" | model dtype: {next(detector._model.parameters()).dtype}")
    print(f"model: {detector.model_name} | window: {detector.max_window_tokens} tokens")
    print(f"label map: {detector._model.config.id2label}")
    print(f"entailment index: {detector._entailment_label_index}")


def check_windows(detector: NLIChunkedDetector) -> bool:
    """Windows of every case must respect the limit, and every pair must fit."""
    print("\n--- windows (real tokenizer) ---")
    ok = True
    for name, context, response, _ in CASES:
        windows = window_text(context, detector._count_tokens, detector.max_window_tokens)
        window_sizes = detector._count_tokens(windows) if windows else [0]
        sentences = split_sentences(response)
        longest_sentence = max(detector._count_tokens(sentences), default=0)
        within = max(window_sizes) <= detector.max_window_tokens
        fits = max(window_sizes) + longest_sentence + 3 <= detector.max_sequence_length
        ok = ok and within and fits
        print(
            f"{name:<40} windows={len(windows):<3} max window={max(window_sizes):<4} "
            f"sentences={len(sentences):<2} longest sentence={longest_sentence:<4} "
            f"{'OK' if within and fits else 'CHECK'}"
        )
    return ok


def run_cases(detector: NLIChunkedDetector) -> bool:
    print("\n--- hand-written cases ---")
    ok = True
    for name, context, response, expectation in CASES:
        _, scores = detector.predict_batch([context], [response])
        verdict = meets(expectation, scores[0])
        ok = ok and verdict
        status = "info" if expectation == "any" else ("OK" if verdict else "CHECK")
        print(f"\n{name}: score={scores[0]:.3f} (expected {expectation}) {status}")
        for sentence, support in detector.explain(context, response):
            print(f"    support={support:.3f}  {sentence}")
    return ok


def compare_with_plain_nli(detector: NLIChunkedDetector) -> None:
    """The long-context case with the plain detector, which cuts the context
    at 512 tokens (the evidence sits at the end), next to the same context
    with NO evidence at all. A detector that reads the evidence must score
    the first low and the second high.
    """
    name, context, response, _ = CASES[3]
    contexts = {
        "evidence at the end": context,
        "no evidence at all": build_long_context(with_evidence=False),
    }
    plain = NLIDetector(
        model_name=detector.model_name,
        max_sequence_length=detector.max_sequence_length,
        device=detector._device_name,
    )
    print(f"\n--- '{name}': plain nli vs nli_chunked ---")
    for label, text in contexts.items():
        _, plain_scores = plain.predict_batch([text], [response])
        _, chunked_scores = detector.predict_batch([text], [response])
        print(
            f"{label:<22} plain nli={plain_scores[0]:.3f} | nli_chunked={chunked_scores[0]:.3f}"
        )
    print("(informative only: the plain detector cuts the context at 512 tokens)")


def run_train_sample(detector: NLIChunkedDetector, data_dir: Path, n_samples: int) -> None:
    passages = {
        record["source_id"]: record["source_info"]["passages"].strip()
        for record in read_jsonl(data_dir / "source_info.jsonl")
        if record.get("task_type") == "QA"
    }
    rows = [
        record
        for record in read_jsonl(data_dir / "response.jsonl")
        if record["source_id"] in passages and record["split"] == "train"
    ]
    random.Random(SEED).shuffle(rows)
    rows = rows[:n_samples]

    contexts = [passages[row["source_id"]] for row in rows]
    responses = [row["response"] for row in rows]
    labels = np.asarray([1 if row["labels"] else 0 for row in rows])

    window_counts = [
        len(window_text(context, detector._count_tokens, detector.max_window_tokens))
        for context in contexts
    ]
    sentence_counts = [len(split_sentences(response)) for response in responses]

    start = time.perf_counter()
    _, scores_list = detector.predict_batch(contexts, responses)
    elapsed = time.perf_counter() - start
    scores = np.asarray(scores_list)

    print(f"\n--- official RAGTRUTH TRAIN sample (n={len(rows)}, seed={SEED}) ---")
    print(f"windows per context: median={int(np.median(window_counts))} max={max(window_counts)}")
    print(
        f"sentences per response: median={int(np.median(sentence_counts))} "
        f"max={max(sentence_counts)}"
    )
    print(f"empty responses or contexts (score=1.0 by rule): {int((scores == 1.0).sum())}")
    print(f"score range: min={scores.min():.3f} max={scores.max():.3f} std={scores.std():.3f}")
    print(f"scores above 0.99: {int((scores > 0.99).sum())}/{len(scores)}")
    if scores.std() > 0 and len(set(sentence_counts)) > 1:
        rho = spearmanr(scores, sentence_counts).statistic
        print(f"Spearman(score, number of sentences): {rho:.2f}")
    for label, name in ((0, "correct"), (1, "hallucinated")):
        mask = labels == label
        if mask.any():
            print(f"mean score, {name} (n={int(mask.sum())}): {scores[mask].mean():.3f}")
    print(f"batch time: {elapsed:.1f}s ({1000 * elapsed / len(rows):.0f} ms per response, batched)")
    print("(no AUC on purpose: this sample must not guide any decision)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="First contact of nli_chunked with the real model."
    )
    parser.add_argument("--ragtruth-dir", type=Path, default=Path("data/raw/ragtruth"))
    parser.add_argument("--n-train", type=int, default=40, help="train responses to score; 0 skips")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    detector = NLIChunkedDetector(batch_size=args.batch_size)
    print_environment(detector)

    windows_ok = check_windows(detector)
    cases_ok = run_cases(detector)
    compare_with_plain_nli(detector)
    if args.n_train > 0:
        run_train_sample(detector, args.ragtruth_dir, args.n_train)

    print(f"\nwindows: {'OK' if windows_ok else 'CHECK'} | cases: {'OK' if cases_ok else 'CHECK'}")
    sys.exit(0 if windows_ok and cases_ok else 1)


if __name__ == "__main__":
    main()