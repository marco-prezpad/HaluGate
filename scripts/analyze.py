#!/usr/bin/env python3
"""
analyze_lexical_ragtruth.py

Exploratory breakdown of the lexical detector on RAGTruth QA: AUC overall,
per generating model and per response-length quartile, plus the mean score
of correct and hallucinated responses. It uses ONLY the official train
partition (the lexical detector fits nothing, so nothing leaks) and never
reads the test partition, so it cannot influence any protocol decision.

Usage:
    python scripts/analyze_lexical_ragtruth.py [--data-dir data/raw/ragtruth] [--n-gram-size 1]

Author: Marco Pérez Padilla
Date:   11-10-2026
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategies.lexical_overlap import LexicalOverlapDetector  # noqa: E402

QUARTILE_EDGES = (25, 50, 75)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if line.strip():
                yield json.loads(line)


def summarize(name: str, labels: np.ndarray, scores: np.ndarray) -> str:
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return f"{name:<28} n={len(labels):<5} positive={positives / len(labels):5.1%}  AUC=n/a (one class)"
    return (
        f"{name:<28} n={len(labels):<5} positive={positives / len(labels):5.1%}  "
        f"AUC={roc_auc_score(labels, scores):.3f}  "
        f"mean score: correct={scores[labels == 0].mean():.3f} hallucinated={scores[labels == 1].mean():.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lexical detector breakdown on RAGTruth train.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ragtruth"))
    parser.add_argument("--n-gram-size", type=int, default=1)
    args = parser.parse_args()

    passages = {
        record["source_id"]: record["source_info"]["passages"].strip()
        for record in read_jsonl(args.data_dir / "source_info.jsonl")
        if record.get("task_type") == "QA"
    }
    rows = [
        record
        for record in read_jsonl(args.data_dir / "response.jsonl")
        if record["source_id"] in passages and record["split"] == "train"
    ]

    detector = LexicalOverlapDetector(n_gram_size=args.n_gram_size)
    _, score_list = detector.predict_batch(
        [passages[r["source_id"]] for r in rows], [r["response"] for r in rows]
    )
    scores = np.asarray(score_list)
    labels = np.asarray([1 if r["labels"] else 0 for r in rows])
    models = np.asarray([r["model"] for r in rows])
    words = np.asarray([len(r["response"].split()) for r in rows])

    print(f"official train only: {len(rows)} responses, n_gram_size={args.n_gram_size}")
    print(summarize("ALL", labels, scores))

    print("\n--- by generating model ---")
    for model_name in sorted(set(models)):
        mask = models == model_name
        print(f"{summarize(model_name, labels[mask], scores[mask])}  median words={int(np.median(words[mask]))}")

    print("\n--- by response length (words), quartiles ---")
    edges = [0.0, *np.percentile(words, QUARTILE_EDGES), np.inf]
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (words >= low) & (words < high) if np.isfinite(high) else words >= low
        if mask.any():
            label = f"{int(words[mask].min())}-{int(words[mask].max())} words"
            print(summarize(label, labels[mask], scores[mask]))


if __name__ == "__main__":
    main()