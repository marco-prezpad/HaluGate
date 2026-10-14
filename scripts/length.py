#!/usr/bin/env python3
"""
length_baseline.py

Checks whether response length alone separates hallucinated from correct
responses: AUC of the word count used as a score (longer = more likely
hallucinated), with the median length of each class, on HaluEval (all
samples) and on the RAGTruth QA official train partition. It never reads
the RAGTruth test partition, and it fits nothing.

A length-only AUC far from 0.5 means the dataset carries a shortcut a
trained model can learn without looking at the evidence.

Usage:
    python scripts/length_baseline.py [--data-root data]

Author: Marco Pérez Padilla
Date:   11-10-2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import median

from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import HallucinationSample, get_dataset_loader  # noqa: E402


def describe(name: str, samples: list[HallucinationSample]) -> str:
    labels = [int(s.is_hallucination) for s in samples]
    words = [len(s.response.split()) for s in samples]
    correct = [w for w, label in zip(words, labels) if label == 0]
    hallucinated = [w for w, label in zip(words, labels) if label == 1]
    return (
        f"{name:<28} n={len(samples):<6} positive={sum(labels) / len(labels):5.1%}  "
        f"length-only AUC={roc_auc_score(labels, words):.3f}  "
        f"median words: correct={median(correct):g} hallucinated={median(hallucinated):g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Length-only baseline.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    halueval = get_dataset_loader("halueval", args.data_root).load()
    ragtruth_train = [
        s for s in get_dataset_loader("ragtruth_qa", args.data_root).load()
        if s.official_split == "train"
    ]
    print(describe("HaluEval (all)", halueval))
    print(describe("RAGTruth QA (official train)", ragtruth_train))


if __name__ == "__main__":
    main()