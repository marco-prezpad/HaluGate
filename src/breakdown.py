"""
breakdown.py

AUC computed inside each group of samples (for RAGTruth: inside each
generating model). A pooled AUC can be inflated when groups differ both in
how often they hallucinate and in how their responses look, because a
detector can then earn points just by recognising the group. The AUC
inside each group does not have that problem.

Author: Marco Pérez Padilla
Date:   14-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class GroupAuc:
    """AUC of one group; ``auc`` is None when the group has a single class."""

    group: str
    sample_count: int
    positive_count: int
    auc: float | None


def auc_by_group(
    labels: list[int], scores: list[float], groups: list[str]
) -> list[GroupAuc]:
    """AUC inside each group, sorted by group name.

    Raises:
        ValueError: if the three lists do not have the same length.
    """
    if not len(labels) == len(scores) == len(groups):
        raise ValueError(
            f"labels, scores and groups must have equal length: "
            f"{len(labels)}, {len(scores)}, {len(groups)}"
        )

    results: list[GroupAuc] = []
    for group in sorted(set(groups)):
        indices = [i for i, g in enumerate(groups) if g == group]
        group_labels = [labels[i] for i in indices]
        positives = sum(group_labels)
        auc = None
        if 0 < positives < len(indices):
            auc = float(roc_auc_score(group_labels, [scores[i] for i in indices]))
        results.append(GroupAuc(group, len(indices), positives, auc))
    return results