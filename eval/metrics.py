"""Evaluation metrics for explanation and predictive performance."""
from __future__ import annotations

from typing import Iterable, Sequence, Set

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def roc_auc(labels: Sequence[int], preds: Sequence[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, preds))


def pr_auc(labels: Sequence[int], preds: Sequence[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, preds))


def jaccard_distance(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 0.0
    return 1 - len(a & b) / len(a | b)


def explanation_size(edges: Iterable[int], nodes: Iterable[int]) -> tuple[int, int]:
    return len(set(edges)), len(set(nodes))


def robustness_stability(clean_edges: Set[int], corrupted_edges: Set[int]) -> float:
    return jaccard_distance(clean_edges, corrupted_edges)
