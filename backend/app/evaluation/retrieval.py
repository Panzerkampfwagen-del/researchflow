"""Pure retrieval-quality metrics: Precision@K, Recall@K and NDCG@K.

Every function here is deterministic and dependency-free so it can be unit
tested exhaustively without any database or network access.
"""

from __future__ import annotations

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-``k`` retrieved items that are relevant.

    The denominator is the number of items actually present in the top-``k``
    slice, so retrieving fewer than ``k`` items is not double-penalized.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant items found within the top-``k`` retrieved."""
    if not relevant or k <= 0:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def ndcg_at_k(retrieved: list[str], relevance_scores: dict[str, int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at ``k``.

    ``relevance_scores`` maps an item id to its graded relevance (0 if absent).
    The result is the achieved DCG divided by the ideal DCG and lies in [0, 1].
    """
    if k <= 0:
        return 0.0

    def dcg(scores: list[int]) -> float:
        return sum(score / math.log2(idx + 2) for idx, score in enumerate(scores))

    gains = [relevance_scores.get(item, 0) for item in retrieved[:k]]
    ideal = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0.0:
        return 0.0
    return dcg(gains) / idcg


def compute_retrieval_metrics(
    retrieved: list[str],
    relevant: set[str],
    relevance_scores: dict[str, int],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute P@K, R@K and NDCG@K for every ``k`` in ``k_values``.

    Returns a flat dict with keys like ``precision_at_5`` / ``recall_at_10`` /
    ``ndcg_at_10`` so callers can select whichever cut-offs they report.
    """
    if k_values is None:
        k_values = [5, 10, 20]
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"precision_at_{k}"] = round(precision_at_k(retrieved, relevant, k), 4)
        metrics[f"recall_at_{k}"] = round(recall_at_k(retrieved, relevant, k), 4)
        metrics[f"ndcg_at_{k}"] = round(ndcg_at_k(retrieved, relevance_scores, k), 4)
    return metrics
