"""Reciprocal Rank Fusion (RRF) for combining heterogeneous rankers.

The dense (cosine), lexical (BM25) and citation signals live on different,
incomparable scales, so a linear ``0.5/0.3/0.2`` blend silently lets whichever
signal happens to have the widest numeric range dominate. RRF sidesteps that by
fusing *ranks* instead of *scores*: a document's contribution from each ranker
is ``1 / (k + rank)``, which is bounded, monotonic, and scale-free. ``k``
(default 60, from Cormack et al. 2009) damps the influence of low ranks.

Everything here is pure-Python and deterministic so it can be exhaustively unit
tested without numpy, a model, or any I/O.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = 60, weights: list[float] | None = None
) -> dict[int, float]:
    """Fuse several ranked id-lists into one score per id.

    Each entry of ``rankings`` is a list of item ids ordered best-first (rank 0
    is the top result). An item's fused score is the ``weights``-weighted sum of
    ``1 / (k + rank)`` across every ranking in which it appears; items missing
    from a ranking simply contribute nothing from that ranker. ``weights``
    defaults to all-ones (plain RRF) and must match ``rankings`` in length.

    Returns a ``{id: fused_score}`` map; callers sort it to obtain the final
    order. Higher is better.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings in length")

    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item_id in enumerate(ranking):
            fused[item_id] = fused.get(item_id, 0.0) + weight / (k + rank)
    return fused


def ranking_from_scores(scores: list[float]) -> list[int]:
    """Return indices ordered by descending score (best first).

    Ties break by original index so the fusion is deterministic regardless of
    the underlying sort's stability guarantees.
    """
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))
