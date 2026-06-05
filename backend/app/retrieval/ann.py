"""Backend-agnostic dense ranking seam used by the Discovery stage.

Discovery needs the *order* of candidates by semantic similarity to feed rank
fusion. This module hides whether that order came from the hand-built HNSW index
or from exact brute force, and picks between them on pool size: HNSW only earns
its graph-construction overhead once the candidate pool is large, so small pools
(the common arXiv + Semantic Scholar case) stay exact and the result is honest.
"""

from __future__ import annotations

import numpy as np
import structlog

from app.retrieval.hnsw import HNSWIndex, brute_force_search

logger = structlog.get_logger(__name__)


def dense_ann_ranking(
    matrix: np.ndarray,
    query: np.ndarray,
    top_n: int,
    *,
    backend: str = "hnsw",
    min_candidates_for_hnsw: int = 256,
    m: int = 16,
    ef_construction: int = 200,
    ef_search: int = 64,
) -> list[int]:
    """Return candidate row indices ordered by descending cosine similarity.

    Uses the HNSW index when ``backend == "hnsw"`` and the pool has at least
    ``min_candidates_for_hnsw`` rows; otherwise falls back to exact brute force.
    A query is always answerable: if HNSW construction raises, it degrades to the
    exact path rather than failing the run.
    """
    n = matrix.shape[0]
    if n == 0 or top_n <= 0:
        return []
    k = min(top_n, n)

    if backend == "hnsw" and n >= min_candidates_for_hnsw:
        try:
            index = HNSWIndex.build(
                matrix, m=m, ef_construction=ef_construction, ef_search=ef_search
            )
            ranking = index.search(query, k)
            if ranking:
                return ranking
        except Exception as exc:  # noqa: BLE001 - never fail retrieval over ANN
            logger.warning("hnsw_failed_falling_back", error=str(exc))

    return brute_force_search(matrix, query, k)
