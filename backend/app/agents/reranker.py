"""Local cross-encoder reranker for the Discovery stage.

A bi-encoder (the embedding model) scores the query and each abstract
independently; a cross-encoder instead attends over the query and document
*together*, which is markedly more accurate for relevance ranking. We run a
small CPU model (``cross-encoder/ms-marco-MiniLM-L-6-v2``, ~80 MB) locally to
re-order the top candidates — a hosted ML component we own, not an API call.

The model is loaded lazily and reranking degrades gracefully: if the model is
unavailable or scoring fails, the original hybrid order is kept.
"""

from __future__ import annotations

import asyncio

import structlog

from app.core.config import settings
from app.graph.state import PaperMetadata

logger = structlog.get_logger(__name__)

RERANK_WEIGHT = 0.7
CITATION_WEIGHT = 0.3


class CrossEncoderReranker:
    """Lazy wrapper around a sentence-transformers ``CrossEncoder``."""

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        """Load the cross-encoder model once. Safe to call repeatedly."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("loading_cross_encoder", model=settings.RERANK_MODEL)
            self._model = CrossEncoder(settings.RERANK_MODEL)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return cross-encoder relevance scores for each (query, text) pair."""
        self.load()
        pairs = [(query, text) for text in texts]
        return [float(s) for s in self._model.predict(pairs)]

    async def rerank(self, query: str, papers: list[PaperMetadata]) -> bool:
        """Set a normalized ``rerank_score`` on each paper in place.

        Returns ``True`` if reranking ran, ``False`` if it was skipped (no
        papers, disabled, or the model could not score). Scoring runs in a worker
        thread so the event loop is not blocked.
        """
        if not papers:
            return False
        texts = [f"{p.title}. {p.abstract}" for p in papers]
        try:
            scores = await asyncio.to_thread(self.score, query, texts)
        except Exception as exc:  # noqa: BLE001 - degrade to hybrid order
            logger.warning("rerank_failed", error=str(exc))
            return False

        low, high = min(scores), max(scores)
        # All-equal scores (or a single candidate) carry no ordering signal;
        # map them to 1.0 ("equally top") rather than collapsing every paper to
        # 0.0, which would otherwise sink the reranked head below the tail.
        if high <= low:
            for paper in papers:
                paper.rerank_score = 1.0
            return True
        for paper, score in zip(papers, scores, strict=False):
            paper.rerank_score = (score - low) / (high - low)
        return True


reranker = CrossEncoderReranker()
