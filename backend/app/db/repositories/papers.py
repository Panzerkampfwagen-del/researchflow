"""Repository for papers, session-paper links and vector search."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Paper, SessionPaper
from app.graph.state import PaperMetadata


class PaperRepository:
    """All read/write access to the ``papers`` and ``session_papers`` tables."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert(
        self, meta: PaperMetadata, embedding: list[float] | None
    ) -> Paper:
        """Insert or update a paper keyed on arXiv / Semantic Scholar id.

        Existing rows are matched on whichever external id is present and have
        their metadata and embedding refreshed; otherwise a new row is created.
        Returns the persisted ``Paper`` with its database id populated.
        """
        existing = await self._find_existing(meta.arxiv_id, meta.semantic_scholar_id)
        if existing is not None:
            existing.title = meta.title
            existing.authors = meta.authors
            existing.abstract = meta.abstract
            existing.year = meta.year or existing.year
            existing.venue = meta.venue or existing.venue
            existing.citation_count = max(meta.citation_count, existing.citation_count or 0)
            existing.url = meta.url or existing.url
            if embedding is not None:
                existing.embedding = embedding
            if meta.arxiv_id and not existing.arxiv_id:
                existing.arxiv_id = meta.arxiv_id
            if meta.semantic_scholar_id and not existing.semantic_scholar_id:
                existing.semantic_scholar_id = meta.semantic_scholar_id
            await self.session.flush()
            return existing

        paper = Paper(
            arxiv_id=meta.arxiv_id,
            semantic_scholar_id=meta.semantic_scholar_id,
            title=meta.title,
            authors=meta.authors,
            abstract=meta.abstract,
            year=meta.year or None,
            venue=meta.venue,
            citation_count=meta.citation_count,
            url=meta.url,
            embedding=embedding,
        )
        self.session.add(paper)
        await self.session.flush()
        return paper

    async def _find_existing(
        self, arxiv_id: str | None, semantic_scholar_id: str | None
    ) -> Paper | None:
        """Return an existing paper matching either external id, if any."""
        if arxiv_id:
            result = await self.session.execute(
                select(Paper).where(Paper.arxiv_id == arxiv_id)
            )
            found = result.scalar_one_or_none()
            if found is not None:
                return found
        if semantic_scholar_id:
            result = await self.session.execute(
                select(Paper).where(Paper.semantic_scholar_id == semantic_scholar_id)
            )
            return result.scalar_one_or_none()
        return None

    async def link_to_session(
        self,
        session_id: uuid.UUID,
        paper_id: uuid.UUID,
        relevance_score: float,
        rank: int,
    ) -> None:
        """Create or update the ``session_papers`` row for a ranked paper."""
        existing = await self.session.get(SessionPaper, (session_id, paper_id))
        if existing is not None:
            existing.relevance_score = relevance_score
            existing.rank = rank
            return
        self.session.add(
            SessionPaper(
                session_id=session_id,
                paper_id=paper_id,
                relevance_score=relevance_score,
                rank=rank,
            )
        )
        await self.session.flush()

    async def list_for_session(self, session_id: uuid.UUID) -> list[SessionPaper]:
        """Return session-paper links (with eager-loaded papers) ordered by rank."""
        result = await self.session.execute(
            select(SessionPaper)
            .where(SessionPaper.session_id == session_id)
            .order_by(SessionPaper.rank.asc().nulls_last())
        )
        return list(result.scalars().all())

    async def semantic_search(
        self, embedding: list[float], limit: int = 20
    ) -> list[Paper]:
        """Return papers ranked by cosine similarity to ``embedding``."""
        result = await self.session.execute(
            select(Paper)
            .where(Paper.embedding.is_not(None))
            .order_by(Paper.embedding.cosine_distance(embedding))
            .limit(limit)
        )
        return list(result.scalars().all())
