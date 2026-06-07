"""Routes for semantic search over stored papers."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import PaperResult
from app.core.llm import embedding_client
from app.db.database import get_db
from app.db.repositories.papers import PaperRepository

router = APIRouter()


@router.get("/papers/search", response_model=list[PaperResult])
async def search_papers(
    q: str = Query(min_length=1),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> list[PaperResult]:
    """Semantic search over stored paper embeddings via pgvector cosine distance."""
    query_vec = (await embedding_client.aembed([q]))[0].tolist()
    papers = await PaperRepository(db).semantic_search(query_vec, limit)
    return [
        PaperResult(
            id=str(paper.id),
            arxiv_id=paper.arxiv_id,
            semantic_scholar_id=paper.semantic_scholar_id,
            title=paper.title,
            authors=paper.authors or [],
            abstract=paper.abstract,
            year=paper.year,
            venue=paper.venue,
            citation_count=paper.citation_count or 0,
            url=paper.url,
        )
        for paper in papers
    ]
