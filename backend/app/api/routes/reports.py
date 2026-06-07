"""Routes for fetching synthesized reports and the session knowledge graph."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.repositories.papers import PaperRepository
from app.db.repositories.reports import AnalysisRepository, ReportRepository
from app.graph.state import PaperAnalysis, PaperMetadata
from app.knowledge_graph.graph import build_graph, graph_to_json

router = APIRouter()


def _parse_uuid(value: str) -> uuid.UUID:
    """Parse a path id into a UUID, raising a 404 on malformed input."""
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.get("/reports/{session_id}")
async def get_report(session_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Return the full report for a session plus its raw markdown."""
    sid = _parse_uuid(session_id)
    report = await ReportRepository(db).get(sid)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return {
        "session_id": session_id,
        "executive_summary": report.executive_summary,
        "methodology_comparison": report.methodology_comparison,
        "research_gaps": report.research_gaps,
        "trends": report.trends,
        "future_directions": report.future_directions,
        "citations": report.citations,
        "markdown_content": report.markdown_content,
    }


@router.get("/knowledge-graph/{session_id}")
async def get_knowledge_graph(
    session_id: str, db: AsyncSession = Depends(get_db)
) -> dict:
    """Build and return the knowledge graph for a session's analyses."""
    sid = _parse_uuid(session_id)
    links = await PaperRepository(db).list_for_session(sid)
    analysis_rows = await AnalysisRepository(db).list_for_session(sid)

    papers = [
        PaperMetadata(
            paper_id=str(link.paper.id),
            arxiv_id=link.paper.arxiv_id,
            semantic_scholar_id=link.paper.semantic_scholar_id,
            title=link.paper.title,
            authors=link.paper.authors or [],
            abstract=link.paper.abstract or "",
            year=link.paper.year or 0,
            venue=link.paper.venue,
            citation_count=link.paper.citation_count or 0,
            url=link.paper.url or "",
        )
        for link in links
    ]
    analyses = [
        PaperAnalysis(
            paper_id=str(row.paper_id),
            problem=row.problem or "",
            methodology=row.methodology or "",
            datasets=row.datasets or [],
            metrics=row.metrics or [],
            key_results=row.key_results or "",
            limitations=row.limitations or "",
            confidence=row.confidence or 0.0,
        )
        for row in analysis_rows
    ]

    graph = build_graph(analyses, papers)
    return graph_to_json(graph)
