"""Repositories for paper analyses and synthesized reports."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import PaperAnalysisRow, Report
from app.graph.state import PaperAnalysis, ResearchReport


class AnalysisRepository:
    """Read/write access to the ``paper_analyses`` table."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert(
        self, session_id: uuid.UUID, analysis: PaperAnalysis
    ) -> PaperAnalysisRow:
        """Insert or update the analysis for one paper within a session."""
        paper_uuid = uuid.UUID(analysis.paper_id)
        result = await self.session.execute(
            select(PaperAnalysisRow).where(
                PaperAnalysisRow.session_id == session_id,
                PaperAnalysisRow.paper_id == paper_uuid,
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = PaperAnalysisRow(session_id=session_id, paper_id=paper_uuid)
            self.session.add(row)
        row.problem = analysis.problem
        row.methodology = analysis.methodology
        row.datasets = analysis.datasets
        row.metrics = analysis.metrics
        row.key_results = analysis.key_results
        row.limitations = analysis.limitations
        row.confidence = analysis.confidence
        await self.session.flush()
        return row

    async def list_for_session(self, session_id: uuid.UUID) -> list[PaperAnalysisRow]:
        """Return all analyses recorded for a session."""
        result = await self.session.execute(
            select(PaperAnalysisRow).where(PaperAnalysisRow.session_id == session_id)
        )
        return list(result.scalars().all())


class ReportRepository:
    """Read/write access to the ``reports`` table (one report per session)."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert(self, session_id: uuid.UUID, report: ResearchReport) -> Report:
        """Insert or replace the synthesized report for a session."""
        result = await self.session.execute(
            select(Report).where(Report.session_id == session_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = Report(session_id=session_id)
            self.session.add(row)
        row.executive_summary = report.executive_summary
        row.methodology_comparison = report.methodology_comparison
        row.research_gaps = [gap.model_dump() for gap in report.research_gaps]
        row.trends = report.trends
        row.future_directions = report.future_directions
        row.citations = report.citations
        row.markdown_content = report.markdown_content
        await self.session.flush()
        return row

    async def get(self, session_id: uuid.UUID) -> Report | None:
        """Fetch the report for a session, or ``None`` if not yet generated."""
        result = await self.session.execute(
            select(Report).where(Report.session_id == session_id)
        )
        return result.scalar_one_or_none()
