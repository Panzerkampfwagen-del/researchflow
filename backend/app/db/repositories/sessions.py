"""Repositories for research sessions and per-agent run telemetry."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentRun, ResearchSession
from app.graph.state import ResearchPlan


class SessionRepository:
    """Read/write access to the ``research_sessions`` table."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, query: str) -> ResearchSession:
        """Create a new session in the ``pending`` state and return it."""
        row = ResearchSession(query=query, status="pending")
        self.session.add(row)
        await self.session.flush()
        return row

    async def get(self, session_id: uuid.UUID) -> ResearchSession | None:
        """Fetch a session by id, or ``None`` if it does not exist."""
        return await self.session.get(ResearchSession, session_id)

    async def set_plan(self, session_id: uuid.UUID, plan: ResearchPlan) -> None:
        """Persist the planner's output on the session row."""
        row = await self.session.get(ResearchSession, session_id)
        if row is not None:
            row.plan = plan.model_dump()

    async def set_status(
        self, session_id: uuid.UUID, status: str, completed: bool = False
    ) -> None:
        """Update the session status and optionally stamp ``completed_at``."""
        row = await self.session.get(ResearchSession, session_id)
        if row is not None:
            row.status = status
            if completed:
                row.completed_at = datetime.now(UTC)

    async def fail_stale(self) -> int:
        """Mark every unfinished session as failed; return how many were swept.

        The pipeline runs as an in-memory task, so a session left in ``pending``
        or ``running`` when the process boots is orphaned — its driving task died
        with the previous process and can never resume. Sweeping them on startup
        keeps the UI from hanging on a session that will never progress. Their
        still-``running`` agent runs are closed out too.
        """
        now = datetime.now(UTC)
        result = await self.session.execute(
            update(ResearchSession)
            .where(ResearchSession.status.in_(("pending", "running")))
            .values(status="failed", completed_at=now)
        )
        await self.session.execute(
            update(AgentRun)
            .where(AgentRun.status == "running")
            .values(status="failed", error_message="interrupted by restart", completed_at=now)
        )
        return result.rowcount or 0


class AgentRunRepository:
    """Read/write access to the ``agent_runs`` telemetry table."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def start(self, session_id: uuid.UUID, agent_name: str) -> AgentRun:
        """Record the start of an agent run and return the row."""
        row = AgentRun(session_id=session_id, agent_name=agent_name, status="running")
        self.session.add(row)
        await self.session.flush()
        return row

    async def complete(
        self,
        run: AgentRun,
        tokens: int = 0,
        latency_ms: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Mark an agent run complete with its usage and latency metrics."""
        run.status = "completed"
        run.tokens_used = tokens
        run.latency_ms = latency_ms
        run.cost_usd = cost_usd
        run.completed_at = datetime.now(UTC)

    async def fail(self, run: AgentRun, message: str, latency_ms: int = 0) -> None:
        """Mark an agent run as failed with an error message."""
        run.status = "failed"
        run.error_message = message
        run.latency_ms = latency_ms
        run.completed_at = datetime.now(UTC)

    async def list_for_session(self, session_id: uuid.UUID) -> list[AgentRun]:
        """Return all agent runs for a session ordered by start time."""
        result = await self.session.execute(
            select(AgentRun)
            .where(AgentRun.session_id == session_id)
            .order_by(AgentRun.started_at.asc())
        )
        return list(result.scalars().all())
