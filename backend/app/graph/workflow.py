"""ResearchFlow workflow: the four agents wired onto our own pipeline runner.

Cross-cutting concerns (SSE lifecycle events, ``agent_runs`` telemetry, timing,
Prometheus metrics) live in the reusable ``agent_stage`` context manager so each
node stays focused on its agent call plus persistence. Node-level retry and
error isolation are provided by :class:`ResearchPipeline`.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.analysis import run_analysis
from app.agents.discovery import run_discovery
from app.agents.planner import run_planner
from app.agents.synthesis import run_synthesis
from app.core import events, metrics
from app.core.llm import LLMResponse
from app.db.database import async_session_factory
from app.db.repositories.papers import PaperRepository
from app.db.repositories.reports import AnalysisRepository, ReportRepository
from app.db.repositories.sessions import AgentRunRepository, SessionRepository
from app.graph.pipeline import ResearchPipeline
from app.graph.state import ResearchState

logger = structlog.get_logger(__name__)

_session_params: dict[str, tuple[int | None, int | None]] = {}


def _ms_since(start: float) -> int:
    """Milliseconds elapsed since a ``time.perf_counter`` reading."""
    return int((time.perf_counter() - start) * 1000)


@dataclass
class _Stage:
    """Per-stage handle exposing the DB session and collecting usage."""

    db: AsyncSession
    tokens: int = 0
    cost: float = 0.0
    model: str = ""

    def record_usage(self, usage: LLMResponse) -> None:
        """Capture token/cost/model usage for telemetry on stage exit."""
        self.tokens = usage.tokens
        self.cost = usage.cost_usd
        self.model = usage.model


@asynccontextmanager
async def agent_stage(session_id: str, name: str):
    """Wrap an agent stage with SSE, telemetry, timing and metrics.

    Emits ``agent_start`` on entry; on success records the ``agent_runs`` row,
    emits ``agent_complete`` and updates Prometheus; on failure records the
    failed run, emits ``agent_error`` and re-raises so the pipeline can retry or
    record the error.
    """
    sid = uuid.UUID(session_id)
    await events.agent_start(session_id, name)
    start = time.perf_counter()
    async with async_session_factory() as db:
        run_repo = AgentRunRepository(db)
        run = await run_repo.start(sid, name)
        stage = _Stage(db=db)
        try:
            yield stage
        except Exception as exc:
            await run_repo.fail(run, str(exc), _ms_since(start))
            await db.commit()
            await events.agent_error(session_id, name, str(exc))
            logger.error("agent_failed", agent=name, session_id=session_id, error=str(exc))
            raise
        latency = _ms_since(start)
        await run_repo.complete(run, stage.tokens, latency, stage.cost)
        await db.commit()
        await events.agent_complete(session_id, name, stage.tokens, latency, stage.cost)
        metrics.record_agent(name, latency, stage.tokens, stage.model)


async def planner_node(state: ResearchState) -> None:
    """Plan the research and persist the plan onto the session."""
    session_id = state["session_id"]
    sid = uuid.UUID(session_id)
    year_start, year_end = _session_params.get(session_id, (None, None))
    async with agent_stage(session_id, "planner") as stage:
        plan, usage = await run_planner(state["query"], year_start, year_end)
        stage.record_usage(usage)
        await SessionRepository(stage.db).set_plan(sid, plan)
        state["plan"] = plan


async def discovery_node(state: ResearchState) -> None:
    """Discover and rank papers, persist them, and link them to the session."""
    session_id = state["session_id"]
    sid = uuid.UUID(session_id)
    plan = state.get("plan")
    if plan is None:
        return
    async with agent_stage(session_id, "discovery") as stage:
        papers, embeddings = await run_discovery(plan, state["query"], session_id)
        # Expose discovered papers to downstream stages before persistence, so a
        # mid-loop DB failure doesn't hide them (the loop enriches paper_id in
        # place on the same objects).
        state["papers"] = papers
        paper_repo = PaperRepository(stage.db)
        for rank, (paper, embedding) in enumerate(
            zip(papers, embeddings, strict=False), start=1
        ):
            row = await paper_repo.upsert(paper, embedding)
            paper.paper_id = str(row.id)
            await paper_repo.link_to_session(sid, row.id, paper.relevance_score, rank)


async def analysis_node(state: ResearchState) -> None:
    """Analyze each discovered paper and persist the structured extractions."""
    session_id = state["session_id"]
    sid = uuid.UUID(session_id)
    papers = state.get("papers", [])
    if not papers:
        state["analyses"] = []
        return
    async with agent_stage(session_id, "analysis") as stage:
        analyses, usage = await run_analysis(papers, session_id)
        stage.record_usage(usage)
        analysis_repo = AnalysisRepository(stage.db)
        for analysis in analyses:
            if analysis.paper_id:
                await analysis_repo.upsert(sid, analysis)
        state["analyses"] = analyses


async def synthesis_node(state: ResearchState) -> None:
    """Synthesize the report, persist it, and mark the session completed."""
    session_id = state["session_id"]
    sid = uuid.UUID(session_id)
    plan = state.get("plan")
    if plan is None:
        return
    async with agent_stage(session_id, "synthesis") as stage:
        report, usage, verification = await run_synthesis(
            state.get("analyses", []), state.get("papers", []), plan, session_id
        )
        stage.record_usage(usage)
        await ReportRepository(stage.db).upsert(sid, report)
        grounding = verification.grounding
        metrics.record_verification(
            verification.verified_citations,
            verification.hallucination_rate,
            grounding["ungrounded_rate"] if grounding else None,
        )
        state["report"] = report
        # Mark completed last — after the report is persisted and telemetry is
        # recorded — so a late failure cannot commit a "completed" status (the
        # agent_stage error path commits the same session) for a failed run.
        await SessionRepository(stage.db).set_status(sid, "completed", completed=True)


def build_pipeline() -> ResearchPipeline:
    """Construct the four-stage research pipeline.

    Planner and synthesis carry one retry (cheap, idempotent stages); discovery
    and analysis do not retry since their LLM/network sub-calls already tolerate
    partial failures and re-running them is wasteful.
    """
    return (
        ResearchPipeline()
        .add("planner", planner_node, retries=1)
        .add("discovery", discovery_node)
        .add("analysis", analysis_node)
        .add("synthesis", synthesis_node, retries=1)
    )


research_pipeline = build_pipeline()


async def run_workflow(
    session_id: str,
    query: str,
    year_start: int | None = None,
    year_end: int | None = None,
) -> None:
    """Execute the full research pipeline for a session as a background task.

    Marks the session ``running``, runs the pipeline, then emits the terminal
    ``done``/``failed`` SSE event based on whether a report was produced.
    """
    _session_params[session_id] = (year_start, year_end)
    metrics.RUNS_TOTAL.inc()
    sid = uuid.UUID(session_id)
    async with async_session_factory() as db:
        await SessionRepository(db).set_status(sid, "running")
        await db.commit()

    state: ResearchState = {
        "session_id": session_id,
        "query": query,
        "plan": None,
        "papers": [],
        "analyses": [],
        "report": None,
        "agent_events": [],
        "errors": [],
    }

    try:
        await research_pipeline.run(state)
    except Exception as exc:  # noqa: BLE001 - top-level safety net
        logger.error("workflow_crashed", session_id=session_id, error=str(exc))
        async with async_session_factory() as db:
            await SessionRepository(db).set_status(sid, "failed", completed=True)
            await db.commit()
        await events.failed(session_id, str(exc))
        _session_params.pop(session_id, None)
        return

    _session_params.pop(session_id, None)
    if state.get("report") is not None:
        await events.done(session_id, len(state.get("papers", [])))
    else:
        async with async_session_factory() as db:
            await SessionRepository(db).set_status(sid, "failed", completed=True)
            await db.commit()
        errors = state.get("errors", [])
        await events.failed(session_id, "; ".join(errors) or "Workflow produced no report")
