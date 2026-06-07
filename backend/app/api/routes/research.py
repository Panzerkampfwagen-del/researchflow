"""Routes for starting research sessions and streaming their progress."""

from __future__ import annotations

import asyncio
import json
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.grounding import grounding_checker
from app.agents.synthesis import narrative_text_of
from app.api.schemas import (
    ResearchCreateResponse,
    ResearchRequest,
    SessionStatusResponse,
)
from app.core import events
from app.core.config import settings
from app.db.database import async_session_factory, get_db
from app.db.repositories.papers import PaperRepository
from app.db.repositories.reports import AnalysisRepository, ReportRepository
from app.db.repositories.sessions import AgentRunRepository, SessionRepository
from app.evaluation.dataset import load_ground_truth
from app.evaluation.extraction import extraction_accuracy
from app.evaluation.retrieval import compute_retrieval_metrics
from app.graph.workflow import run_workflow

logger = structlog.get_logger(__name__)
router = APIRouter()

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}
HEARTBEAT_SECONDS = 15.0

_background_tasks: set[asyncio.Task] = set()


def _parse_uuid(value: str) -> uuid.UUID:
    """Parse a path id into a UUID, raising a 404 on malformed input."""
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


def _frame(payload: dict) -> str:
    """Serialize an event dict into an SSE ``data:`` frame."""
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/research", response_model=ResearchCreateResponse, status_code=202)
async def create_research(
    payload: ResearchRequest, db: AsyncSession = Depends(get_db)
) -> ResearchCreateResponse:
    """Create a session and launch the research workflow in the background."""
    session_row = await SessionRepository(db).create(payload.query)
    await db.commit()
    session_id = str(session_row.id)

    events.create_queue(session_id)
    task = asyncio.create_task(
        run_workflow(session_id, payload.query, payload.year_start, payload.year_end)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return ResearchCreateResponse(session_id=session_id, status="pending")


async def _event_generator(session_id: str, queue: asyncio.Queue, request: Request):
    """Yield SSE frames from a session queue with periodic heartbeats."""
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
            except TimeoutError:
                yield _frame({"event": "ping"})
                continue
            yield _frame(event)
            if event.get("event") in ("done", "failed"):
                break
    finally:
        events.remove_queue(session_id)


@router.get("/research/{session_id}/stream")
async def stream_research(session_id: str, request: Request) -> StreamingResponse:
    """Stream agent lifecycle events for a session over SSE."""
    queue = events.get_queue(session_id)
    if queue is not None:
        return StreamingResponse(
            _event_generator(session_id, queue, request),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    sid = _parse_uuid(session_id)
    async with async_session_factory() as db:
        session_row = await SessionRepository(db).get(sid)
    if session_row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    terminal = "done" if session_row.status == "completed" else "failed"

    async def _replay():
        yield _frame(
            {
                "event": terminal,
                "data": {"session_id": session_id, "status": session_row.status},
            }
        )

    return StreamingResponse(_replay(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.get("/research/{session_id}", response_model=SessionStatusResponse)
async def get_research(
    session_id: str, db: AsyncSession = Depends(get_db)
) -> SessionStatusResponse:
    """Return the status, plan and progress counters for a session."""
    sid = _parse_uuid(session_id)
    session_row = await SessionRepository(db).get(sid)
    if session_row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    links = await PaperRepository(db).list_for_session(sid)
    report = await ReportRepository(db).get(sid)
    return SessionStatusResponse(
        session_id=session_id,
        query=session_row.query,
        status=session_row.status,
        plan=session_row.plan,
        paper_count=len(links),
        report_ready=report is not None,
        created_at=session_row.created_at,
        completed_at=session_row.completed_at,
    )


@router.get("/evals/{session_id}")
async def get_evals(session_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Return retrieval, extraction and cost metrics for a session.

    Retrieval and extraction metrics are scored against the labeled ground-truth
    dataset; cost and latency are read from recorded ``agent_runs`` telemetry.
    """
    sid = _parse_uuid(session_id)
    session_row = await SessionRepository(db).get(sid)
    if session_row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    links = await PaperRepository(db).list_for_session(sid)
    analyses = await AnalysisRepository(db).list_for_session(sid)
    runs = await AgentRunRepository(db).list_for_session(sid)
    report = await ReportRepository(db).get(sid)
    ground_truth = load_ground_truth()

    retrieval = _retrieval_metrics(links, ground_truth)
    extraction = _extraction_metrics(links, analyses, ground_truth)
    cost = _cost_metrics(runs)
    grounding = await _grounding_metrics(links, report)
    return {
        "retrieval": retrieval,
        "extraction": extraction,
        "cost": cost,
        "grounding": grounding,
    }


def _retrieval_metrics(links, ground_truth: list[dict]) -> dict:
    """Compute the reported retrieval cut-offs against ground-truth ids."""
    retrieved = [link.paper.arxiv_id for link in links if link.paper.arxiv_id]
    relevant = {entry["arxiv_id"] for entry in ground_truth if entry.get("arxiv_id")}
    relevance_scores = {arxiv_id: 1 for arxiv_id in relevant}
    metrics = compute_retrieval_metrics(retrieved, relevant, relevance_scores, [5, 10, 20])
    return {
        "precision_at_5": metrics["precision_at_5"],
        "precision_at_10": metrics["precision_at_10"],
        "recall_at_10": metrics["recall_at_10"],
        "ndcg_at_10": metrics["ndcg_at_10"],
    }


def _extraction_metrics(links, analyses, ground_truth: list[dict]) -> dict:
    """Score per-field extraction F1 for papers present in the ground truth."""
    arxiv_by_paper = {str(link.paper.id): link.paper.arxiv_id for link in links}
    analysis_by_arxiv = {}
    for analysis in analyses:
        arxiv_id = arxiv_by_paper.get(str(analysis.paper_id))
        if arxiv_id:
            analysis_by_arxiv[arxiv_id] = analysis

    predictions: list[dict] = []
    truths: list[dict] = []
    for entry in ground_truth:
        analysis = analysis_by_arxiv.get(entry.get("arxiv_id"))
        if analysis is None:
            continue
        predictions.append(
            {
                "methodology": analysis.methodology,
                "datasets": analysis.datasets,
                "metrics": analysis.metrics,
            }
        )
        truths.append(entry.get("ground_truth", {}))

    return extraction_accuracy(predictions, truths, ["methodology", "datasets", "metrics"])


def _cost_metrics(runs) -> dict:
    """Aggregate token, cost and per-agent latency telemetry."""
    total_tokens = sum(run.tokens_used or 0 for run in runs)
    total_cost = round(sum(run.cost_usd or 0.0 for run in runs), 6)
    latency = {run.agent_name: run.latency_ms or 0 for run in runs}
    return {
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "latency_ms": latency,
    }


async def _grounding_metrics(links, report) -> dict | None:
    """Score NLI factuality and citation accuracy for a completed report.

    Reconstructs the LLM narrative from the stored report, maps every citation
    index to its source abstract (joined to the session papers by arXiv id), and
    runs the entailment grounder. ``factuality`` is the share of cited claims
    entailed by some cited source; ``citation_accuracy`` is the share of
    (claim, cited-source) pairs where *that* source entails the claim. Best-effort:
    returns ``None`` (rather than failing the endpoint) when there is no report or
    grounding is disabled/unavailable/malformed.
    """
    if report is None:
        return None
    try:
        abstract_by_arxiv = {
            link.paper.arxiv_id: (link.paper.abstract or link.paper.title)
            for link in links
            if link.paper.arxiv_id
        }
        premises: dict[int, str] = {}
        for cite in report.citations or []:
            arxiv_id = cite.get("arxiv_id")
            index = cite.get("index")
            if isinstance(index, int) and arxiv_id in abstract_by_arxiv:
                premises[index] = abstract_by_arxiv[arxiv_id]
        narrative = narrative_text_of(
            report.executive_summary, report.trends, report.future_directions
        )
        result = await grounding_checker.check_with_premises(narrative, premises)
    except Exception as exc:  # noqa: BLE001 - eval grounding must not 500 the endpoint
        logger.warning("grounding_metrics_failed", error=str(exc))
        return None
    if result is None:
        return None

    threshold = settings.GROUNDING_ENTAILMENT_THRESHOLD
    pairs = [(c, idx) for c in result.claims for idx in c.citation_indices]
    grounded_pairs = sum(
        1 for c, idx in pairs if c.source_entailments.get(idx, 0.0) >= threshold
    )
    return {
        "factuality": round(result.grounded_claims / result.total_claims, 4)
        if result.total_claims
        else 0.0,
        "citation_accuracy": round(grounded_pairs / len(pairs), 4) if pairs else 0.0,
        "grounded_claims": result.grounded_claims,
        "total_claims": result.total_claims,
        "ungrounded_rate": result.ungrounded_rate,
    }
