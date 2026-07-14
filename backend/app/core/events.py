"""In-process pub/sub for server-sent events, keyed by session id.

A single-instance substitute for Redis pub/sub: each session owns an
``asyncio.Queue`` that the research workflow publishes to and the SSE endpoint
drains. The trade-off (no cross-instance fan-out) is documented in the README.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

_session_queues: dict[str, asyncio.Queue] = {}


def create_queue(session_id: str) -> asyncio.Queue:
    """Create (or replace) the event queue for a session and return it."""
    queue: asyncio.Queue = asyncio.Queue()
    _session_queues[session_id] = queue
    return queue


def get_queue(session_id: str) -> asyncio.Queue | None:
    """Return the queue for a session, or ``None`` if none is registered."""
    return _session_queues.get(session_id)


def remove_queue(session_id: str) -> None:
    """Drop a session's queue once its stream has completed."""
    _session_queues.pop(session_id, None)


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


async def publish(session_id: str, event: str, data: dict) -> None:
    """Publish an ``{event, data}`` frame to a session's queue if present."""
    queue = _session_queues.get(session_id)
    if queue is not None:
        await queue.put({"event": event, "data": data})


async def agent_start(session_id: str, agent: str) -> None:
    """Emit an ``agent_start`` lifecycle event."""
    await publish(session_id, "agent_start", {"agent": agent, "timestamp": _now()})


async def agent_complete(
    session_id: str, agent: str, tokens: int, latency_ms: int, cost_usd: float
) -> None:
    """Emit an ``agent_complete`` event with usage and latency."""
    await publish(
        session_id,
        "agent_complete",
        {
            "agent": agent,
            "tokens": tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
        },
    )


async def agent_error(session_id: str, agent: str, message: str) -> None:
    """Emit an ``agent_error`` event for a non-fatal agent failure."""
    await publish(session_id, "agent_error", {"agent": agent, "message": message})


async def papers_found(
    session_id: str, source: str, count: int, total_so_far: int
) -> None:
    """Emit a discovery-progress ``papers_found`` event."""
    await publish(
        session_id,
        "papers_found",
        {"source": source, "count": count, "total_so_far": total_so_far},
    )


async def paper_analyzed(session_id: str, title: str, index: int, total: int) -> None:
    """Emit an analysis-progress ``paper_analyzed`` event."""
    await publish(
        session_id,
        "paper_analyzed",
        {"title": title, "index": index, "total": total},
    )


async def citation_verification(session_id: str, result: dict) -> None:
    """Emit a ``citation_verification`` event with grounding results."""
    await publish(
        session_id,
        "citation_verification",
        {
            "total": result.get("total_citations", 0),
            "verified": result.get("verified_citations", 0),
            "hallucination_rate": result.get("hallucination_rate", 0.0),
            "unsupported": result.get("unsupported_references", []),
        },
    )


async def done(session_id: str, paper_count: int) -> None:
    """Emit the terminal ``done`` event for a successful session."""
    await publish(session_id, "done", {"session_id": session_id, "paper_count": paper_count})


async def failed(session_id: str, message: str) -> None:
    """Emit the terminal ``failed`` event for an aborted session."""
    await publish(session_id, "failed", {"message": message})
