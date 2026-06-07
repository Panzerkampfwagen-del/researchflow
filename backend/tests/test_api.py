"""API tests for the research endpoints (require a reachable database)."""

from __future__ import annotations

import uuid

import httpx
import pytest

from app.api.routes import research as research_routes
from app.main import app

pytestmark = pytest.mark.usefixtures("requires_db")


async def _noop_workflow(*args, **kwargs) -> None:
    """Stand-in for the real workflow so POST tests do not hit the network."""
    return None


def _client() -> httpx.AsyncClient:
    """Build an in-process ASGI test client."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_health():
    async with _client() as client:
        resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": "1.0.0"}


async def test_create_research_returns_shape(monkeypatch):
    monkeypatch.setattr(research_routes, "run_workflow", _noop_workflow)
    async with _client() as client:
        resp = await client.post(
            "/api/research",
            json={"query": "LLM quantization methods", "year_start": 2020, "year_end": 2024},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert set(body) == {"session_id", "status"}
    assert body["status"] == "pending"
    uuid.UUID(body["session_id"])  # well-formed id


async def test_create_research_rejects_short_query(monkeypatch):
    monkeypatch.setattr(research_routes, "run_workflow", _noop_workflow)
    async with _client() as client:
        resp = await client.post("/api/research", json={"query": "short"})
    assert resp.status_code == 422
    assert "error" in resp.json()


async def test_get_report_unknown_returns_404():
    async with _client() as client:
        resp = await client.get(f"/api/reports/{uuid.uuid4()}")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]
    assert "detail" in body
