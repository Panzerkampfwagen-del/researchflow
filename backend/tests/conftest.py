"""Shared pytest fixtures and test-time environment configuration."""

from __future__ import annotations

import asyncio
import os

os.environ.setdefault("LOAD_EMBEDDINGS_ON_STARTUP", "false")
os.environ.setdefault("DB_USE_NULLPOOL", "true")
os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("ANTHROPIC_API_KEY", "")

import pytest  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db import models  # noqa: E402,F401 - register ORM models on metadata
from app.db.database import Base, engine  # noqa: E402


async def _create_schema() -> None:
    """Create the pgvector extension and a fresh schema for tests."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


@pytest.fixture(scope="session")
def db_ready() -> bool:
    """Attempt to provision the test database, reporting whether it is usable."""
    try:
        asyncio.run(_create_schema())
        return True
    except Exception:  # noqa: BLE001 - DB-less environments skip DB tests
        return False


@pytest.fixture
def requires_db(db_ready: bool) -> None:
    """Skip a test when no database is reachable."""
    if not db_ready:
        pytest.skip("database not available")
