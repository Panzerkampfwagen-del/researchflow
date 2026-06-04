"""Async SQLAlchemy engine, session factory and FastAPI dependency."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.core.config import settings


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model."""


# NullPool (used under tests) avoids reusing asyncpg connections across event
# loops, which otherwise breaks pytest-asyncio's per-test loops.
_engine_kwargs: dict = {"echo": False, "pool_pre_ping": True}
if settings.DB_USE_NULLPOOL:
    _engine_kwargs = {"echo": False, "poolclass": NullPool}

engine: AsyncEngine = create_async_engine(settings.DATABASE_URL, **_engine_kwargs)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a request-scoped async session, committing on success."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
