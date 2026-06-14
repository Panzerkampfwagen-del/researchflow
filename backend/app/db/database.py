"""Async SQLAlchemy engine, session factory and FastAPI dependency."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

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


def prepare_url(url: str) -> tuple[str, dict]:
    """Strip libpq-only query params from URL and return (clean_url, connect_args).

    asyncpg rejects libpq parameters like ``sslmode`` and ``channel_binding`` that
    Neon/managed Postgres append to the DSN. SSL is instead requested via
    connect_args. SSL is enabled whenever the DSN asks for it in any form
    (``sslmode`` other than disable/allow, or a ``channel_binding`` requirement
    which implies TLS) — Neon refuses non-SSL connections, so erring toward SSL
    is correct for managed Postgres.
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    ssl_mode = params.pop("sslmode", [None])[0]
    channel_binding = params.pop("channel_binding", [None])[0]
    # asyncpg understands none of these libpq DSN params; drop them.
    for libpq_only in ("options", "gssencmode", "target_session_attrs"):
        params.pop(libpq_only, None)
    connect_args: dict = {}
    wants_ssl = (ssl_mode is not None and ssl_mode not in ("disable", "allow")) or (
        channel_binding not in (None, "disable")
    )
    if wants_ssl:
        connect_args["ssl"] = True
    new_query = urlencode({k: v[0] for k, v in params.items()})
    clean_url = urlunparse(parsed._replace(query=new_query))
    return clean_url, connect_args


_db_url, _connect_args = prepare_url(settings.DATABASE_URL)

# NullPool (used under tests) avoids reusing asyncpg connections across event
# loops, which otherwise breaks pytest-asyncio's per-test loops.
_engine_kwargs: dict = {"echo": False, "pool_pre_ping": True, "connect_args": _connect_args}
if settings.DB_USE_NULLPOOL:
    _engine_kwargs = {"echo": False, "poolclass": NullPool, "connect_args": _connect_args}

engine: AsyncEngine = create_async_engine(_db_url, **_engine_kwargs)

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
