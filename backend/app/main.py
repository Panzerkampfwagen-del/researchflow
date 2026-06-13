"""FastAPI application entry point.

Wires the lifespan (embedding model load), CORS, a consistent error envelope,
the health endpoint and all ``/api`` routers.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import papers, reports, research
from app.core.config import settings
from app.core.llm import embedding_client
from app.db.database import Base, engine

VERSION = "1.0.0"


def _configure_logging() -> None:
    """Configure structlog over the standard logging module."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )


logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model once at startup; tear down on shutdown."""
    _configure_logging()
    from sqlalchemy import text
    from app.db import models  # noqa: F401 - register models on Base.metadata
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
        logger.info("database_ready")
    except Exception as exc:  # noqa: BLE001
        logger.error("database_init_failed", error=str(exc))
    if settings.LOAD_EMBEDDINGS_ON_STARTUP:
        try:
            await asyncio.to_thread(embedding_client.load)
            logger.info("embedding_model_ready", model=settings.EMBEDDING_MODEL)
        except Exception as exc:  # noqa: BLE001 - startup must not hard-crash
            logger.error("embedding_model_load_failed", error=str(exc))
    yield


app = FastAPI(title="ResearchFlow", version=VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Render HTTP errors using the standard ``{error, detail}`` envelope."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail), "detail": str(exc.detail)},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Render request-validation errors with the standard envelope."""
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler returning a 500 with the standard envelope."""
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )


@app.get("/api/health")
async def health() -> dict:
    """Liveness probe used by the deploy pipeline."""
    return {"status": "ok", "version": VERSION}


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    """Expose Prometheus metrics in the text exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(research.router, prefix="/api", tags=["research"])
app.include_router(reports.router, prefix="/api", tags=["reports"])
app.include_router(papers.router, prefix="/api", tags=["papers"])
