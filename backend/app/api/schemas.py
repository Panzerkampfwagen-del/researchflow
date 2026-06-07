"""Request and response models for the HTTP API.

Kept separate from the route handlers so the wire contracts live in one place.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, model_validator


def _current_year() -> int:
    """Return the current UTC year."""
    return datetime.now(UTC).year


class ResearchRequest(BaseModel):
    """Payload for starting a research session."""

    query: str = Field(min_length=10, max_length=500)
    year_start: int = 2020
    year_end: int = Field(default_factory=_current_year)

    @model_validator(mode="after")
    def _validate_years(self) -> ResearchRequest:
        """Ensure the year range is ordered and not in the future."""
        current = _current_year()
        if self.year_start > self.year_end:
            raise ValueError("year_start must be <= year_end")
        if self.year_end > current:
            raise ValueError(f"year_end must be <= {current}")
        return self


class ResearchCreateResponse(BaseModel):
    """Response returned immediately after a session is queued."""

    session_id: str
    status: str


class SessionStatusResponse(BaseModel):
    """Status snapshot for a research session."""

    session_id: str
    query: str
    status: str
    plan: dict | None
    paper_count: int
    report_ready: bool
    created_at: datetime | None
    completed_at: datetime | None


class PaperResult(BaseModel):
    """A single paper in a search or session listing."""

    id: str
    arxiv_id: str | None
    semantic_scholar_id: str | None
    title: str
    authors: list[str]
    abstract: str | None
    year: int | None
    venue: str | None
    citation_count: int
    url: str | None
    relevance_score: float | None = None
