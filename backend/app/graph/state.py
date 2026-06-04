"""Pydantic data contracts and the LangGraph state definition.

These typed contracts are the interfaces between agents. Every value that flows
through the workflow is one of these models, which keeps each agent boundary
explicit and independently testable.
"""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    """Decomposition of a query into subtopics and concrete search queries."""

    goal: str
    subtopics: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    year_start: int
    year_end: int


class PaperMetadata(BaseModel):
    """Normalized metadata for a single discovered paper."""

    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    year: int = 0
    venue: str | None = None
    citation_count: int = 0
    url: str = ""
    relevance_score: float = 0.0
    dense_score: float = 0.0
    lexical_score: float = 0.0
    citation_score: float = 0.0
    rerank_score: float = 0.0
    paper_id: str | None = None


class PaperAnalysis(BaseModel):
    """Structured extraction for one paper.

    ``paper_id`` defaults to empty so the extraction LLM may omit it; the
    Analysis agent stamps the real database id after parsing.
    """

    paper_id: str = ""
    problem: str = ""
    methodology: str = ""
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    key_results: str = ""
    limitations: str = ""
    confidence: float = 0.0


class ResearchGap(BaseModel):
    """A specific, evidence-backed gap in the surveyed literature."""

    description: str
    supporting_evidence: list[str] = Field(default_factory=list)
    opportunity: str = ""


class ResearchReport(BaseModel):
    """The final synthesized report returned to the client."""

    executive_summary: str = ""
    methodology_comparison: list[dict] = Field(default_factory=list)
    research_gaps: list[ResearchGap] = Field(default_factory=list)
    trends: list[str] = Field(default_factory=list)
    future_directions: list[str] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    markdown_content: str = ""


class ResearchState(TypedDict):
    """Mutable state threaded through the LangGraph workflow."""

    session_id: str
    query: str
    plan: ResearchPlan | None
    papers: list[PaperMetadata]
    analyses: list[PaperAnalysis]
    report: ResearchReport | None
    agent_events: list[dict]
    errors: list[str]
