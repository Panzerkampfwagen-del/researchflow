"""Analysis agent: extract structured fields from each paper's abstract."""

from __future__ import annotations

import asyncio

import structlog

from app.core import events
from app.core.llm import LLMResponse, llm_client
from app.graph.state import PaperAnalysis, PaperMetadata

logger = structlog.get_logger(__name__)

MIN_ABSTRACT_CHARS = 100
# Groq free tier: 6000 TPM for llama-3.1-8b-instant. Each paper uses ~900 tokens,
# so 12 s between calls keeps us at ~4500 TPM, safely under the limit.
MAX_PAPERS_TO_ANALYZE = 15
_INTER_CALL_SLEEP = 12.0

ANALYSIS_SYSTEM = (
    "You are a meticulous research analyst. Extract the following from the given paper "
    "abstract and return JSON only:\n"
    "- problem: the specific research problem being addressed\n"
    "- methodology: the core technical approach or algorithm\n"
    "- datasets: list of benchmark datasets used\n"
    "- metrics: list of evaluation metrics reported\n"
    "- key_results: the single most important quantitative finding\n"
    "- limitations: stated or implied limitations\n"
    "- confidence: your confidence in extraction accuracy from 0.0 to 1.0"
)


def _insufficient(paper_id: str) -> PaperAnalysis:
    """Return a low-confidence placeholder for an abstract with no content."""
    return PaperAnalysis(
        paper_id=paper_id,
        problem="Insufficient abstract",
        methodology="Insufficient abstract",
        datasets=[],
        metrics=[],
        key_results="Insufficient abstract",
        limitations="Insufficient abstract",
        confidence=0.3,
    )


def _unavailable(paper_id: str) -> PaperAnalysis:
    """Return a placeholder when the LLM extraction failed (provider error)."""
    return PaperAnalysis(
        paper_id=paper_id,
        problem="Analysis unavailable",
        methodology="Analysis unavailable",
        datasets=[],
        metrics=[],
        key_results="Analysis unavailable",
        limitations="Analysis unavailable",
        confidence=0.0,
    )


async def _extract_one(
    messages: list[dict[str, str]],
) -> tuple[PaperAnalysis | None, LLMResponse | None]:
    """Extract one paper's analysis, falling back fast model → reasoning model.

    The fast extraction model (Groq llama-3.1-8b) can exhaust its free-tier daily
    quota independently of the reasoning model, so on failure we retry once with
    the reasoning model (separate quota) before giving up. Returns ``(None, None)``
    if both fail, so the caller can substitute a placeholder instead of aborting.
    """
    for use_reasoning in (False, True):
        try:
            return await llm_client.structured_complete(
                messages, PaperAnalysis, use_reasoning=use_reasoning
            )
        except Exception as exc:  # noqa: BLE001 - try the next model, then degrade
            logger.warning("analysis_attempt_failed", use_reasoning=use_reasoning, error=str(exc))
    return None, None


async def run_analysis(
    papers: list[PaperMetadata], session_id: str
) -> tuple[list[PaperAnalysis], LLMResponse]:
    """Analyze each paper sequentially with the fast extraction model.

    Abstracts shorter than ``MIN_ABSTRACT_CHARS`` are skipped with a
    low-confidence placeholder rather than wasting an LLM call. Emits a
    ``paper_analyzed`` event per paper and returns the analyses plus aggregate
    usage.
    """
    papers = papers[:MAX_PAPERS_TO_ANALYZE]
    analyses: list[PaperAnalysis] = []
    total = len(papers)
    total_tokens = 0
    total_cost = 0.0
    last_model = ""

    for index, paper in enumerate(papers, start=1):
        paper_id = paper.paper_id or ""
        abstract = paper.abstract or ""
        if len(abstract.strip()) < MIN_ABSTRACT_CHARS:
            analyses.append(_insufficient(paper_id))
        else:
            messages = [
                {"role": "system", "content": ANALYSIS_SYSTEM},
                {
                    "role": "user",
                    "content": f"Title: {paper.title}\n\nAbstract: {abstract}",
                },
            ]
            analysis, usage = await _extract_one(messages)
            if analysis is None or usage is None:
                # Both models failed (e.g. provider outage / quota) — record a
                # placeholder and keep going so the report still generates from
                # the papers that succeeded instead of losing the whole run.
                analyses.append(_unavailable(paper_id))
            else:
                analysis.paper_id = paper_id
                total_tokens += usage.tokens
                total_cost += usage.cost_usd
                last_model = usage.model
                analyses.append(analysis)
            # Throttle to stay within Groq free-tier TPM limit
            if index < total:
                await asyncio.sleep(_INTER_CALL_SLEEP)

        await events.paper_analyzed(session_id, paper.title, index, total)

    aggregate = LLMResponse(
        content="", tokens=total_tokens, cost_usd=round(total_cost, 6), model=last_model
    )
    return analyses, aggregate
