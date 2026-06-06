"""Analysis agent: extract structured fields from each paper's abstract."""

from __future__ import annotations

from app.core import events
from app.core.llm import LLMResponse, llm_client
from app.graph.state import PaperAnalysis, PaperMetadata

MIN_ABSTRACT_CHARS = 100

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


async def run_analysis(
    papers: list[PaperMetadata], session_id: str
) -> tuple[list[PaperAnalysis], LLMResponse]:
    """Analyze each paper sequentially with the fast extraction model.

    Abstracts shorter than ``MIN_ABSTRACT_CHARS`` are skipped with a
    low-confidence placeholder rather than wasting an LLM call. Emits a
    ``paper_analyzed`` event per paper and returns the analyses plus aggregate
    usage.
    """
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
            analysis, usage = await llm_client.structured_complete(
                messages, PaperAnalysis, use_reasoning=False
            )
            analysis.paper_id = paper_id
            total_tokens += usage.tokens
            total_cost += usage.cost_usd
            last_model = usage.model
            analyses.append(analysis)

        await events.paper_analyzed(session_id, paper.title, index, total)

    aggregate = LLMResponse(
        content="", tokens=total_tokens, cost_usd=round(total_cost, 6), model=last_model
    )
    return analyses, aggregate
