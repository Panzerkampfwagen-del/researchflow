"""Planner agent: decompose a query into a structured research plan."""

from __future__ import annotations

from datetime import UTC, datetime

from app.core.llm import LLMResponse, llm_client
from app.graph.state import ResearchPlan

PLANNER_SYSTEM = (
    "You are a research planning expert. Given a research query, decompose it into "
    "3-5 specific subtopics. For each subtopic, generate 2-3 targeted search queries "
    "optimized for academic search engines (concise keyword phrases, not questions). "
    "Infer a reasonable publication year range for the field. "
    "Return valid JSON only, matching the requested schema."
)


async def run_planner(
    query: str,
    year_start: int | None = None,
    year_end: int | None = None,
) -> tuple[ResearchPlan, LLMResponse]:
    """Produce a :class:`ResearchPlan` for ``query`` using the reasoning model.

    When the caller supplies an explicit year range it overrides whatever the
    model infers, so the plan always reflects the user's constraints. Returns
    the plan together with the LLM usage for telemetry.
    """
    current_year = datetime.now(UTC).year
    hint = (
        f"The user requested a year range of {year_start}-{year_end}."
        if year_start and year_end
        else f"No explicit year range was given; the current year is {current_year}."
    )
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {
            "role": "user",
            "content": f"Research query: {query}\n{hint}",
        },
    ]
    plan, usage = await llm_client.structured_complete(
        messages, ResearchPlan, use_reasoning=True
    )
    if year_start is not None:
        plan.year_start = year_start
    if year_end is not None:
        plan.year_end = year_end
    return plan, usage
