"""LLM-as-judge scoring of a synthesized report.

Asks the reasoning model to rate a report on three axes (1-5). Used by the
offline evaluation harness to report a coherence/relevance/gap-identification
distribution across many queries, complementing the hard retrieval metrics.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.llm import LLMResponse, llm_client

JUDGE_SYSTEM = (
    "You are a strict scientific reviewer scoring an automatically generated literature "
    "review. Score the report from 1 (poor) to 5 (excellent) on three axes:\n"
    "- coherence: is the writing well-structured and internally consistent?\n"
    "- relevance: do the papers and discussion actually address the query?\n"
    "- gap_identification: are the identified research gaps specific and well-supported?\n"
    "Provide a one-sentence rationale. Return JSON only."
)


class JudgeScores(BaseModel):
    """A judge's 1-5 ratings for one report."""

    coherence: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    gap_identification: int = Field(ge=1, le=5)
    rationale: str = ""


async def judge_report(query: str, markdown_content: str) -> tuple[JudgeScores, LLMResponse]:
    """Score a report's markdown for a query, returning ratings and usage."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": f"Research query: {query}\n\nReport:\n{markdown_content}",
        },
    ]
    return await llm_client.structured_complete(messages, JudgeScores, use_reasoning=True)
