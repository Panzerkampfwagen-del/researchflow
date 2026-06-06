"""Citation grounding and hallucination detection for synthesized reports.

The Synthesis agent assembles citations deterministically from retrieved
metadata, so this layer is a guardrail: it confirms every citation maps to a
real retrieved paper (fuzzy title match) and that every bracketed reference the
LLM narrative emits points at an existing citation entry. Any failure surfaces
as a warning marker and a "Verification Notes" section rather than being hidden.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from app.graph.state import PaperMetadata

TITLE_MATCH_THRESHOLD = 75.0
_BRACKET_RE = re.compile(r"\[([0-9][0-9\s,–-]*)\]")


def parse_citation_indices(text: str) -> list[int]:
    """Extract citation indices from bracketed references in ``text``.

    Handles single ``[1]``, grouped ``[1, 2]`` and range ``[1-3]`` (and en-dash
    ``[1–3]``) forms, returning a sorted list of unique indices. Brackets whose
    contents are not purely numeric/comma/range (e.g. ``[CLS]``) never match.
    """
    found: set[int] = set()
    for group in _BRACKET_RE.findall(text or ""):
        for part in group.replace("–", "-").split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = (p.strip() for p in part.split("-", 1))
                if lo.isdigit() and hi.isdigit():
                    found.update(range(int(lo), int(hi) + 1))
            elif part.isdigit():
                found.add(int(part))
    return sorted(found)


class CitationCheck(BaseModel):
    """Verification verdict for a single citation entry."""

    index: int
    title: str
    verified: bool
    score: float


class VerificationResult(BaseModel):
    """Aggregate citation-grounding result attached to a synthesis run."""

    citations: list[CitationCheck] = Field(default_factory=list)
    unsupported_references: list[str] = Field(default_factory=list)
    total_citations: int = 0
    verified_citations: int = 0
    hallucination_rate: float = 0.0
    # NLI entailment grounding (app/agents/grounding.py), serialized to a dict to
    # keep this module model-only and free of the heavyweight NLI import. ``None``
    # when grounding is disabled or unavailable.
    grounding: dict | None = None


def verify_report(
    citations: list[dict], papers: list[PaperMetadata], narrative_text: str
) -> VerificationResult:
    """Verify citations against retrieved papers and check reference integrity.

    Each citation title is fuzzy-matched (token-set ratio) against the titles of
    the papers actually retrieved; a match at or above ``TITLE_MATCH_THRESHOLD``
    is considered grounded. Bracketed references appearing in the LLM narrative
    (including grouped/range forms) are checked against the set of real citation
    indices, and any dangling reference is listed in ``unsupported_references``.
    ``hallucination_rate`` is a clean per-citation rate — the share of citations
    that could not be matched to a retrieved paper — so it is comparable across
    runs regardless of how many bracketed references the narrative emits.
    """
    known_titles = [p.title for p in papers if p.title]
    checks: list[CitationCheck] = []
    for cite in citations:
        title = str(cite.get("title", ""))
        score = (
            max((fuzz.token_set_ratio(title, known) for known in known_titles), default=0.0)
            if known_titles
            else 0.0
        )
        checks.append(
            CitationCheck(
                index=int(cite.get("index", 0)),
                title=title,
                verified=score >= TITLE_MATCH_THRESHOLD,
                score=round(float(score), 1),
            )
        )

    valid_indices = {check.index for check in checks}
    referenced = set(parse_citation_indices(narrative_text))
    unsupported = [
        f"[{n}] cited in the narrative has no matching reference entry"
        for n in sorted(referenced)
        if n not in valid_indices
    ]
    for check in checks:
        if not check.verified:
            unsupported.append(
                f'Citation [{check.index}] "{check.title}" could not be matched '
                f"to a retrieved paper (match score {check.score})"
            )

    total = len(checks)
    verified = sum(1 for check in checks if check.verified)
    # Per-citation rate: share of citation entries not grounded to a retrieved
    # paper. Dangling narrative references are reported separately above rather
    # than folded into this rate, which kept it from mixing two populations.
    hallucination_rate = round((total - verified) / total, 4) if total else 0.0

    return VerificationResult(
        citations=checks,
        unsupported_references=unsupported,
        total_citations=total,
        verified_citations=verified,
        hallucination_rate=hallucination_rate,
    )
