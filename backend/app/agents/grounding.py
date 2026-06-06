"""NLI-based grounding: does each cited claim actually follow from its source?

Citation verification (``app/agents/verification.py``) answers a weaker
question — "does citation [N] point at a real retrieved paper?" — by fuzzy title
match. It cannot catch the more dangerous failure where a *real* paper is cited
for a claim it never makes. This module closes that gap with natural-language
inference: for every sentence in the LLM narrative that cites ``[N]``, we run an
entailment model with the cited paper's abstract as the premise and the sentence
as the hypothesis. A claim is "grounded" only if some cited source entails it.

The NLI model (a small cross-encoder) is loaded lazily and runs in a worker
thread; if it is unavailable the check degrades to ``None`` and synthesis simply
skips grounding rather than failing. The pure, model-free assembly of
claim/premise pairs is unit-tested without any download.
"""

from __future__ import annotations

import asyncio
import re

import structlog
from pydantic import BaseModel, Field

from app.agents.verification import parse_citation_indices
from app.core.config import settings
from app.graph.state import PaperMetadata

logger = structlog.get_logger(__name__)

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|\S[^.!?]*$")
# Label order emitted by cross-encoder/nli-* models: contradiction, entailment, neutral.
_ENTAILMENT_LABEL_INDEX = 1


class ClaimGrounding(BaseModel):
    """Grounding verdict for one cited claim sentence."""

    claim: str
    citation_indices: list[int] = Field(default_factory=list)
    entailment: float = 0.0
    grounded: bool = False
    # Per-cited-source entailment {citation_index: entailment_prob}; lets callers
    # judge each citation individually rather than only the claim-level max.
    source_entailments: dict[int, float] = Field(default_factory=dict)


class GroundingResult(BaseModel):
    """Aggregate entailment-grounding result for a synthesized narrative."""

    claims: list[ClaimGrounding] = Field(default_factory=list)
    total_claims: int = 0
    grounded_claims: int = 0
    ungrounded_rate: float = 0.0


def split_sentences(text: str) -> list[str]:
    """Split narrative text into trimmed sentences (lightweight, no nltk)."""
    return [s.strip() for s in _SENTENCE_RE.findall(text or "") if s.strip()]


def extract_claims(narrative_text: str) -> list[tuple[str, list[int]]]:
    """Pair each citing sentence with the citation indices it references.

    Sentences without a bracketed reference are dropped: there is no source to
    entail them against, so they are out of scope for grounding. Grouped/range
    citations (``[1, 2]`` / ``[1-3]``) are parsed via ``parse_citation_indices``.
    """
    claims: list[tuple[str, list[int]]] = []
    for sentence in split_sentences(narrative_text):
        indices = parse_citation_indices(sentence)
        if indices:
            claims.append((sentence, indices))
    return claims


def _premises_by_index(papers: list[PaperMetadata]) -> dict[int, str]:
    """Map a 1-based citation index to its source premise (abstract or title).

    Citations are built from ``papers`` in order (index = position + 1), so the
    nth paper backs citation ``n``.
    """
    premises: dict[int, str] = {}
    for position, paper in enumerate(papers, start=1):
        premises[position] = paper.abstract or paper.title
    return premises


class GroundingChecker:
    """Lazy wrapper around an NLI cross-encoder producing entailment scores."""

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        """Load the NLI cross-encoder once. Safe to call repeatedly."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("loading_nli_model", model=settings.NLI_MODEL)
            self._model = CrossEncoder(settings.NLI_MODEL)

    def score(self, premise_hypothesis_pairs: list[tuple[str, str]]) -> list[float]:
        """Return the entailment probability for each (premise, hypothesis) pair."""
        import numpy as np

        self.load()
        logits = np.asarray(self._model.predict(premise_hypothesis_pairs))
        if logits.ndim == 1:  # single-logit regression head: treat as the score
            return [float(x) for x in logits]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return [float(p) for p in probs[:, _ENTAILMENT_LABEL_INDEX]]

    async def check(
        self, narrative_text: str, papers: list[PaperMetadata]
    ) -> GroundingResult | None:
        """Entailment-check every cited claim against its source abstract.

        Convenience wrapper that derives the index→premise map from ``papers`` in
        citation order, then defers to :meth:`check_with_premises`.
        """
        return await self.check_with_premises(narrative_text, _premises_by_index(papers))

    async def check_with_premises(
        self, narrative_text: str, premises: dict[int, str]
    ) -> GroundingResult | None:
        """Entailment-check cited claims against an explicit index→premise map.

        Returns a :class:`GroundingResult`, or ``None`` if grounding is disabled,
        there are no checkable claims, or the model could not score (in which
        case the caller silently skips grounding).
        """
        if not settings.GROUNDING_ENABLED:
            return None
        claims = extract_claims(narrative_text)
        if not claims or not premises:
            return None

        # One (premise, hypothesis) pair per (claim, cited source with an abstract).
        pairs: list[tuple[str, str]] = []
        owners: list[int] = []  # pair index -> claim index
        pair_cites: list[int] = []  # pair index -> citation index
        for claim_idx, (sentence, indices) in enumerate(claims):
            for cite_idx in indices:
                premise = premises.get(cite_idx)
                if premise:
                    pairs.append((premise, sentence))
                    owners.append(claim_idx)
                    pair_cites.append(cite_idx)
        if not pairs:
            return None

        try:
            scores = await asyncio.to_thread(self.score, pairs)
        except Exception as exc:  # noqa: BLE001 - degrade to "grounding unavailable"
            logger.warning("grounding_failed", error=str(exc))
            return None

        # Keep each source's entailment per claim so callers can judge citations
        # individually; the claim-level grounded flag uses the best source.
        per_source: dict[int, dict[int, float]] = {}
        for claim_idx, cite_idx, score in zip(owners, pair_cites, scores, strict=True):
            bucket = per_source.setdefault(claim_idx, {})
            bucket[cite_idx] = max(bucket.get(cite_idx, 0.0), score)

        threshold = settings.GROUNDING_ENTAILMENT_THRESHOLD
        results: list[ClaimGrounding] = []
        for claim_idx, (sentence, indices) in enumerate(claims):
            sources = {ci: round(v, 4) for ci, v in per_source.get(claim_idx, {}).items()}
            entailment = round(max(sources.values(), default=0.0), 4)
            results.append(
                ClaimGrounding(
                    claim=sentence,
                    citation_indices=indices,
                    entailment=entailment,
                    grounded=entailment >= threshold,
                    source_entailments=sources,
                )
            )

        grounded = sum(1 for c in results if c.grounded)
        total = len(results)
        return GroundingResult(
            claims=results,
            total_claims=total,
            grounded_claims=grounded,
            ungrounded_rate=round((total - grounded) / total, 4) if total else 0.0,
        )


grounding_checker = GroundingChecker()
