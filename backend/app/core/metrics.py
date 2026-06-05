"""Prometheus metrics exposed at ``/metrics``.

Collectors are module-level singletons registered on the default registry. The
workflow nodes update durations/tokens/run counts; the Synthesis stage updates
the citation-verification gauges.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

RUNS_TOTAL = Counter(
    "researchflow_runs_total",
    "Total number of research workflow runs started.",
)

AGENT_DURATION = Histogram(
    "researchflow_agent_duration_seconds",
    "Wall-clock duration of each agent stage.",
    ["agent"],
)

TOKENS_USED = Counter(
    "researchflow_tokens_used_total",
    "LLM tokens consumed, labeled by model.",
    ["model"],
)

CITATIONS_VERIFIED = Gauge(
    "researchflow_citations_verified",
    "Number of citations grounded against retrieved papers in the latest run.",
)

HALLUCINATION_RATE = Gauge(
    "researchflow_hallucination_rate",
    "Share of ungrounded citations/references in the latest run (0-1).",
)

UNGROUNDED_CLAIM_RATE = Gauge(
    "researchflow_ungrounded_claim_rate",
    "Share of cited claims not entailed by their source (NLI grounding, 0-1).",
)


def record_agent(agent: str, latency_ms: int, tokens: int, model: str) -> None:
    """Record duration and token usage for a completed agent stage."""
    AGENT_DURATION.labels(agent=agent).observe(latency_ms / 1000.0)
    if tokens:
        # Record tokens even when the model label is missing, so the total is not
        # silently undercounted for a stage whose usage carried no model string.
        TOKENS_USED.labels(model=model or "unknown").inc(tokens)


def record_verification(
    verified: int, hallucination_rate: float, ungrounded_claim_rate: float | None = None
) -> None:
    """Update the citation-verification gauges for the latest run.

    ``ungrounded_claim_rate`` is the NLI-grounding miss rate; ``None`` when
    grounding did not run (model unavailable or disabled), in which case the
    gauge is set to NaN rather than left showing a previous run's value.
    """
    CITATIONS_VERIFIED.set(verified)
    HALLUCINATION_RATE.set(hallucination_rate)
    UNGROUNDED_CLAIM_RATE.set(
        ungrounded_claim_rate if ungrounded_claim_rate is not None else float("nan")
    )
