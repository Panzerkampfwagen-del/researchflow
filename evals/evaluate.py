#!/usr/bin/env python3
"""Macro-averaged evaluation across a diverse query set.

Runs every query in ``evals/queries.json`` through a live ResearchFlow backend,
then reports macro-averaged Precision@10 / Recall@10 / NDCG@10 against the
hand-curated expected arXiv IDs, an LLM-as-judge score distribution (coherence /
relevance / gap identification), and NLI-based factuality + citation accuracy
(the share of cited claims actually entailed by their sources). Results are
written to ``EVAL.md``.

Honesty note: the expected-id labels are a *curated* set of seminal papers per
domain, not a fabricated "100 hand-labeled" corpus. The factuality and
citation-accuracy figures are computed by the backend's own NLI grounder
(``/api/evals``), so they are measured rather than asserted.

This requires a running backend and a configured ``GROQ_API_KEY`` (for the
judge). It reuses the backend's own metric functions to avoid drift.

Usage:
    python evals/evaluate.py --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent / "backend"
sys.path.insert(0, str(BACKEND))

from app.evaluation.judge import judge_report  # noqa: E402
from app.evaluation.retrieval import compute_retrieval_metrics  # noqa: E402

QUERIES_PATH = HERE / "queries.json"
POLL_TIMEOUT_SECONDS = 600
POLL_INTERVAL_SECONDS = 3


async def _run_one(client: httpx.AsyncClient, api_url: str, entry: dict) -> dict | None:
    """Run a single query end to end and return its metrics, or None on failure."""
    base = api_url.rstrip("/")
    resp = await client.post(
        f"{base}/api/research",
        json={
            "query": entry["query"],
            "year_start": entry.get("year_start", 2020),
            "year_end": entry.get("year_end", 2024),
        },
    )
    resp.raise_for_status()
    session_id = resp.json()["session_id"]

    waited = 0
    status = "pending"
    while waited < POLL_TIMEOUT_SECONDS:
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        waited += POLL_INTERVAL_SECONDS
        status_resp = await client.get(f"{base}/api/research/{session_id}")
        status = status_resp.json().get("status")
        if status in ("completed", "failed"):
            break
    if status != "completed":
        print(f"  [skip] '{entry['query']}' ended with status={status}")
        return None

    report = (await client.get(f"{base}/api/reports/{session_id}")).json()
    retrieved = [c["arxiv_id"] for c in report.get("citations", []) if c.get("arxiv_id")]
    relevant = set(entry["expected_arxiv_ids"])
    relevance_scores = {arxiv_id: 1 for arxiv_id in relevant}
    metrics = compute_retrieval_metrics(retrieved, relevant, relevance_scores, [10])

    judge = None
    try:
        scores, _ = await judge_report(entry["query"], report.get("markdown_content", ""))
        judge = scores.model_dump()
    except Exception as exc:  # noqa: BLE001 - judge is best-effort
        print(f"  [warn] judge failed for '{entry['query']}': {exc}")

    # Factuality / citation accuracy come from the backend's NLI grounder, which
    # has the source abstracts; None if grounding is disabled or unavailable.
    grounding = None
    try:
        evals = (await client.get(f"{base}/api/evals/{session_id}")).json()
        grounding = evals.get("grounding")
    except Exception as exc:  # noqa: BLE001 - grounding is best-effort
        print(f"  [warn] grounding fetch failed for '{entry['query']}': {exc}")

    return {
        "query": entry["query"],
        "domain": entry.get("domain", ""),
        "precision_at_10": metrics["precision_at_10"],
        "recall_at_10": metrics["recall_at_10"],
        "ndcg_at_10": metrics["ndcg_at_10"],
        "judge": judge,
        "grounding": grounding,
    }


def _macro(values: list[float]) -> float:
    """Macro-average (mean) of a list, or 0.0 if empty."""
    return round(statistics.fmean(values), 4) if values else 0.0


def _write_eval_md(results: list[dict], path: Path) -> None:
    """Write a markdown summary of per-query and macro-averaged results."""
    lines = ["# Evaluation Results", "", "## Per-query retrieval", "",
             "| Query | Domain | P@10 | R@10 | NDCG@10 |",
             "|-------|--------|------|------|---------|"]
    for r in results:
        lines.append(
            f"| {r['query']} | {r['domain']} | {r['precision_at_10']:.3f} | "
            f"{r['recall_at_10']:.3f} | {r['ndcg_at_10']:.3f} |"
        )
    lines += [
        "",
        "## Macro-averaged",
        "",
        f"- **Precision@10:** {_macro([r['precision_at_10'] for r in results]):.3f}",
        f"- **Recall@10:** {_macro([r['recall_at_10'] for r in results]):.3f}",
        f"- **NDCG@10:** {_macro([r['ndcg_at_10'] for r in results]):.3f}",
    ]
    grounded = [r["grounding"] for r in results if r.get("grounding")]
    if grounded:
        lines += [
            "",
            "## Grounding (NLI entailment of cited claims)",
            "",
            "| Query | Factuality | Citation accuracy | Grounded / total |",
            "|-------|-----------|-------------------|------------------|",
        ]
        for r in results:
            g = r.get("grounding")
            if g:
                lines.append(
                    f"| {r['query']} | {g['factuality']:.3f} | "
                    f"{g['citation_accuracy']:.3f} | "
                    f"{g['grounded_claims']}/{g['total_claims']} |"
                )
        lines += [
            "",
            f"- **Macro factuality:** {_macro([g['factuality'] for g in grounded]):.3f}",
            "- **Macro citation accuracy:** "
            f"{_macro([g['citation_accuracy'] for g in grounded]):.3f}",
        ]

    judged = [r["judge"] for r in results if r["judge"]]
    if judged:
        lines += [
            "",
            "## LLM-as-judge (mean, 1-5)",
            "",
            f"- **Coherence:** {_macro([j['coherence'] for j in judged]):.2f}",
            f"- **Relevance:** {_macro([j['relevance'] for j in judged]):.2f}",
            f"- **Gap identification:** {_macro([j['gap_identification'] for j in judged]):.2f}",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main_async(api_url: str) -> None:
    """Run all queries, print a summary, and write EVAL.md."""
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    results: list[dict] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for entry in queries:
            print(f"Running: {entry['query']}")
            result = await _run_one(client, api_url, entry)
            if result is not None:
                results.append(result)

    if not results:
        sys.exit("No queries completed successfully.")

    print(f"\nMacro Precision@10: {_macro([r['precision_at_10'] for r in results]):.3f}")
    print(f"Macro Recall@10:    {_macro([r['recall_at_10'] for r in results]):.3f}")
    print(f"Macro NDCG@10:      {_macro([r['ndcg_at_10'] for r in results]):.3f}")

    out = HERE.parent / "EVAL.md"
    _write_eval_md(results, out)
    print(f"\nWrote {out}")


def main() -> None:
    """Parse arguments and run the evaluation."""
    parser = argparse.ArgumentParser(description="ResearchFlow diverse-query evaluation")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Backend base URL")
    args = parser.parse_args()
    asyncio.run(main_async(args.api_url))


if __name__ == "__main__":
    main()
