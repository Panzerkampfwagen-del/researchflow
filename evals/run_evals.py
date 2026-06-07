#!/usr/bin/env python3
"""CLI to fetch and report evaluation metrics for a research session.

Usage:
    python run_evals.py <session_id> [--api-url http://localhost:8000]

Loads the labeled ground-truth dataset for context, fetches the session's
metrics from the running API, prints a formatted table, and writes the raw
results to ``evals/results/<session_id>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
GROUND_TRUTH_PATH = HERE / "labeled_data" / "quantization_ground_truth.json"
RESULTS_DIR = HERE / "results"


def load_ground_truth() -> list[dict]:
    """Load the labeled ground-truth dataset."""
    with GROUND_TRUTH_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def fetch_metrics(api_url: str, session_id: str) -> dict:
    """Fetch computed metrics for a session from the API."""
    url = f"{api_url.rstrip('/')}/api/evals/{session_id}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        sys.exit(f"API returned HTTP {exc.code} for {url}")
    except urllib.error.URLError as exc:
        sys.exit(f"Could not reach API at {url}: {exc.reason}")


def _print_table(title: str, rows: list[tuple[str, str]]) -> None:
    """Print a simple two-column metrics table."""
    print(f"\n{title}")
    print("-" * 44)
    for label, value in rows:
        print(f"  {label:<28} {value:>12}")


def print_report(metrics: dict, ground_truth: list[dict]) -> None:
    """Print a formatted report of all metric groups."""
    print("=" * 44)
    print("ResearchFlow Evaluation Report")
    print("=" * 44)
    print(f"Ground-truth labeled papers: {len(ground_truth)}")

    retrieval = metrics.get("retrieval", {})
    _print_table(
        "Retrieval Quality",
        [(key, f"{value:.4f}") for key, value in retrieval.items()],
    )

    extraction = metrics.get("extraction", {})
    _print_table(
        "Extraction Accuracy (token F1)",
        [(key, f"{value:.4f}") for key, value in extraction.items()],
    )

    cost = metrics.get("cost", {})
    _print_table(
        "Cost",
        [
            ("total_tokens", str(cost.get("total_tokens", 0))),
            ("total_cost_usd", f"{cost.get('total_cost_usd', 0.0):.5f}"),
        ],
    )
    _print_table(
        "Per-Agent Latency (ms)",
        [(agent, str(ms)) for agent, ms in cost.get("latency_ms", {}).items()],
    )


def main() -> None:
    """Parse arguments, fetch metrics, print and persist results."""
    parser = argparse.ArgumentParser(description="Run ResearchFlow evaluation metrics")
    parser.add_argument("session_id", help="Research session UUID")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Backend base URL")
    args = parser.parse_args()

    ground_truth = load_ground_truth()
    metrics = fetch_metrics(args.api_url, args.session_id)
    print_report(metrics, ground_truth)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.session_id}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
