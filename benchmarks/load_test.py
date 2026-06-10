#!/usr/bin/env python3
"""Concurrent load / latency benchmark for a live ResearchFlow backend.

Fires N research queries concurrently with ``asyncio.gather``, measures
end-to-end latency percentiles (p50/p95/p99), and breaks latency down per agent
stage using the recorded ``agent_runs`` telemetry exposed at ``/api/evals``.
The point is to surface the bottleneck (typically the Groq rate limit or the
sequential analysis loop) under load.

Requires a running backend. Does not fabricate numbers — it measures a live
system.

Usage:
    python benchmarks/load_test.py --api-url http://localhost:8000 --concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

QUERIES = [
    "post-training quantization for large language models",
    "policy gradient methods for deep reinforcement learning",
    "denoising diffusion probabilistic models for image generation",
    "transformer self-attention architecture for sequence modeling",
    "vision transformers for image classification",
    "retrieval-augmented generation for question answering",
    "mixture of experts for efficient large language models",
    "contrastive learning for self-supervised representations",
    "graph neural networks for molecular property prediction",
    "speculative decoding for faster language model inference",
]

POLL_TIMEOUT_SECONDS = 600
POLL_INTERVAL_SECONDS = 3


async def _one_query(client: httpx.AsyncClient, base: str, query: str) -> dict:
    """Run one query end to end and return its latency breakdown."""
    started = time.perf_counter()
    resp = await client.post(
        f"{base}/api/research",
        json={"query": query, "year_start": 2018, "year_end": 2024},
    )
    resp.raise_for_status()
    session_id = resp.json()["session_id"]

    waited = 0
    status = "pending"
    while waited < POLL_TIMEOUT_SECONDS:
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        waited += POLL_INTERVAL_SECONDS
        status = (await client.get(f"{base}/api/research/{session_id}")).json().get("status")
        if status in ("completed", "failed"):
            break
    end_to_end_ms = (time.perf_counter() - started) * 1000

    per_stage: dict[str, int] = {}
    if status == "completed":
        evals = (await client.get(f"{base}/api/evals/{session_id}")).json()
        per_stage = evals.get("cost", {}).get("latency_ms", {})

    return {"query": query, "status": status, "end_to_end_ms": end_to_end_ms, "stages": per_stage}


def _pct(values: list[float], p: float) -> float:
    """Return the p-th percentile (0-100) of values using linear interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


async def main_async(api_url: str, concurrency: int) -> None:
    """Fire ``concurrency`` queries at once and print latency statistics."""
    base = api_url.rstrip("/")
    queries = [QUERIES[i % len(QUERIES)] for i in range(concurrency)]
    print(f"Firing {concurrency} concurrent queries at {base} ...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        wall_start = time.perf_counter()
        results = await asyncio.gather(*(_one_query(client, base, q) for q in queries))
        wall_ms = (time.perf_counter() - wall_start) * 1000

    completed = [r for r in results if r["status"] == "completed"]
    latencies = [r["end_to_end_ms"] for r in completed]

    print(f"\nCompleted {len(completed)}/{concurrency} in {wall_ms / 1000:.1f}s wall-clock")
    if latencies:
        print("End-to-end latency (ms):")
        print(f"  p50 {_pct(latencies, 50):.0f}  p95 {_pct(latencies, 95):.0f}  "
              f"p99 {_pct(latencies, 99):.0f}  max {max(latencies):.0f}")

    stages = ("planner", "discovery", "analysis", "synthesis")
    print("\nPer-stage latency (ms, mean across completed runs):")
    for stage in stages:
        samples = [r["stages"].get(stage, 0) for r in completed if stage in r["stages"]]
        if samples:
            print(f"  {stage:<10} {statistics.fmean(samples):.0f}")

    if completed:
        slowest = max(stages, key=lambda s: statistics.fmean(
            [r["stages"].get(s, 0) for r in completed if s in r["stages"]] or [0]
        ))
        print(f"\nLikely bottleneck stage: {slowest}")


def main() -> None:
    """Parse arguments and run the load test."""
    parser = argparse.ArgumentParser(description="ResearchFlow concurrent load test")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Backend base URL")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent queries")
    args = parser.parse_args()
    asyncio.run(main_async(args.api_url, args.concurrency))


if __name__ == "__main__":
    main()
