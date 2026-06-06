"""A minimal sequential pipeline runner — the orchestration core we own.

This replaces a graph framework with a small class: nodes run in order over a
shared mutable ``state`` dict, each wrapped with per-node retries and error
isolation so one node's failure is recorded in ``state["errors"]`` and the run
continues to the next stage rather than aborting. LLM provider failover
(Groq → Anthropic) is handled per call inside the LLM client; this layer adds
node-level retry, timing, and graceful degradation.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

State = dict[str, Any]
NodeFn = Callable[[State], Awaitable[None]]


@dataclass
class Node:
    """A single pipeline stage: a named async callable over the state."""

    name: str
    fn: NodeFn
    retries: int = 0  # extra attempts after the first


@dataclass
class ResearchPipeline:
    """Run a list of async nodes sequentially over a shared state dict."""

    nodes: list[Node] = field(default_factory=list)

    def add(self, name: str, fn: NodeFn, retries: int = 0) -> ResearchPipeline:
        """Append a node and return self for fluent construction."""
        self.nodes.append(Node(name=name, fn=fn, retries=retries))
        return self

    async def run(self, state: State) -> State:
        """Execute every node in order; never raises — failures land in state."""
        state.setdefault("errors", [])
        for node in self.nodes:
            await self._run_node(node, state)
        return state

    async def _run_node(self, node: Node, state: State) -> None:
        """Run one node with retries; record a final failure in ``state``."""
        attempts = node.retries + 1
        for attempt in range(attempts):
            start = time.perf_counter()
            try:
                await node.fn(state)
                return
            except Exception as exc:  # noqa: BLE001 - isolate node failure
                remaining = attempts - attempt - 1
                logger.warning(
                    "pipeline_node_error",
                    node=node.name,
                    attempt=attempt,
                    remaining=remaining,
                    elapsed_ms=int((time.perf_counter() - start) * 1000),
                    error=str(exc),
                )
                if remaining > 0:
                    continue
                state["errors"].append(f"{node.name}: {exc}")
