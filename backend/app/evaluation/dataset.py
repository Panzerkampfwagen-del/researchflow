"""Loader for the labeled ground-truth dataset used by the eval endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

_RELATIVE = Path("evals") / "labeled_data" / "quantization_ground_truth.json"


def _candidate_paths() -> list[Path]:
    """Return plausible locations of the ground-truth file across layouts."""
    here = Path(__file__).resolve()
    roots = [
        here.parents[3],  # repository root (sibling of backend/)
        here.parents[2],  # backend/
        Path.cwd(),
        Path.cwd().parent,
    ]
    return [root / _RELATIVE for root in roots]


def load_ground_truth() -> list[dict]:
    """Load the labeled dataset, returning ``[]`` if it cannot be found.

    Returning an empty list lets the eval endpoint degrade gracefully in minimal
    deployments that do not ship the ``evals/`` directory.
    """
    for path in _candidate_paths():
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
    logger.warning("ground_truth_not_found", searched=[str(p) for p in _candidate_paths()])
    return []
