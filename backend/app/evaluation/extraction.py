"""Pure extraction-quality metrics: token-level F1 over labeled fields."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split a string into alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def _stringify(value: Any) -> str:
    """Render a field value (string, list or ``None``) as a flat string."""
    if value is None:
        return ""
    if isinstance(value, list | tuple | set):
        return " ".join(str(item) for item in value)
    return str(value)


def token_f1(predicted: str, ground_truth: str) -> float:
    """Token-level F1 between two strings (SQuAD-style overlap).

    Two empty strings score 1.0; a single empty side scores 0.0.
    """
    pred_tokens = _tokenize(predicted)
    gt_tokens = _tokenize(ground_truth)
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def extraction_accuracy(
    predictions: list[dict],
    ground_truth: list[dict],
    fields: list[str],
) -> dict[str, float]:
    """Per-field mean token-F1 across an index-aligned labeled dataset.

    ``predictions[i]`` is compared against ``ground_truth[i]`` for each field.
    List-valued fields (e.g. datasets, metrics) are flattened to a string
    before scoring. Returns ``{field: mean_f1}``.
    """
    pairs = list(zip(predictions, ground_truth, strict=False))
    scores: dict[str, float] = {}
    for field in fields:
        if not pairs:
            scores[f"{field}_f1"] = 0.0
            continue
        field_scores = [
            token_f1(_stringify(pred.get(field)), _stringify(truth.get(field)))
            for pred, truth in pairs
        ]
        scores[f"{field}_f1"] = round(sum(field_scores) / len(field_scores), 4)
    return scores
