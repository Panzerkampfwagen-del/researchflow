"""Unit tests for the pure evaluation metric functions."""

from __future__ import annotations

import math

import pytest

from app.evaluation.extraction import extraction_accuracy, token_f1
from app.evaluation.retrieval import (
    compute_retrieval_metrics,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_partial_relevance(self):
        assert precision_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5

    def test_perfect(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_fewer_retrieved_than_k(self):
        assert precision_at_k(["a"], {"a"}, 5) == 1.0

    def test_no_relevant(self):
        assert precision_at_k(["a", "b"], set(), 2) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_k([], {"a"}, 5) == 0.0

    def test_zero_k(self):
        assert precision_at_k(["a"], {"a"}, 0) == 0.0


class TestRecallAtK:
    def test_partial(self):
        assert recall_at_k(["a", "b", "c"], {"a", "c", "z"}, 2) == pytest.approx(1 / 3)

    def test_all_found(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0

    def test_no_relevant(self):
        assert recall_at_k(["a"], set(), 3) == 0.0

    def test_cutoff_excludes_later_hits(self):
        assert recall_at_k(["x", "y", "a"], {"a"}, 2) == 0.0


class TestNDCGAtK:
    scores = {"a": 3, "b": 2, "c": 1}

    def test_perfect_order_is_one(self):
        assert ndcg_at_k(["a", "b", "c"], self.scores, 3) == pytest.approx(1.0)

    def test_reversed_order_in_unit_interval(self):
        value = ndcg_at_k(["c", "b", "a"], self.scores, 3)
        assert 0.0 < value < 1.0

    def test_manual_value(self):
        # single relevant item at rank 2 -> 1 / log2(3)
        value = ndcg_at_k(["x", "a"], {"a": 1}, 2)
        assert value == pytest.approx((1 / math.log2(3)) / 1.0)

    def test_no_scores(self):
        assert ndcg_at_k(["a", "b"], {}, 5) == 0.0


class TestTokenF1:
    def test_identical(self):
        assert token_f1("hello world", "hello world") == 1.0

    def test_both_empty(self):
        assert token_f1("", "") == 1.0

    def test_one_empty(self):
        assert token_f1("hello", "") == 0.0

    def test_partial_overlap(self):
        assert token_f1("the cat sat", "the dog sat") == pytest.approx(2 / 3)

    def test_case_and_punctuation_insensitive(self):
        assert token_f1("GPTQ, Quantization!", "gptq quantization") == 1.0


class TestExtractionAccuracy:
    def test_string_field_exact(self):
        preds = [{"methodology": "weight quantization"}]
        truth = [{"methodology": "weight quantization"}]
        assert extraction_accuracy(preds, truth, ["methodology"])["methodology_f1"] == 1.0

    def test_list_field(self):
        preds = [{"datasets": ["C4", "WikiText2"]}]
        truth = [{"datasets": ["C4", "WikiText2"]}]
        assert extraction_accuracy(preds, truth, ["datasets"])["datasets_f1"] == 1.0

    def test_empty_dataset(self):
        assert extraction_accuracy([], [], ["metrics"]) == {"metrics_f1": 0.0}


class TestComputeRetrievalMetrics:
    def test_keys_present(self):
        metrics = compute_retrieval_metrics(["a", "b"], {"a"}, {"a": 1}, [5, 10])
        for key in ("precision_at_5", "recall_at_5", "ndcg_at_5", "precision_at_10"):
            assert key in metrics
