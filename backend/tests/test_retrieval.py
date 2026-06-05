"""Tests for the hand-built retrieval primitives: RRF fusion and HNSW.

The HNSW test is the important one: it pins the index's recall against the exact
brute-force oracle, so a regression in the graph construction or search shows up
as a measurable recall drop rather than silently degrading ranking quality.
"""

from __future__ import annotations

import numpy as np

from app.retrieval.ann import dense_ann_ranking
from app.retrieval.fusion import ranking_from_scores, reciprocal_rank_fusion
from app.retrieval.hnsw import HNSWIndex, brute_force_search


class TestReciprocalRankFusion:
    def test_top_in_both_lists_wins(self):
        # Item 2 is rank 0 in both rankings, so it must come out on top.
        fused = reciprocal_rank_fusion([[2, 0, 1], [2, 1, 0]], k=60)
        order = sorted(fused, key=lambda i: fused[i], reverse=True)
        assert order[0] == 2

    def test_present_in_both_beats_top_in_one_only(self):
        # The defining RRF property: A is merely 2nd in both lists, while B and C
        # are each 1st in one list but absent from the other. Consistency wins.
        # A=0, B=1, C=2.
        fused = reciprocal_rank_fusion([[1, 0], [2, 0]], k=60)
        assert fused[0] > fused[1]
        assert fused[0] > fused[2]

    def test_missing_items_contribute_nothing(self):
        # Item 2 only appears in the second ranking.
        fused = reciprocal_rank_fusion([[0, 1], [2, 1, 0]], k=10)
        # Item 1 is in both near the top; it should beat item 2 (one list only).
        assert fused[1] > fused[2]

    def test_weights_bias_the_first_ranker(self):
        base = reciprocal_rank_fusion([[0, 1], [1, 0]], k=60)
        assert base[0] == base[1]  # symmetric → tie
        weighted = reciprocal_rank_fusion([[0, 1], [1, 0]], k=60, weights=[2.0, 1.0])
        assert weighted[0] > weighted[1]  # first ranker (prefers 0) dominates

    def test_ranking_from_scores_breaks_ties_by_index(self):
        assert ranking_from_scores([0.5, 0.9, 0.9, 0.1]) == [1, 2, 0, 3]


class TestBruteForce:
    def test_orders_by_cosine(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
        query = np.array([1.0, 0.0])
        assert brute_force_search(matrix, query, k=2) == [0, 2]

    def test_empty(self):
        assert brute_force_search(np.zeros((0, 4)), np.ones(4), k=5) == []


class TestHNSWRecall:
    def _recall_at_k(self, n: int, dim: int, k: int) -> float:
        rng = np.random.default_rng(0)
        matrix = rng.standard_normal((n, dim)).astype(np.float32)
        index = HNSWIndex.build(matrix, m=16, ef_construction=200, ef_search=128, seed=1)
        hits = 0
        total = 0
        for _ in range(25):
            query = rng.standard_normal(dim).astype(np.float32)
            exact = set(brute_force_search(matrix, query, k))
            approx = set(index.search(query, k))
            hits += len(exact & approx)
            total += len(exact)
        return hits / total

    def test_recall_matches_brute_force(self):
        # A correct HNSW recovers nearly all exact neighbours on random data.
        recall = self._recall_at_k(n=600, dim=32, k=10)
        assert recall >= 0.85, f"recall@10 too low: {recall:.3f}"

    def test_search_empty_index(self):
        assert HNSWIndex(dim=8).search(np.ones(8), k=5) == []

    def test_returns_at_most_k(self):
        rng = np.random.default_rng(2)
        matrix = rng.standard_normal((50, 16)).astype(np.float32)
        index = HNSWIndex.build(matrix, seed=3)
        assert len(index.search(rng.standard_normal(16), k=7)) == 7


class TestDenseAnnRanking:
    def test_small_pool_uses_exact_path(self):
        # Below the HNSW threshold the ranking must equal exact brute force.
        rng = np.random.default_rng(4)
        matrix = rng.standard_normal((20, 8)).astype(np.float32)
        query = rng.standard_normal(8).astype(np.float32)
        ranking = dense_ann_ranking(
            matrix, query, top_n=20, backend="hnsw", min_candidates_for_hnsw=256
        )
        assert ranking == brute_force_search(matrix, query, 20)

    def test_empty_matrix(self):
        assert dense_ann_ranking(np.zeros((0, 8)), np.ones(8), top_n=5) == []
