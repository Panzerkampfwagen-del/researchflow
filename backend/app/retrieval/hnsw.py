"""A Hierarchical Navigable Small World (HNSW) index, built from scratch.

This is the approximate-nearest-neighbour structure described by Malkov &
Yashunin (2018), implemented directly on numpy rather than wrapping faiss/hnswlib
so the retrieval pipeline is an artifact we own and can reason about. It indexes
unit-normalised vectors and answers top-``k`` cosine-nearest queries by greedily
descending a multi-layer proximity graph.

Why it exists: dense scoring over a handful of candidates is fine with exact
cosine, but the persistent paper corpus grows without bound, and at corpus scale
exact search is O(N·d) per query. HNSW gives ~log(N) graph hops at a small,
measurable recall cost. ``app/retrieval/__init__`` keeps it dependency-light so
its recall can be unit-tested against the exact brute-force oracle below.

Measured (``benchmarks/hnsw_vs_faiss.py`` on the 2,995-paper corpus): this index
hits ~0.99 recall@10 vs exact — algorithmically on par with Faiss
``IndexHNSWFlat`` — but ~6–7× slower per query (pure Python vs Faiss's C++/SIMD).
It is still sublinear: by 10k vectors it beats exact brute force. So Faiss is the
production ANN; this is the owned, verifiable implementation of the algorithm, and
it is wired into Discovery only above a candidate-pool threshold (see
``ann.py``) where the pool is large enough for ANN to matter. See HNSW_BENCH.md.
"""

from __future__ import annotations

import heapq
import math
import random

import numpy as np


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Return ``vector`` scaled to unit L2 norm (zero vectors pass through)."""
    vec = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def brute_force_search(
    matrix: np.ndarray, query: np.ndarray, k: int
) -> list[int]:
    """Exact cosine top-``k`` over every row of ``matrix`` (the recall oracle)."""
    if matrix.shape[0] == 0 or k <= 0:
        return []
    mat = matrix.astype(np.float32)
    row_norms = np.linalg.norm(mat, axis=1)
    row_norms[row_norms == 0] = 1.0
    q = _normalize(query)
    sims = (mat @ q) / row_norms
    k = min(k, mat.shape[0])
    # argsort ascending then reverse; ties break by index for determinism.
    return sorted(range(mat.shape[0]), key=lambda i: (-float(sims[i]), i))[:k]


class HNSWIndex:
    """An incrementally-built HNSW graph over unit-normalised vectors.

    Parameters mirror the paper: ``M`` neighbours per node per layer (``2M`` at
    the base layer), ``ef_construction`` candidates explored while inserting, and
    ``ef_search`` while querying. Construction is seeded so builds are
    reproducible — essential for the recall test to be stable.
    """

    def __init__(
        self,
        dim: int,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.m = m
        self.m_max0 = 2 * m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._ml = 1.0 / math.log(m) if m > 1 else 1.0
        self._rng = random.Random(seed)
        self._vectors: list[np.ndarray] = []
        self._neighbors: list[list[set[int]]] = []  # node -> layer -> neighbour ids
        self._entry: int | None = None
        self._top = -1

    def __len__(self) -> int:
        return len(self._vectors)

    def _random_level(self) -> int:
        """Sample an insertion level from the exponential level distribution."""
        return int(-math.log(self._rng.random() or 1e-12) * self._ml)

    def _distance(self, query: np.ndarray, node: int) -> float:
        """Cosine distance (``1 - cosine``) between a query and a stored node."""
        return 1.0 - float(np.dot(query, self._vectors[node]))

    def _neighbors_at(self, node: int, layer: int) -> set[int]:
        """Neighbour set of ``node`` at ``layer`` (empty if node is too shallow)."""
        layers = self._neighbors[node]
        return layers[layer] if layer < len(layers) else set()

    def _search_layer(
        self, query: np.ndarray, entry_points: list[int], layer: int, ef: int
    ) -> list[tuple[float, int]]:
        """Greedy best-first search of one layer (Malkov & Yashunin, Alg. 2).

        Returns up to ``ef`` ``(distance, node)`` pairs sorted nearest-first.
        """
        visited: set[int] = set(entry_points)
        candidates: list[tuple[float, int]] = []  # min-heap by distance
        results: list[tuple[float, int]] = []  # max-heap via negated distance
        for ep in entry_points:
            d = self._distance(query, ep)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        while candidates:
            dist_c, current = heapq.heappop(candidates)
            if -results[0][0] < dist_c:
                break  # nearest remaining candidate is worse than our worst kept
            for neighbor in self._neighbors_at(current, layer):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                dist_n = self._distance(query, neighbor)
                if dist_n < -results[0][0] or len(results) < ef:
                    heapq.heappush(candidates, (dist_n, neighbor))
                    heapq.heappush(results, (-dist_n, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)

        return sorted((-neg_d, node) for neg_d, node in results)

    def _prune(self, node: int, layer: int, max_conn: int) -> None:
        """Keep only the ``max_conn`` closest neighbours of ``node`` at ``layer``."""
        current = self._neighbors[node][layer]
        if len(current) <= max_conn:
            return
        ranked = sorted(current, key=lambda nb: self._distance(self._vectors[node], nb))
        self._neighbors[node][layer] = set(ranked[:max_conn])

    def add(self, vector: np.ndarray) -> int:
        """Insert one vector and return its node id."""
        vec = _normalize(vector)
        node = len(self._vectors)
        level = self._random_level()
        self._vectors.append(vec)
        self._neighbors.append([set() for _ in range(level + 1)])

        if self._entry is None:
            self._entry, self._top = node, level
            return node

        entry = self._entry
        # Descend the upper layers greedily (ef=1) to find a good entry point.
        for layer in range(self._top, level, -1):
            entry = self._search_layer(vec, [entry], layer, ef=1)[0][1]

        # Connect at every layer the new node participates in.
        for layer in range(min(level, self._top), -1, -1):
            found = self._search_layer(vec, [entry], layer, self.ef_construction)
            max_conn = self.m_max0 if layer == 0 else self.m
            selected = [node_id for _, node_id in found[:max_conn]]
            for neighbor in selected:
                self._neighbors[node][layer].add(neighbor)
                self._neighbors[neighbor][layer].add(node)
                self._prune(neighbor, layer, max_conn)
            if found:
                entry = found[0][1]

        if level > self._top:
            self._top, self._entry = level, node
        return node

    def search(self, query: np.ndarray, k: int) -> list[int]:
        """Return the ids of the approximate ``k`` nearest neighbours."""
        if self._entry is None or k <= 0:
            return []
        q = _normalize(query)
        entry = self._entry
        for layer in range(self._top, 0, -1):
            entry = self._search_layer(q, [entry], layer, ef=1)[0][1]
        ef = max(self.ef_search, k)
        found = self._search_layer(q, [entry], 0, ef)
        return [node for _, node in found[:k]]

    @classmethod
    def build(cls, matrix: np.ndarray, **kwargs) -> HNSWIndex:
        """Construct an index from a ``(n, dim)`` matrix in row order."""
        index = cls(dim=matrix.shape[1] if matrix.shape[0] else 0, **kwargs)
        for row in matrix:
            index.add(row)
        return index
