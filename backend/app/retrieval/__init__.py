"""Hand-built retrieval primitives: rank fusion and an HNSW ANN index.

These are deliberately dependency-light (numpy only) so the retrieval pipeline
is an artifact we own and can unit-test deterministically, rather than a thin
wrapper around a hosted search API.
"""
