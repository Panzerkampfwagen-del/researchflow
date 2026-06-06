"""Knowledge-graph construction over paper analyses using NetworkX.

Builds a directed graph linking papers to the methods, datasets and metrics
they use, and exposes a JSON projection for the frontend force-graph plus two
structural queries used for differentiation and benchmark signals.
"""

from __future__ import annotations

import re

import networkx as nx

from app.graph.state import PaperAnalysis, PaperMetadata

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Lowercase a label into a hyphenated slug usable as a node id."""
    return _SLUG_RE.sub("-", text.lower()).strip("-")


def build_graph(
    analyses: list[PaperAnalysis], papers: list[PaperMetadata]
) -> nx.DiGraph:
    """Build a directed knowledge graph from analyses and paper metadata.

    Nodes are typed ``paper`` / ``method`` / ``dataset`` / ``metric``. Edges run
    paper to method (``uses``), paper to dataset (``evaluated_on``) and paper to
    metric (``measures``). Method/dataset/metric nodes are deduplicated by slug.
    """
    graph = nx.DiGraph()
    paper_by_id = {p.paper_id: p for p in papers if p.paper_id}

    for analysis in analyses:
        paper = paper_by_id.get(analysis.paper_id)
        if paper is None:
            continue
        paper_node = f"paper_{analysis.paper_id}"
        graph.add_node(
            paper_node,
            type="paper",
            label=paper.title,
            year=paper.year,
            url=paper.url,
        )

        if analysis.methodology and analysis.methodology != "Insufficient abstract":
            method_node = f"method_{slugify(analysis.methodology)}"
            graph.add_node(method_node, type="method", label=analysis.methodology)
            graph.add_edge(paper_node, method_node, relation="uses")

        for dataset in analysis.datasets:
            if not dataset.strip():
                continue
            dataset_node = f"dataset_{slugify(dataset)}"
            graph.add_node(dataset_node, type="dataset", label=dataset)
            graph.add_edge(paper_node, dataset_node, relation="evaluated_on")

        for metric in analysis.metrics:
            if not metric.strip():
                continue
            metric_node = f"metric_{slugify(metric)}"
            graph.add_node(metric_node, type="metric", label=metric)
            graph.add_edge(paper_node, metric_node, relation="measures")

    return graph


def graph_to_json(graph: nx.DiGraph) -> dict:
    """Project a NetworkX graph into ``{nodes: [...], edges: [...]}``."""
    nodes = [{"id": node_id, **attrs} for node_id, attrs in graph.nodes(data=True)]
    edges = [
        {"source": source, "target": target, "relation": attrs.get("relation", "")}
        for source, target, attrs in graph.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}


def find_isolated_methods(graph: nx.DiGraph) -> list[str]:
    """Return method labels used by exactly one paper (differentiation signal)."""
    isolated: list[str] = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("type") != "method":
            continue
        if graph.in_degree(node_id) == 1:
            isolated.append(attrs.get("label", node_id))
    return sorted(isolated)


def find_common_datasets(graph: nx.DiGraph, min_papers: int = 3) -> list[str]:
    """Return dataset labels used by at least ``min_papers`` papers (core benchmarks)."""
    common: list[str] = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("type") != "dataset":
            continue
        if graph.in_degree(node_id) >= min_papers:
            common.append(attrs.get("label", node_id))
    return sorted(common)
