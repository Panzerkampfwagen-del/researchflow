"""Discovery agent: search arXiv and Semantic Scholar, dedupe, rank.

Network calls and the embedding model are isolated behind small helpers so the
deterministic pieces (title normalization, deduplication, scoring) can be unit
tested without any I/O.
"""

from __future__ import annotations

import asyncio
import re
import string
import xml.etree.ElementTree as ET

import httpx
import numpy as np
import structlog
from rank_bm25 import BM25Okapi

from app.agents.reranker import CITATION_WEIGHT, RERANK_WEIGHT, reranker
from app.core import events
from app.core.config import settings
from app.core.llm import embedding_client
from app.graph.state import PaperMetadata, ResearchPlan
from app.retrieval.fusion import ranking_from_scores, reciprocal_rank_fusion

logger = structlog.get_logger(__name__)

ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_FIELDS = "title,authors,abstract,year,venue,citationCount,externalIds,url"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_ARXIV_VERSION_RE = re.compile(r"v\d+$")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WORD_RE = re.compile(r"[a-z0-9]+")

# Hybrid ranking weights: dense (semantic) + lexical (BM25) + citation signal.
WEIGHT_DENSE = 0.5
WEIGHT_LEXICAL = 0.3
WEIGHT_CITATION = 0.2


def normalize_title(title: str) -> str:
    """Lowercase a title, strip punctuation and collapse whitespace.

    Used as the deduplication key across the two paper sources.
    """
    lowered = title.lower().translate(_PUNCT_TABLE)
    return " ".join(lowered.split())


def _metadata_richness(paper: PaperMetadata) -> int:
    """Score how much metadata a paper carries, for dedup tie-breaking."""
    score = 0
    if paper.abstract:
        score += len(paper.abstract)
    if paper.authors:
        score += 10 * len(paper.authors)
    if paper.venue:
        score += 20
    if paper.citation_count:
        score += 5
    if paper.arxiv_id:
        score += 5
    if paper.semantic_scholar_id:
        score += 5
    return score


def deduplicate(papers: list[PaperMetadata]) -> list[PaperMetadata]:
    """Collapse papers sharing a normalized title, keeping the richest version.

    When two records describe the same paper, external ids from the discarded
    record are merged onto the kept one so downstream upserts stay linked.
    """
    best: dict[str, PaperMetadata] = {}
    for paper in papers:
        key = normalize_title(paper.title)
        if not key:
            continue
        current = best.get(key)
        if current is None:
            best[key] = paper
            continue
        if _metadata_richness(paper) > _metadata_richness(current):
            paper.arxiv_id = paper.arxiv_id or current.arxiv_id
            paper.semantic_scholar_id = paper.semantic_scholar_id or current.semantic_scholar_id
            best[key] = paper
        else:
            current.arxiv_id = current.arxiv_id or paper.arxiv_id
            current.semantic_scholar_id = (
                current.semantic_scholar_id or paper.semantic_scholar_id
            )
    return list(best.values())


async def search_arxiv(
    query: str, client: httpx.AsyncClient, max_results: int = 20
) -> list[PaperMetadata]:
    """Query the arXiv Atom API and parse entries into ``PaperMetadata``."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        resp = await client.get(ARXIV_ENDPOINT, params=params, timeout=30.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("arxiv_request_failed", query=query, error=str(exc))
        return []
    return _parse_arxiv(resp.text)


def _parse_arxiv(xml_text: str) -> list[PaperMetadata]:
    """Parse an arXiv Atom feed string into paper metadata."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("arxiv_parse_failed", error=str(exc))
        return []
    papers: list[PaperMetadata] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title_el = entry.find("atom:title", ATOM_NS)
        summary_el = entry.find("atom:summary", ATOM_NS)
        published_el = entry.find("atom:published", ATOM_NS)
        id_el = entry.find("atom:id", ATOM_NS)
        if title_el is None or title_el.text is None:
            continue
        authors = [
            name_el.text.strip()
            for author in entry.findall("atom:author", ATOM_NS)
            if (name_el := author.find("atom:name", ATOM_NS)) is not None and name_el.text
        ]
        year = 0
        if published_el is not None and published_el.text:
            try:
                year = int(published_el.text[:4])
            except ValueError:
                year = 0
        arxiv_id = ""
        url = ""
        if id_el is not None and id_el.text:
            url = id_el.text.strip()
            arxiv_id = _ARXIV_VERSION_RE.sub("", url.rsplit("/", 1)[-1])
        papers.append(
            PaperMetadata(
                arxiv_id=arxiv_id or None,
                semantic_scholar_id=None,
                title=" ".join(title_el.text.split()),
                authors=authors,
                abstract=(summary_el.text or "").strip() if summary_el is not None else "",
                year=year,
                venue="arXiv",
                citation_count=0,
                url=url or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""),
            )
        )
    return papers


async def search_semantic_scholar(
    query: str, client: httpx.AsyncClient, limit: int = 20
) -> list[PaperMetadata]:
    """Query the Semantic Scholar paper-search API into ``PaperMetadata``."""
    params = {"query": query, "fields": SEMANTIC_SCHOLAR_FIELDS, "limit": limit}
    try:
        resp = await client.get(SEMANTIC_SCHOLAR_ENDPOINT, params=params, timeout=30.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("semantic_scholar_request_failed", query=query, error=str(exc))
        return []
    return _parse_semantic_scholar(resp.json())


def _parse_semantic_scholar(payload: dict) -> list[PaperMetadata]:
    """Parse a Semantic Scholar search payload into paper metadata."""
    papers: list[PaperMetadata] = []
    for item in payload.get("data") or []:
        title = item.get("title")
        if not title:
            continue
        external = item.get("externalIds") or {}
        authors = [a.get("name", "") for a in item.get("authors") or [] if a.get("name")]
        papers.append(
            PaperMetadata(
                arxiv_id=external.get("ArXiv"),
                semantic_scholar_id=item.get("paperId"),
                title=" ".join(title.split()),
                authors=authors,
                abstract=item.get("abstract") or "",
                year=item.get("year") or 0,
                venue=item.get("venue") or None,
                citation_count=item.get("citationCount") or 0,
                url=item.get("url") or "",
            )
        )
    return papers


def _cosine_similarities(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and each row of ``matrix``."""
    query_norm = np.linalg.norm(query_vec) or 1.0
    row_norms = np.linalg.norm(matrix, axis=1)
    row_norms[row_norms == 0] = 1.0
    return (matrix @ query_vec) / (row_norms * query_norm)


def bm25_scores(papers: list[PaperMetadata], query: str) -> np.ndarray:
    """Return BM25 lexical relevance of the query against title+abstract.

    Scores are normalized to [0, 1] by the maximum so they combine cleanly with
    the dense and citation signals.
    """
    corpus = [_WORD_RE.findall(f"{p.title} {p.abstract}".lower()) for p in papers]
    if not corpus:
        return np.zeros(0)
    bm25 = BM25Okapi(corpus)
    scores = np.array(bm25.get_scores(_WORD_RE.findall(query.lower())), dtype=float)
    # BM25's IDF goes negative for terms present in >half the corpus; clamp those
    # to 0 so a very common term reads as "neutral", not "less relevant than absent".
    scores = np.maximum(scores, 0.0)
    top = scores.max() if scores.size else 0.0
    return scores / top if top > 0 else scores


def score_papers(
    papers: list[PaperMetadata],
    abstract_vecs: np.ndarray,
    query_vec: np.ndarray,
    lexical_scores: np.ndarray | None = None,
) -> None:
    """Record the dense/lexical/citation components and a weighted relevance.

    The weighted blend ``0.5 * cosine + 0.3 * bm25 + 0.2 * citation`` (citation =
    ``log(1 + c) / log(1 + max_c)``) is computed here and used directly when
    ``FUSION_METHOD == "weighted"``. Under the default ``"rrf"`` method
    :func:`apply_fusion` overwrites ``relevance_score`` afterwards using the
    recorded components. ``lexical_scores`` defaults to zeros when BM25 is
    unavailable.
    """
    sims = _cosine_similarities(query_vec, abstract_vecs)
    lexical = lexical_scores if lexical_scores is not None else np.zeros(len(papers))
    citations = np.array([max(p.citation_count, 0) for p in papers], dtype=float)
    max_citations = citations.max() if citations.size else 0.0
    if max_citations > 0:
        norm_citations = np.log1p(citations) / np.log1p(max_citations)
    else:
        norm_citations = np.zeros_like(citations)
    for paper, sim, lex, cit in zip(papers, sims, lexical, norm_citations, strict=False):
        paper.dense_score = float(sim)
        paper.lexical_score = float(lex)
        paper.citation_score = float(cit)
        paper.relevance_score = float(
            WEIGHT_DENSE * sim + WEIGHT_LEXICAL * lex + WEIGHT_CITATION * cit
        )


def apply_fusion(papers: list[PaperMetadata]) -> None:
    """Overwrite ``relevance_score`` with reciprocal rank fusion of the signals.

    Builds three rankings from the components ``score_papers`` already wrote on
    each paper — dense (exact cosine), lexical (BM25) and citation — and fuses
    them with RRF, weighted to preserve the semantic > lexical > citation
    emphasis. The dense ranking reuses each paper's exact ``dense_score`` rather
    than re-deriving the order through the ANN index: the candidate pool is small
    and the exact cosine is already in hand, so an approximate index would only
    add cost and approximation here (the HNSW index earns its keep at corpus
    scale — see ``app/retrieval/hnsw.py`` / ``benchmarks/hnsw_vs_faiss.py``). A
    no-op when ``FUSION_METHOD`` is not ``"rrf"``, leaving the weighted blend in
    place.
    """
    if settings.FUSION_METHOD != "rrf" or not papers:
        return
    dense_ranking = ranking_from_scores([p.dense_score for p in papers])
    lexical_ranking = ranking_from_scores([p.lexical_score for p in papers])
    citation_ranking = ranking_from_scores([p.citation_score for p in papers])
    fused = reciprocal_rank_fusion(
        [dense_ranking, lexical_ranking, citation_ranking],
        k=settings.RRF_K,
        weights=[WEIGHT_DENSE, WEIGHT_LEXICAL, WEIGHT_CITATION],
    )
    for idx, paper in enumerate(papers):
        paper.relevance_score = float(fused.get(idx, 0.0))


async def run_discovery(
    plan: ResearchPlan, query: str, session_id: str
) -> tuple[list[PaperMetadata], list[list[float]]]:
    """Search both sources, dedupe and rank papers for a research plan.

    Returns the top ``MAX_PAPERS_PER_QUERY`` papers together with their abstract
    embeddings (index-aligned), so the workflow node can persist both.
    """
    queries = plan.search_queries or [plan.goal or query]

    async def _search_source(fetch, rate_limit: float) -> list[PaperMetadata]:
        """Run every query against one source, respecting its per-host rate limit."""
        found: list[PaperMetadata] = []
        for idx, search_query in enumerate(queries):
            if idx > 0:
                await asyncio.sleep(rate_limit)
            found.extend(await fetch(search_query, client))
        return found

    # The two sources are independent and rate-limited per host, so run them
    # concurrently instead of one full loop after the other.
    # follow_redirects: arXiv now 301-redirects the bare http endpoint to https
    # (HSTS); without this httpx returns the empty 301 body and discovery finds
    # nothing. Harmless for the already-https endpoints.
    async with httpx.AsyncClient(
        headers={"User-Agent": "ResearchFlow/1.0"}, follow_redirects=True
    ) as client:
        arxiv_papers, ss_papers = await asyncio.gather(
            _search_source(search_arxiv, settings.ARXIV_RATE_LIMIT_SECONDS),
            _search_source(search_semantic_scholar, settings.SEMANTIC_SCHOLAR_RATE_LIMIT_SECONDS),
        )
    await events.papers_found(session_id, "arxiv", len(arxiv_papers), len(arxiv_papers))
    total = len(arxiv_papers) + len(ss_papers)
    await events.papers_found(session_id, "semantic_scholar", len(ss_papers), total)

    papers = deduplicate(arxiv_papers + ss_papers)
    if not papers:
        return [], []

    # One embedding pass for abstracts + the query (the query is the last row).
    abstracts = [p.abstract or p.title for p in papers]
    all_vecs = await embedding_client.aembed([*abstracts, query])
    abstract_vecs = all_vecs[:-1]
    query_vec = all_vecs[-1]
    lexical = bm25_scores(papers, query)
    score_papers(papers, abstract_vecs, query_vec, lexical)
    apply_fusion(papers)

    order = sorted(range(len(papers)), key=lambda i: papers[i].relevance_score, reverse=True)
    if settings.RERANK_ENABLED:
        order = await _apply_reranking(query, papers, order)

    top = order[: settings.MAX_PAPERS_PER_QUERY]
    ranked_papers = [papers[i] for i in top]
    ranked_vecs = [abstract_vecs[i].tolist() for i in top]
    return ranked_papers, ranked_vecs


async def _apply_reranking(
    query: str, papers: list[PaperMetadata], order: list[int]
) -> list[int]:
    """Rerank the top candidates with the cross-encoder and re-order them.

    Only the leading ``RERANK_TOP_N`` candidates (by hybrid score) are reranked;
    their ``relevance_score`` becomes ``0.7 * rerank + 0.3 * citation`` and they
    are re-sorted ahead of the untouched tail. Falls back to the hybrid order if
    the cross-encoder is unavailable.
    """
    candidate_idx = order[: settings.RERANK_TOP_N]
    candidates = [papers[i] for i in candidate_idx]
    if not await reranker.rerank(query, candidates):
        return order
    for paper in candidates:
        paper.relevance_score = float(
            RERANK_WEIGHT * paper.rerank_score + CITATION_WEIGHT * paper.citation_score
        )
    reordered = sorted(
        candidate_idx, key=lambda i: papers[i].relevance_score, reverse=True
    )
    return reordered + order[settings.RERANK_TOP_N :]
