"""Unit tests for agent logic with LLM and HTTP calls mocked out."""

from __future__ import annotations

import numpy as np

from app.agents.analysis import run_analysis
from app.agents.discovery import (
    _parse_arxiv,
    _parse_semantic_scholar,
    deduplicate,
    normalize_title,
    score_papers,
)
from app.agents.planner import run_planner
from app.core.llm import LLMResponse, llm_client
from app.graph.state import PaperMetadata

ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2210.17323v1</id>
    <title>GPTQ: Accurate Post-Training Quantization</title>
    <summary>We present GPTQ, a one-shot weight quantization method.</summary>
    <published>2022-10-31T00:00:00Z</published>
    <author><name>Elias Frantar</name></author>
    <author><name>Saleh Ashkboos</name></author>
  </entry>
</feed>
"""

SS_PAYLOAD = {
    "data": [
        {
            "paperId": "abc123",
            "title": "GPTQ: Accurate Post-Training Quantization",
            "abstract": "We present GPTQ, a one-shot weight quantization method.",
            "year": 2022,
            "venue": "ICLR",
            "citationCount": 500,
            "externalIds": {"ArXiv": "2210.17323"},
            "url": "https://semanticscholar.org/paper/abc123",
            "authors": [{"name": "Elias Frantar"}],
        }
    ]
}


class TestNormalizeAndDedup:
    def test_normalize_title(self):
        result = normalize_title("GPTQ: Post-Training, Quantization!")
        assert result == "gptq posttraining quantization"

    def test_deduplicate_keeps_richer(self):
        sparse = PaperMetadata(title="Same Title", arxiv_id="1234.5678")
        rich = PaperMetadata(
            title="same title",
            abstract="A long and detailed abstract about quantization methods.",
            authors=["A", "B"],
            venue="ICLR",
            citation_count=100,
            semantic_scholar_id="ss-99",
        )
        result = deduplicate([sparse, rich])
        assert len(result) == 1
        kept = result[0]
        assert kept.venue == "ICLR"
        # external id from the discarded record is merged onto the kept one
        assert kept.arxiv_id == "1234.5678"
        assert kept.semantic_scholar_id == "ss-99"


class TestParsers:
    def test_parse_arxiv(self):
        papers = _parse_arxiv(ARXIV_XML)
        assert len(papers) == 1
        paper = papers[0]
        assert paper.arxiv_id == "2210.17323"
        assert paper.year == 2022
        assert len(paper.authors) == 2
        assert paper.venue == "arXiv"

    def test_parse_arxiv_malformed(self):
        assert _parse_arxiv("not xml at all") == []

    def test_parse_semantic_scholar(self):
        papers = _parse_semantic_scholar(SS_PAYLOAD)
        assert len(papers) == 1
        paper = papers[0]
        assert paper.semantic_scholar_id == "abc123"
        assert paper.arxiv_id == "2210.17323"
        assert paper.citation_count == 500
        assert paper.venue == "ICLR"

    def test_parse_semantic_scholar_empty(self):
        assert _parse_semantic_scholar({}) == []


class TestScoring:
    def test_score_orders_by_similarity_and_citations(self):
        papers = [
            PaperMetadata(title="far", abstract="x", citation_count=0),
            PaperMetadata(title="near", abstract="y", citation_count=1000),
        ]
        query_vec = np.array([1.0, 0.0])
        abstract_vecs = np.array([[0.0, 1.0], [1.0, 0.0]])
        score_papers(papers, abstract_vecs, query_vec)
        assert papers[1].relevance_score > papers[0].relevance_score
        assert all(0.0 <= p.relevance_score <= 1.0 for p in papers)


class TestPlanner:
    async def test_parses_plan_and_overrides_years(self, monkeypatch):
        plan_json = (
            '{"goal": "Survey LLM quantization", '
            '"subtopics": ["PTQ", "QAT"], '
            '"search_queries": ["llm quantization", "post training quantization"], '
            '"year_start": 2018, "year_end": 2023}'
        )

        async def fake_complete(messages, use_reasoning=True, temperature=0.1):
            return LLMResponse(content=plan_json, tokens=42, cost_usd=0.0, model="test")

        monkeypatch.setattr(llm_client, "complete", fake_complete)
        plan, usage = await run_planner("LLM quantization", year_start=2020, year_end=2024)
        assert plan.goal == "Survey LLM quantization"
        assert plan.subtopics == ["PTQ", "QAT"]
        assert plan.year_start == 2020
        assert plan.year_end == 2024
        assert usage.tokens == 42

    async def test_retries_on_bad_json(self, monkeypatch):
        calls = {"n": 0}
        good = (
            '{"goal": "g", "subtopics": [], "search_queries": [], '
            '"year_start": 2020, "year_end": 2024}'
        )

        async def flaky_complete(messages, use_reasoning=True, temperature=0.1):
            calls["n"] += 1
            content = "not json" if calls["n"] == 1 else good
            return LLMResponse(content=content, tokens=10, cost_usd=0.0, model="test")

        monkeypatch.setattr(llm_client, "complete", flaky_complete)
        plan, _ = await run_planner("q")
        assert plan.goal == "g"
        assert calls["n"] == 2


class TestAnalysis:
    async def test_insufficient_abstract_skips_llm(self):
        papers = [PaperMetadata(title="Tiny", abstract="too short", paper_id="p1")]
        analyses, usage = await run_analysis(papers, session_id="test-session")
        assert len(analyses) == 1
        assert analyses[0].confidence == 0.3
        assert analyses[0].methodology == "Insufficient abstract"
        assert usage.tokens == 0

    async def test_normal_abstract_extraction(self, monkeypatch):
        analysis_json = (
            '{"problem": "p", "methodology": "weight quantization", '
            '"datasets": ["C4"], "metrics": ["perplexity"], '
            '"key_results": "3-bit", "limitations": "none", "confidence": 0.9}'
        )

        async def fake_complete(messages, use_reasoning=True, temperature=0.1):
            return LLMResponse(content=analysis_json, tokens=30, cost_usd=0.0, model="test")

        monkeypatch.setattr(llm_client, "complete", fake_complete)
        long_abstract = "This paper introduces a weight quantization method. " * 5
        papers = [PaperMetadata(title="GPTQ", abstract=long_abstract, paper_id="paper-uuid")]
        analyses, usage = await run_analysis(papers, session_id="s")
        assert analyses[0].paper_id == "paper-uuid"
        assert analyses[0].methodology == "weight quantization"
        assert usage.tokens == 30
