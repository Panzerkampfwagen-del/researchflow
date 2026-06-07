"""Tests for the enhancement features: json_repair, BM25 ranking, verification."""

from __future__ import annotations

import numpy as np

from app.agents.discovery import WEIGHT_LEXICAL, _apply_reranking, bm25_scores, score_papers
from app.agents.grounding import extract_claims, grounding_checker, split_sentences
from app.agents.reranker import reranker
from app.agents.verification import verify_report
from app.core.llm import LLMResponse, llm_client
from app.evaluation.judge import judge_report
from app.graph.state import PaperMetadata, ResearchPlan


class TestJsonRepairFallback:
    async def test_repairs_malformed_json_without_retry(self, monkeypatch):
        # Single quotes + trailing comma: invalid JSON, but repairable.
        malformed = (
            "{'goal': 'g', 'subtopics': [], 'search_queries': [], "
            "'year_start': 2020, 'year_end': 2024,}"
        )
        calls = {"n": 0}

        async def fake_complete(messages, use_reasoning=True, temperature=0.1):
            calls["n"] += 1
            return LLMResponse(content=malformed, tokens=5, cost_usd=0.0, model="test")

        monkeypatch.setattr(llm_client, "complete", fake_complete)
        plan, _ = await llm_client.structured_complete([], ResearchPlan)
        assert plan.goal == "g"
        assert calls["n"] == 1  # repaired in place, no correction round-trip


class TestBM25:
    def test_lexical_scores_rank_matching_doc_first(self):
        # Several docs so the query terms have a non-zero BM25 IDF.
        papers = [
            PaperMetadata(title="LLM weight quantization", abstract="quantize language models"),
            PaperMetadata(title="Image segmentation", abstract="semantic segmentation of images"),
            PaperMetadata(title="Reinforcement learning", abstract="policy gradient methods"),
            PaperMetadata(title="Graph neural networks", abstract="message passing on graphs"),
        ]
        scores = bm25_scores(papers, "llm weight quantization")
        assert scores[0] == 1.0  # normalized max
        assert all(scores[0] > scores[i] for i in (1, 2, 3))

    def test_empty_corpus(self):
        assert bm25_scores([], "anything").shape == (0,)

    def test_score_papers_records_components(self):
        papers = [
            PaperMetadata(title="a", abstract="x", citation_count=0),
            PaperMetadata(title="b", abstract="y", citation_count=100),
        ]
        abstract_vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        query_vec = np.array([1.0, 0.0])
        lexical = np.array([0.2, 0.9])
        score_papers(papers, abstract_vecs, query_vec, lexical)
        assert papers[0].lexical_score == 0.2
        assert papers[1].lexical_score == 0.9
        # combined score includes the weighted lexical contribution
        assert papers[1].relevance_score >= WEIGHT_LEXICAL * 0.9
        assert all(0.0 <= p.relevance_score <= 1.0 for p in papers)


class TestReranker:
    async def test_rerank_normalizes_scores(self, monkeypatch):
        papers = [
            PaperMetadata(title="a", abstract="x"),
            PaperMetadata(title="b", abstract="y"),
            PaperMetadata(title="c", abstract="z"),
        ]
        monkeypatch.setattr(reranker, "score", lambda q, texts: [0.1, 0.9, 0.5])
        ran = await reranker.rerank("query", papers)
        assert ran is True
        assert papers[0].rerank_score == 0.0
        assert papers[1].rerank_score == 1.0  # max normalized to 1
        assert papers[2].rerank_score == 0.5

    async def test_rerank_empty(self):
        assert await reranker.rerank("query", []) is False

    async def test_apply_reranking_reorders_top_candidates(self, monkeypatch):
        # Hybrid order is [0, 1, 2]; the cross-encoder strongly prefers paper 2.
        papers = [
            PaperMetadata(title="p0", abstract="a", relevance_score=0.9, citation_score=0.0),
            PaperMetadata(title="p1", abstract="b", relevance_score=0.6, citation_score=0.0),
            PaperMetadata(title="p2", abstract="c", relevance_score=0.3, citation_score=0.0),
        ]
        monkeypatch.setattr(reranker, "score", lambda q, texts: [0.1, 0.2, 0.9])
        new_order = await _apply_reranking("query", papers, [0, 1, 2])
        assert new_order[0] == 2  # reranked to the top despite lowest hybrid score


class TestCitationVerification:
    def test_grounded_and_hallucinated(self):
        citations = [
            {"index": 1, "title": "GPTQ: Accurate Post-Training Quantization"},
            {"index": 2, "title": "A Completely Fabricated Paper Title"},
        ]
        full_title = (
            "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
        )
        papers = [PaperMetadata(title=full_title)]
        narrative = "The method works well [1] but see also [3]."
        result = verify_report(citations, papers, narrative)

        verdicts = {c.index: c.verified for c in result.citations}
        assert verdicts[1] is True  # fuzzy-matches a retrieved paper
        assert verdicts[2] is False  # not in the retrieved set
        # dangling [3] reference is flagged
        assert any("[3]" in note for note in result.unsupported_references)
        assert result.verified_citations == 1
        assert result.hallucination_rate > 0.0

    def test_all_grounded_clean(self):
        citations = [{"index": 1, "title": "Deep Residual Learning for Image Recognition"}]
        papers = [PaperMetadata(title="Deep Residual Learning for Image Recognition")]
        result = verify_report(citations, papers, "As shown in [1].")
        assert result.verified_citations == 1
        assert result.unsupported_references == []
        assert result.hallucination_rate == 0.0


class TestGrounding:
    def test_split_sentences(self):
        sentences = split_sentences("First claim [1]. Second one [2]! A third?")
        assert len(sentences) == 3

    def test_extract_claims_only_keeps_cited_sentences(self):
        text = "Method A improves accuracy [1]. This sentence has no citation. See [2][3]."
        claims = extract_claims(text)
        assert len(claims) == 2
        assert claims[0][1] == [1]
        assert claims[1][1] == [2, 3]

    async def test_check_classifies_grounded_and_ungrounded(self, monkeypatch):
        # Two claims; the model entails the first strongly, the second weakly.
        narrative = "Approach X reaches state of the art [1]. Approach Y is unverified [2]."
        premises = {1: "Approach X reaches state of the art on the benchmark.", 2: "Off topic."}

        def fake_score(pairs):
            # pair order follows claim order: [ (premise1, claim1), (premise2, claim2) ]
            return [0.92, 0.10]

        monkeypatch.setattr(grounding_checker, "score", fake_score)
        result = await grounding_checker.check_with_premises(narrative, premises)
        assert result is not None
        assert result.total_claims == 2
        assert result.grounded_claims == 1
        assert result.claims[0].grounded is True
        assert result.claims[1].grounded is False
        assert result.ungrounded_rate == 0.5

    async def test_check_returns_none_when_disabled(self, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "GROUNDING_ENABLED", False)
        result = await grounding_checker.check_with_premises("A claim [1].", {1: "premise"})
        assert result is None

    async def test_check_returns_none_without_citable_claims(self):
        # No bracketed references → nothing to ground.
        result = await grounding_checker.check_with_premises("No citations here.", {1: "p"})
        assert result is None

    async def test_per_source_entailment_recorded(self, monkeypatch):
        # One claim citing two sources: only [1] entails it. Per-source scores
        # must differ so callers can score each citation, not just the claim max.
        narrative = "The result holds [1][2]."
        premises = {1: "The result holds on the benchmark.", 2: "Unrelated work."}
        monkeypatch.setattr(grounding_checker, "score", lambda pairs: [0.9, 0.1])
        result = await grounding_checker.check_with_premises(narrative, premises)
        claim = result.claims[0]
        assert claim.grounded is True  # best source entails
        assert claim.source_entailments[1] == 0.9
        assert claim.source_entailments[2] == 0.1  # this citation does NOT hold


class TestCitationParsing:
    def test_single_grouped_and_range(self):
        from app.agents.verification import parse_citation_indices

        assert parse_citation_indices("see [1] and [2]") == [1, 2]
        assert parse_citation_indices("as in [1, 2] also [4]") == [1, 2, 4]
        assert parse_citation_indices("range [1-3] and en-dash [5–6]") == [1, 2, 3, 5, 6]
        assert parse_citation_indices("non-numeric [CLS] [v2]") == []


class TestLLMJudge:
    async def test_parses_scores(self, monkeypatch):
        payload = (
            '{"coherence": 4, "relevance": 5, "gap_identification": 3, '
            '"rationale": "Well structured and on-topic."}'
        )

        async def fake_complete(messages, use_reasoning=True, temperature=0.1):
            return LLMResponse(content=payload, tokens=20, cost_usd=0.0, model="test")

        monkeypatch.setattr(llm_client, "complete", fake_complete)
        scores, usage = await judge_report("LLM quantization", "# Report\n...")
        assert scores.coherence == 4
        assert scores.relevance == 5
        assert scores.gap_identification == 3
        assert usage.tokens == 20
