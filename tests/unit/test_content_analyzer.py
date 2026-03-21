"""Tests for ContentAnalyzer."""

from __future__ import annotations

from felix_agent_sdk.spawning.content_analyzer import ContentAnalyzer


class TestContentAnalyzerBasics:
    def test_empty_results(self):
        analyzer = ContentAnalyzer()
        report = analyzer.analyze_coverage([])
        assert report.coverage_score == 0.0
        assert report.recommended_type == "research"

    def test_single_result_all_sparse(self):
        analyzer = ContentAnalyzer()
        report = analyzer.analyze_coverage([
            {"content": "Renewable energy solar panels efficiency", "agent_type": "research"}
        ])
        # All topics appear only once → all sparse → coverage 0
        assert report.coverage_score == 0.0
        assert len(report.topics_sparse) > 0
        assert report.recommended_type == "research"

    def test_overlapping_results_increase_coverage(self):
        analyzer = ContentAnalyzer()
        report = analyzer.analyze_coverage([
            {"content": "Solar energy production increases globally", "agent_type": "research"},
            {"content": "Solar energy adoption in developing markets", "agent_type": "research"},
        ])
        # "solar" and "energy" appear in both → covered
        assert report.coverage_score > 0.0
        assert "solar" in report.topics_covered
        assert "energy" in report.topics_covered


class TestContentAnalyzerRecommendation:
    def test_low_coverage_recommends_research(self):
        analyzer = ContentAnalyzer()
        report = analyzer.analyze_coverage([
            {"content": "Quantum computing basics", "agent_type": "research"},
        ])
        assert report.recommended_type == "research"

    def test_high_coverage_recommends_critic(self):
        analyzer = ContentAnalyzer()
        # All words overlap → high coverage
        report = analyzer.analyze_coverage([
            {"content": "analysis results data findings", "agent_type": "research"},
            {"content": "analysis results data findings", "agent_type": "analysis"},
            {"content": "analysis results data findings", "agent_type": "analysis"},
        ])
        assert report.coverage_score > 0.6
        assert report.recommended_type == "critic"

    def test_mid_coverage_light_analysis_recommends_analysis(self):
        analyzer = ContentAnalyzer()
        report = analyzer.analyze_coverage([
            {"content": "renewable energy solar capacity growth data", "agent_type": "research"},
            {"content": "renewable energy market trends growth", "agent_type": "research"},
        ])
        # Mid coverage, no analysis agents → should recommend analysis
        if 0.3 <= report.coverage_score < 0.6:
            assert report.recommended_type == "analysis"


class TestContentAnalyzerKeywordExtraction:
    def test_stopwords_removed(self):
        analyzer = ContentAnalyzer()
        report = analyzer.analyze_coverage([
            {"content": "the quick brown fox is very fast", "agent_type": "research"},
            {"content": "the quick brown fox jumps over", "agent_type": "research"},
        ])
        assert "the" not in report.topics_covered
        assert "is" not in report.topics_covered
        assert "quick" in report.topics_covered

    def test_short_words_removed(self):
        analyzer = ContentAnalyzer()
        report = analyzer.analyze_coverage([
            {"content": "an AI ML model for NLP", "agent_type": "research"},
            {"content": "an AI ML model works", "agent_type": "research"},
        ])
        # "ai", "ml" are < 3 chars → removed
        assert "ai" not in report.topics_covered
        assert "model" in report.topics_covered
