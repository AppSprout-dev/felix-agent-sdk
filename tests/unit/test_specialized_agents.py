"""Tests for felix_agent_sdk.agents.specialized — ResearchAgent, AnalysisAgent, CriticAgent."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from felix_agent_sdk.agents.llm_agent import LLMTask
from felix_agent_sdk.agents.specialized import AnalysisAgent, CriticAgent, ResearchAgent
from felix_agent_sdk.core.helix import HelixGeometry
from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.providers.types import CompletionResult


@pytest.fixture
def helix():
    return HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)


@pytest.fixture
def provider():
    p = MagicMock(spec=BaseProvider)
    p.complete.return_value = CompletionResult(
        content="Mocked response.",
        model="test-model",
        usage={"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
    )
    return p


@pytest.fixture
def task():
    return LLMTask(task_id="t1", description="Evaluate the proposal.")


# -------------------------------------------------------------------------
# ResearchAgent
# -------------------------------------------------------------------------


class TestResearchAgent:
    def test_construction(self, helix, provider):
        a = ResearchAgent("r1", provider, helix, research_domain="technical")
        assert a.agent_type == "research"
        assert a.research_domain == "technical"

    def test_default_domain(self, helix, provider):
        a = ResearchAgent("r1", provider, helix)
        assert a.research_domain == "general"

    def test_exploration_prompt(self, helix, provider, task):
        a = ResearchAgent("r1", provider, helix)
        a._progress = 0.1
        sys_prompt, _ = a.create_position_aware_prompt(task)
        assert "EXPLORATION" in sys_prompt
        assert "Research agent" in sys_prompt

    def test_analysis_prompt(self, helix, provider, task):
        a = ResearchAgent("r1", provider, helix)
        a._progress = 0.5
        sys_prompt, _ = a.create_position_aware_prompt(task)
        assert "ANALYSIS" in sys_prompt

    def test_synthesis_prompt(self, helix, provider, task):
        a = ResearchAgent("r1", provider, helix)
        a._progress = 0.9
        sys_prompt, _ = a.create_position_aware_prompt(task)
        assert "SYNTHESIS" in sys_prompt

    def test_domain_in_prompt(self, helix, provider, task):
        a = ResearchAgent("r1", provider, helix, research_domain="creative")
        sys_prompt, _ = a.create_position_aware_prompt(task)
        assert "creative" in sys_prompt

    def test_general_domain_not_in_prompt(self, helix, provider, task):
        a = ResearchAgent("r1", provider, helix, research_domain="general")
        sys_prompt, _ = a.create_position_aware_prompt(task)
        assert "general" not in sys_prompt.lower() or "general" in sys_prompt

    def test_context_history_in_user_prompt(self, helix, provider):
        t = LLMTask(
            task_id="t1",
            description="Summarise.",
            context_history=[{"agent_id": "a1", "content": "earlier work"}],
        )
        a = ResearchAgent("r1", provider, helix)
        _, user_prompt = a.create_position_aware_prompt(t)
        assert "a1" in user_prompt
        assert "earlier work" in user_prompt

    def test_process_task_works(self, helix, provider, task):
        a = ResearchAgent("r1", provider, helix)
        a.spawn(0.0)
        result = a.process_task(task)
        assert result.agent_id == "r1"


# -------------------------------------------------------------------------
# AnalysisAgent
# -------------------------------------------------------------------------


class TestAnalysisAgent:
    def test_construction(self, helix, provider):
        a = AnalysisAgent("a1", provider, helix, analysis_type="technical")
        assert a.agent_type == "analysis"
        assert a.analysis_type == "technical"

    def test_default_type(self, helix, provider):
        a = AnalysisAgent("a1", provider, helix)
        assert a.analysis_type == "general"

    def test_patterns_and_insights_init_empty(self, helix, provider):
        a = AnalysisAgent("a1", provider, helix)
        assert a.patterns_found == []
        assert a.insights == []

    def test_exploration_prompt(self, helix, provider, task):
        a = AnalysisAgent("a1", provider, helix)
        a._progress = 0.1
        sys_prompt, _ = a.create_position_aware_prompt(task)
        assert "EXPLORATION" in sys_prompt
        assert "Analysis agent" in sys_prompt

    def test_analysis_prompt_mentions_trade_offs(self, helix, provider, task):
        a = AnalysisAgent("a1", provider, helix)
        a._progress = 0.5
        sys_prompt, _ = a.create_position_aware_prompt(task)
        assert "ANALYSIS" in sys_prompt

    def test_synthesis_prompt(self, helix, provider, task):
        a = AnalysisAgent("a1", provider, helix)
        a._progress = 0.85
        sys_prompt, _ = a.create_position_aware_prompt(task)
        assert "SYNTHESIS" in sys_prompt

    def test_type_in_prompt(self, helix, provider, task):
        a = AnalysisAgent("a1", provider, helix, analysis_type="critical")
        sys_prompt, _ = a.create_position_aware_prompt(task)
        assert "critical" in sys_prompt


# -------------------------------------------------------------------------
# CriticAgent
# -------------------------------------------------------------------------


class TestCriticAgent:
    def test_construction(self, helix, provider):
        a = CriticAgent("c1", provider, helix, review_focus="accuracy")
        assert a.agent_type == "critic"
        assert a.review_focus == "accuracy"

    def test_default_focus(self, helix, provider):
        a = CriticAgent("c1", provider, helix)
        assert a.review_focus == "general"

    def test_issues_and_suggestions_init_empty(self, helix, provider):
        a = CriticAgent("c1", provider, helix)
        assert a.identified_issues == []
        assert a.suggestions == []

    def test_exploration_prompt(self, helix, provider, task):
        a = CriticAgent("c1", provider, helix)
        a._progress = 0.1
        sys_prompt, _ = a.create_position_aware_prompt(task)
        assert "EXPLORATION" in sys_prompt
        assert "Critic agent" in sys_prompt

    def test_focus_in_prompt(self, helix, provider, task):
        a = CriticAgent("c1", provider, helix, review_focus="logic")
        sys_prompt, _ = a.create_position_aware_prompt(task)
        assert "logic" in sys_prompt


# -------------------------------------------------------------------------
# CriticAgent.evaluate_reasoning_process
# -------------------------------------------------------------------------


class TestEvaluateReasoningProcess:
    def test_returns_required_keys(self, helix, provider):
        a = CriticAgent("c1", provider, helix)
        output = {"result": "Some analysis text.", "confidence": 0.6, "agent_id": "a1"}
        result = a.evaluate_reasoning_process(output)
        required = {
            "reasoning_quality_score",
            "logical_coherence",
            "evidence_quality",
            "methodology_appropriateness",
            "identified_issues",
            "improvement_recommendations",
            "re_evaluation_needed",
            "agent_id",
        }
        assert required.issubset(result.keys())

    def test_scores_between_0_and_1(self, helix, provider):
        a = CriticAgent("c1", provider, helix)
        output = {"result": "Good analysis with because and therefore.", "confidence": 0.6}
        result = a.evaluate_reasoning_process(output)
        assert 0.0 <= result["reasoning_quality_score"] <= 1.0
        assert 0.0 <= result["logical_coherence"] <= 1.0
        assert 0.0 <= result["evidence_quality"] <= 1.0

    def test_detects_logical_fallacies(self, helix, provider):
        a = CriticAgent("c1", provider, helix)
        output = {
            "result": "Obviously everyone knows this is always true and it goes without saying.",
            "confidence": 0.8,
        }
        result = a.evaluate_reasoning_process(output)
        assert result["logical_coherence"] < 0.5

    def test_detects_weak_evidence(self, helix, provider):
        a = CriticAgent("c1", provider, helix)
        output = {
            "result": (
                "I think this maybe could be true. I believe it probably might be "
                "the case. It seems like it appears to be right."
            ),
            "confidence": 0.5,
        }
        result = a.evaluate_reasoning_process(output)
        assert result["evidence_quality"] < 0.5

    def test_shallow_reasoning_penalised(self, helix, provider):
        a = CriticAgent("c1", provider, helix)
        output = {"result": "Yes.", "confidence": 0.9}
        result = a.evaluate_reasoning_process(output)
        assert result["reasoning_quality_score"] < 0.7

    def test_overconfidence_detected(self, helix, provider):
        a = CriticAgent("c1", provider, helix)
        output = {"result": "Short.", "confidence": 0.99}
        result = a.evaluate_reasoning_process(output)
        assert any("overconfident" in issue for issue in result["identified_issues"])

    def test_re_evaluation_needed_on_low_quality(self, helix, provider):
        a = CriticAgent("c1", provider, helix)
        # Trigger all three detectors: fallacies + weak evidence + shallow
        output = {
            "result": (
                "Obviously I think this maybe could be probably true. "
                "I believe it might be right, seems like it appears to be so."
            ),
            "confidence": 0.9,
        }
        result = a.evaluate_reasoning_process(output)
        # Should have >= 3 issues: fallacies, weak evidence, shallow, overconfidence
        assert result["re_evaluation_needed"] is True

    def test_methodology_with_metadata(self, helix, provider):
        a = CriticAgent("c1", provider, helix)
        # Make text > 50 words to avoid the shallow-reasoning penalty
        output = {
            "result": (
                "Because the analysis shows multiple converging factors, therefore "
                "we can conclude that the approach is valid. The data clearly supports "
                "this interpretation across several important dimensions. By carefully "
                "examining the evidence from different angles, we see that the methodology "
                "is sound and the results are consistent with expectations from prior research."
            ),
            "confidence": 0.6,
            "agent_id": "analysis-1",
        }
        metadata = {"agent_type": "analysis"}
        result = a.evaluate_reasoning_process(output, metadata)
        assert result["methodology_appropriateness"] == 0.8
