"""Tests for felix_agent_sdk.agents.llm_agent — LLMAgent, LLMTask, LLMResult."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from felix_agent_sdk.agents.llm_agent import LLMAgent, LLMResult, LLMTask
from felix_agent_sdk.core.helix import HelixGeometry
from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.providers.errors import ProviderError
from felix_agent_sdk.providers.types import CompletionResult


@pytest.fixture
def helix():
    return HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)


@pytest.fixture
def provider():
    p = MagicMock(spec=BaseProvider)
    p.complete.return_value = CompletionResult(
        content=(
            "The analysis reveals several key findings. First, the data indicates "
            "a strong correlation between renewable energy adoption and grid stability. "
            "Furthermore, evidence from multiple studies demonstrates that distributed "
            "generation improves resilience. However, intermittency challenges remain "
            "a specific concern requiring detailed attention."
        ),
        model="test-model",
        usage={"prompt_tokens": 100, "completion_tokens": 60, "total_tokens": 160},
    )
    return p


@pytest.fixture
def agent(helix, provider):
    return LLMAgent("llm-001", provider, helix, agent_type="research")


@pytest.fixture
def task():
    return LLMTask(
        task_id="task-001",
        description="Analyse the impact of renewable energy on grid stability.",
    )


# -------------------------------------------------------------------------
# LLMTask
# -------------------------------------------------------------------------


class TestLLMTask:
    def test_minimal_construction(self):
        t = LLMTask(task_id="t1", description="Do stuff")
        assert t.task_id == "t1"
        assert t.description == "Do stuff"
        assert t.context == ""
        assert t.metadata == {}
        assert t.context_history == []

    def test_full_construction(self):
        t = LLMTask(
            task_id="t2",
            description="desc",
            context="ctx",
            metadata={"key": "val"},
            context_history=[{"agent_id": "a1", "content": "hi"}],
        )
        assert t.context == "ctx"
        assert t.metadata["key"] == "val"
        assert len(t.context_history) == 1


# -------------------------------------------------------------------------
# LLMResult
# -------------------------------------------------------------------------


class TestLLMResult:
    def test_construction(self, provider):
        r = LLMResult(
            agent_id="a1",
            task_id="t1",
            content="output",
            position_info={"progress": 0.5},
            completion_result=provider.complete.return_value,
            processing_time=0.1,
            confidence=0.7,
            temperature_used=0.5,
            token_budget_used=160,
        )
        assert r.agent_id == "a1"
        assert r.confidence == 0.7


# -------------------------------------------------------------------------
# LLMAgent construction
# -------------------------------------------------------------------------


class TestLLMAgentConstruction:
    def test_provider_stored(self, agent, provider):
        assert agent.provider is provider

    def test_agent_type(self, agent):
        assert agent.agent_type == "research"

    def test_default_temperature_range_for_research(self, agent):
        assert agent.temperature_range == (0.4, 0.9)

    def test_custom_temperature_range(self, helix, provider):
        a = LLMAgent("a", provider, helix, temperature_range=(0.2, 0.5))
        assert a.temperature_range == (0.2, 0.5)

    def test_default_max_tokens(self, agent):
        assert agent.max_tokens == 4096

    def test_inherits_agent_state(self, agent):
        from felix_agent_sdk.agents.base import AgentState

        assert agent.state == AgentState.WAITING


# -------------------------------------------------------------------------
# Adaptive temperature
# -------------------------------------------------------------------------


class TestAdaptiveTemperature:
    def test_temperature_at_t0_is_high(self, agent):
        agent._progress = 0.0
        temp = agent.get_adaptive_temperature()
        assert temp == pytest.approx(0.9, abs=0.01)

    def test_temperature_at_t1_is_low(self, agent):
        agent._progress = 1.0
        temp = agent.get_adaptive_temperature()
        assert temp == pytest.approx(0.4, abs=0.01)

    def test_temperature_at_midpoint(self, agent):
        agent._progress = 0.5
        temp = agent.get_adaptive_temperature()
        assert 0.4 < temp < 0.9

    def test_temperature_monotonically_decreasing(self, agent):
        temps = []
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            agent._progress = t
            temps.append(agent.get_adaptive_temperature())
        assert temps == sorted(temps, reverse=True)


# -------------------------------------------------------------------------
# Confidence calculation
# -------------------------------------------------------------------------


class TestConfidenceCalculation:
    def test_empty_content_low_confidence(self, agent):
        c = agent.calculate_confidence("")
        assert c < 0.5

    def test_short_content_lower(self, agent):
        c = agent.calculate_confidence("ok")
        assert c < 0.5

    def test_adequate_content_moderate(self, agent):
        content = (
            "The data indicates a strong correlation. Furthermore, evidence suggests "
            "that the approach is sound. However, additional research is needed to "
            "confirm these findings with specific examples and detailed analysis."
        )
        c = agent.calculate_confidence(content)
        assert 0.3 <= c <= 0.6

    def test_confidence_between_0_and_cap(self, agent):
        c = agent.calculate_confidence("anything")
        assert 0.0 <= c <= 0.6  # research cap

    def test_analysis_agent_higher_cap(self, helix, provider):
        a = LLMAgent("a", provider, helix, agent_type="analysis")
        c = a.calculate_confidence("Some analysis with therefore and because.")
        assert c <= 0.8

    def test_synthesis_agent_highest_cap(self, helix, provider):
        a = LLMAgent("a", provider, helix, agent_type="synthesis")
        a._progress = 1.0
        c = a.calculate_confidence(
            "The comprehensive synthesis of the analysis shows that because of "
            "multiple data points, the conclusion is supported. Furthermore, "
            "the evidence demonstrates specific examples of the phenomenon."
        )
        assert c <= 0.95


# -------------------------------------------------------------------------
# Content quality analysis
# -------------------------------------------------------------------------


class TestContentQualityAnalysis:
    def test_empty_returns_zero(self, agent):
        assert agent._analyze_content_quality("") == 0.0

    def test_whitespace_returns_zero(self, agent):
        assert agent._analyze_content_quality("   ") == 0.0

    def test_good_content_scores_above_half(self, agent):
        content = (
            "The research analysis reveals important data. Because of multiple "
            "studies, we can demonstrate a specific conclusion. Furthermore, "
            "the evidence indicates particular patterns in the 42 samples examined."
        )
        score = agent._analyze_content_quality(content)
        assert score > 0.5

    def test_score_between_0_and_1(self, agent):
        for text in ["hi", "a" * 5000, "normal sentence."]:
            score = agent._analyze_content_quality(text)
            assert 0.0 <= score <= 1.0


# -------------------------------------------------------------------------
# Position-aware prompting
# -------------------------------------------------------------------------


class TestPositionAwarePrompt:
    def test_exploration_phase_keywords(self, agent, task):
        agent._progress = 0.1
        sys_prompt, user_prompt = agent.create_position_aware_prompt(task)
        assert "EXPLORATION" in sys_prompt
        assert "diverse" in sys_prompt.lower() or "broad" in sys_prompt.lower()

    def test_analysis_phase_keywords(self, agent, task):
        agent._progress = 0.5
        sys_prompt, _ = agent.create_position_aware_prompt(task)
        assert "ANALYSIS" in sys_prompt

    def test_synthesis_phase_keywords(self, agent, task):
        agent._progress = 0.9
        sys_prompt, _ = agent.create_position_aware_prompt(task)
        assert "SYNTHESIS" in sys_prompt

    def test_returns_two_strings(self, agent, task):
        result = agent.create_position_aware_prompt(task)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)

    def test_user_prompt_contains_description(self, agent, task):
        _, user_prompt = agent.create_position_aware_prompt(task)
        assert "renewable energy" in user_prompt

    def test_context_history_incorporated(self, agent):
        t = LLMTask(
            task_id="t2",
            description="Summarise findings.",
            context_history=[
                {"agent_id": "research-001", "content": "Found important data about X."}
            ],
        )
        _, user_prompt = agent.create_position_aware_prompt(t)
        assert "research-001" in user_prompt
        assert "important data" in user_prompt


# -------------------------------------------------------------------------
# Provider integration (process_task)
# -------------------------------------------------------------------------


class TestProcessTask:
    def test_calls_provider_complete(self, agent, task, provider):
        agent.spawn(0.0)
        agent.process_task(task)
        provider.complete.assert_called_once()

    def test_passes_temperature(self, agent, task, provider):
        agent.spawn(0.0)
        agent.process_task(task)
        call_kwargs = provider.complete.call_args
        assert "temperature" in call_kwargs.kwargs

    def test_returns_llm_result(self, agent, task):
        agent.spawn(0.0)
        result = agent.process_task(task)
        assert isinstance(result, LLMResult)
        assert result.agent_id == "llm-001"
        assert result.task_id == "task-001"

    def test_records_confidence(self, agent, task):
        agent.spawn(0.0)
        agent.process_task(task)
        assert len(agent._confidence_history) == 1

    def test_measures_processing_time(self, agent, task):
        agent.spawn(0.0)
        result = agent.process_task(task)
        assert result.processing_time > 0

    def test_tracks_total_tokens(self, agent, task):
        agent.spawn(0.0)
        agent.process_task(task)
        assert agent.total_tokens_used == 160

    def test_stores_result_in_history(self, agent, task):
        agent.spawn(0.0)
        agent.process_task(task)
        assert len(agent.processing_results) == 1

    def test_provider_error_propagates(self, agent, task, provider):
        provider.complete.side_effect = ProviderError("fail", provider="test")
        agent.spawn(0.0)
        with pytest.raises(ProviderError):
            agent.process_task(task)

    def test_result_includes_position_info(self, agent, task):
        agent.spawn(0.0)
        result = agent.process_task(task)
        assert "progress" in result.position_info
        assert "phase" in result.position_info


# -------------------------------------------------------------------------
# Helical checkpoints
# -------------------------------------------------------------------------


class TestHelicalCheckpoints:
    def test_checkpoint_values(self):
        assert LLMAgent.HELICAL_CHECKPOINTS == [0.0, 0.3, 0.5, 0.7, 0.9]

    def test_should_process_at_first_checkpoint(self, agent):
        agent._progress = 0.0
        assert agent.should_process_at_checkpoint() is True

    def test_should_not_process_after_marking(self, agent):
        agent._progress = 0.0
        agent.mark_checkpoint_processed()
        assert agent.should_process_at_checkpoint() is False

    def test_should_process_at_next_checkpoint(self, agent):
        agent._progress = 0.0
        agent.mark_checkpoint_processed()
        agent._progress = 0.35
        assert agent.should_process_at_checkpoint() is True

    def test_no_new_checkpoint_between_markers(self, agent):
        agent._progress = 0.3
        agent.mark_checkpoint_processed()
        agent._progress = 0.35
        assert agent.should_process_at_checkpoint() is False


# -------------------------------------------------------------------------
# Repr
# -------------------------------------------------------------------------


class TestLLMAgentRepr:
    def test_repr_contains_type(self, agent):
        assert "research" in repr(agent)

    def test_repr_contains_id(self, agent):
        assert "llm-001" in repr(agent)
