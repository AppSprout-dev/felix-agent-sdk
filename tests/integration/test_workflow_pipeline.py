"""Integration tests for the full Felix workflow pipeline.

These tests run the complete workflow with a mocked LLM provider,
verifying that agents, communication hub, context builder, and
synthesizer all integrate correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.providers.types import CompletionResult
from felix_agent_sdk.workflows.config import SynthesisStrategy, WorkflowConfig
from felix_agent_sdk.workflows.runner import FelixWorkflow, run_felix_workflow
from felix_agent_sdk.workflows.templates import (
    analysis_config,
    research_config,
    review_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CALL_COUNTER = 0


def _make_multi_response_provider() -> BaseProvider:
    """Mock provider that returns distinct responses per call."""
    global _CALL_COUNTER  # noqa: PLW0603
    _CALL_COUNTER = 0

    provider = MagicMock(spec=BaseProvider)

    responses = [
        "The analysis reveals several key findings about renewable energy adoption. "
        "Solar capacity has grown 40% year-over-year with significant grid integration improvements.",
        "Further investigation shows distributed generation improves resilience. "
        "Battery storage costs have declined by 85% since 2010 enabling broader adoption.",
        "Critical review: the data supports the correlation between renewable adoption and stability. "
        "However, intermittency remains a concern requiring detailed attention to storage solutions.",
        "Comprehensive synthesis integrating all findings into a coherent framework. "
        "The evidence strongly supports accelerating renewable energy deployment.",
    ]

    def _complete(messages, **kwargs):
        global _CALL_COUNTER  # noqa: PLW0603
        idx = _CALL_COUNTER % len(responses)
        _CALL_COUNTER += 1
        return CompletionResult(
            content=responses[idx],
            model="test-model",
            usage={"prompt_tokens": 100, "completion_tokens": 60, "total_tokens": 160},
        )

    provider.complete.side_effect = _complete
    provider.count_tokens.return_value = 100
    return provider


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------


class TestResearchWorkflow:
    def test_full_pipeline(self):
        provider = _make_multi_response_provider()
        config = research_config()
        workflow = FelixWorkflow(config, provider)

        result = workflow.run("Analyse the impact of renewable energy on grid stability")

        assert result.synthesis != ""
        assert result.total_rounds >= 1
        assert result.final_confidence > 0
        assert len(result.agent_results) > 0
        assert result.metadata["agents_count"] == 4
        assert provider.complete.called

    def test_agents_produce_results(self):
        provider = _make_multi_response_provider()
        config = research_config(max_rounds=2)
        workflow = FelixWorkflow(config, provider)

        result = workflow.run("Research task")

        # Each agent that processed should have a result
        for agent_result in result.agent_results:
            assert agent_result.content != ""
            assert agent_result.confidence > 0
            assert agent_result.agent_id != ""


class TestAnalysisWorkflow:
    def test_convergence(self):
        """High-confidence responses should cause early convergence."""
        provider = MagicMock(spec=BaseProvider)
        provider.complete.return_value = CompletionResult(
            content=(
                "Detailed analysis with evidence-based conclusions. The data indicates "
                "strong correlation supported by multiple research studies demonstrating "
                "clear patterns. Furthermore, the specific metrics show 95% confidence "
                "in the primary hypothesis."
            ),
            model="test-model",
            usage={"prompt_tokens": 50, "completion_tokens": 50, "total_tokens": 100},
        )
        provider.count_tokens.return_value = 50

        # Use analysis config but with lower threshold to make convergence easier
        config = analysis_config(confidence_threshold=0.50)
        workflow = FelixWorkflow(config, provider)

        result = workflow.run("Analyse data patterns")

        assert result.metadata.get("converged") is True
        assert result.total_rounds <= config.max_rounds


class TestReviewWorkflow:
    def test_multi_critic(self):
        provider = _make_multi_response_provider()
        config = review_config()
        workflow = FelixWorkflow(config, provider)

        result = workflow.run("Review the proposal")

        # Review template has 2 critics
        agent_types = [t for t, _ in result.metadata["team_composition"]]
        assert agent_types.count("critic") == 2
        assert result.synthesis != ""


class TestForcedSynthesis:
    def test_runs_all_rounds_when_threshold_unreachable(self):
        provider = _make_multi_response_provider()
        config = WorkflowConfig(
            confidence_threshold=0.99,  # Unreachable
            max_rounds=2,
            team_composition=[("research", {}), ("analysis", {})],
        )
        workflow = FelixWorkflow(config, provider)

        result = workflow.run("Test task")

        assert result.total_rounds == 2
        assert result.metadata.get("converged") is False


class TestWorkflowCleanup:
    def test_hub_shutdown(self):
        """Verify hub and spokes are cleaned up even on normal completion."""
        provider = _make_multi_response_provider()
        config = WorkflowConfig(
            max_rounds=1,
            team_composition=[("research", {})],
        )
        workflow = FelixWorkflow(config, provider)

        result = workflow.run("Test cleanup")

        # If cleanup failed we'd get errors on subsequent runs
        result2 = workflow.run("Second run")
        assert result2.synthesis != ""


class TestSynthesisStrategies:
    def test_best_result_strategy(self):
        provider = _make_multi_response_provider()
        config = WorkflowConfig(
            synthesis_strategy=SynthesisStrategy.BEST_RESULT,
            max_rounds=1,
            team_composition=[("research", {}), ("analysis", {})],
        )
        workflow = FelixWorkflow(config, provider)

        result = workflow.run("Test best result")

        assert result.synthesis != ""
        # Should not make an extra provider call for synthesis
        # (calls = agent processing only)

    def test_round_robin_strategy(self):
        provider = _make_multi_response_provider()
        config = WorkflowConfig(
            synthesis_strategy=SynthesisStrategy.ROUND_ROBIN,
            max_rounds=1,
            team_composition=[("research", {})],
        )
        workflow = FelixWorkflow(config, provider)

        result = workflow.run("Test round robin")
        assert result.synthesis != ""


class TestWorkflowMetadata:
    def test_metadata_fields(self):
        provider = _make_multi_response_provider()
        config = WorkflowConfig(
            max_rounds=1,
            team_composition=[("research", {}), ("critic", {})],
        )
        workflow = FelixWorkflow(config, provider)

        result = workflow.run("Metadata test")

        assert "elapsed_seconds" in result.metadata
        assert "total_tokens" in result.metadata
        assert "converged" in result.metadata
        assert "agents_count" in result.metadata
        assert result.metadata["agents_count"] == 2
        assert result.metadata["total_tokens"] >= 0


class TestConvenienceFunction:
    def test_run_felix_workflow(self):
        provider = _make_multi_response_provider()
        config = WorkflowConfig(
            max_rounds=1,
            team_composition=[("research", {})],
        )

        result = run_felix_workflow(config, provider, "Convenience test")

        assert result.synthesis != ""
        assert isinstance(result.total_rounds, int)
