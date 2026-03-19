"""Tests for workflow configuration and template configs."""

from __future__ import annotations

import pytest

from felix_agent_sdk.core.helix import HelixConfig
from felix_agent_sdk.workflows.config import (
    SynthesisStrategy,
    WorkflowConfig,
    WorkflowPhase,
    WorkflowResult,
)
from felix_agent_sdk.workflows.templates import (
    analysis_config,
    research_config,
    review_config,
)


class TestWorkflowPhase:
    def test_values(self):
        assert WorkflowPhase.EXPLORATION.value == "exploration"
        assert WorkflowPhase.ANALYSIS.value == "analysis"
        assert WorkflowPhase.SYNTHESIS.value == "synthesis"
        assert len(WorkflowPhase) == 3


class TestSynthesisStrategy:
    def test_values(self):
        assert SynthesisStrategy.BEST_RESULT.value == "best_result"
        assert SynthesisStrategy.COMPRESSED_MERGE.value == "compressed_merge"
        assert SynthesisStrategy.ROUND_ROBIN.value == "round_robin"
        assert len(SynthesisStrategy) == 3


class TestWorkflowConfig:
    def test_defaults(self):
        config = WorkflowConfig()
        assert config.confidence_threshold == 0.80
        assert config.max_rounds == 3
        assert config.synthesis_strategy == SynthesisStrategy.COMPRESSED_MERGE
        assert config.max_agents == 10
        assert config.enable_context_compression is True
        assert len(config.team_composition) == 3

    def test_custom_config(self):
        config = WorkflowConfig(
            helix_config=HelixConfig.research_heavy(),
            team_composition=[("research", {}), ("research", {})],
            confidence_threshold=0.90,
            max_rounds=5,
        )
        assert config.confidence_threshold == 0.90
        assert config.max_rounds == 5
        assert len(config.team_composition) == 2

    def test_default_team_composition(self):
        config = WorkflowConfig()
        types = [t for t, _ in config.team_composition]
        assert "research" in types
        assert "analysis" in types
        assert "critic" in types


class TestWorkflowResult:
    def test_construction(self):
        result = WorkflowResult(
            synthesis="Final answer",
            total_rounds=2,
            final_confidence=0.85,
        )
        assert result.synthesis == "Final answer"
        assert result.total_rounds == 2
        assert result.agent_results == []
        assert result.metadata == {}
        assert result.created_at > 0


class TestTemplateConfigs:
    def test_research_config(self):
        config = research_config()
        assert isinstance(config, WorkflowConfig)
        types = [t for t, _ in config.team_composition]
        assert types.count("research") == 2
        assert config.confidence_threshold == 0.75
        assert config.max_rounds == 4

    def test_analysis_config(self):
        config = analysis_config()
        assert isinstance(config, WorkflowConfig)
        types = [t for t, _ in config.team_composition]
        assert types.count("analysis") == 2
        assert config.confidence_threshold == 0.85

    def test_review_config(self):
        config = review_config()
        assert isinstance(config, WorkflowConfig)
        types = [t for t, _ in config.team_composition]
        assert types.count("critic") == 2
        assert config.confidence_threshold == 0.80

    def test_custom_thresholds(self):
        config = research_config(confidence_threshold=0.60, max_rounds=6)
        assert config.confidence_threshold == 0.60
        assert config.max_rounds == 6
