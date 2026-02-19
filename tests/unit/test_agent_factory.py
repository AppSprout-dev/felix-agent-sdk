"""Tests for felix_agent_sdk.agents.factory — AgentFactory."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from felix_agent_sdk.agents.factory import AgentFactory
from felix_agent_sdk.agents.llm_agent import LLMAgent
from felix_agent_sdk.agents.specialized import AnalysisAgent, CriticAgent, ResearchAgent
from felix_agent_sdk.core.helix import HelixConfig
from felix_agent_sdk.providers.base import BaseProvider


@pytest.fixture
def provider():
    return MagicMock(spec=BaseProvider)


@pytest.fixture
def factory(provider):
    return AgentFactory(provider)


# -------------------------------------------------------------------------
# Construction
# -------------------------------------------------------------------------


class TestAgentFactoryConstruction:
    def test_default_helix_config(self, factory):
        assert factory._config == HelixConfig.default()

    def test_custom_helix_config(self, provider):
        cfg = HelixConfig.research_heavy()
        f = AgentFactory(provider, helix_config=cfg)
        assert f._config == cfg

    def test_provider_stored(self, factory, provider):
        assert factory._provider is provider


# -------------------------------------------------------------------------
# create_agent
# -------------------------------------------------------------------------


class TestCreateAgent:
    def test_create_default_llm_agent(self, factory):
        a = factory.create_agent()
        assert isinstance(a, LLMAgent)

    def test_create_research_agent(self, factory):
        a = factory.create_agent("research")
        assert isinstance(a, ResearchAgent)

    def test_create_analysis_agent(self, factory):
        a = factory.create_agent("analysis")
        assert isinstance(a, AnalysisAgent)

    def test_create_critic_agent(self, factory):
        a = factory.create_agent("critic")
        assert isinstance(a, CriticAgent)

    def test_auto_generated_id(self, factory):
        a = factory.create_agent("research")
        assert a.agent_id == "research-001"

    def test_incremental_ids(self, factory):
        a1 = factory.create_agent()
        a2 = factory.create_agent()
        assert a1.agent_id == "llm-001"
        assert a2.agent_id == "llm-002"

    def test_custom_id(self, factory):
        a = factory.create_agent(agent_id="my-agent")
        assert a.agent_id == "my-agent"

    def test_custom_spawn_time(self, factory):
        a = factory.create_agent(spawn_time=0.5)
        assert a.spawn_time == 0.5

    def test_extra_kwargs_forwarded(self, factory):
        a = factory.create_agent("research", research_domain="creative")
        assert a.research_domain == "creative"

    def test_unknown_type_raises(self, factory):
        with pytest.raises(ValueError, match="Unknown agent type"):
            factory.create_agent("nonexistent")


# -------------------------------------------------------------------------
# create_team
# -------------------------------------------------------------------------


class TestCreateTeam:
    def test_team_size(self, factory):
        team = factory.create_team(team_size=5)
        assert len(team) == 5

    def test_spawn_times_distributed(self, factory):
        team = factory.create_team(team_size=4)
        spawn_times = [a.spawn_time for a in team]
        assert spawn_times == [0.0, 0.25, 0.5, 0.75]

    def test_all_agents_share_helix(self, factory):
        team = factory.create_team(3)
        helices = [a._helix for a in team]
        assert all(h is helices[0] for h in helices)

    def test_team_of_one(self, factory):
        team = factory.create_team(team_size=1)
        assert len(team) == 1
        assert team[0].spawn_time == 0.0

    def test_team_type(self, factory):
        team = factory.create_team(team_size=2, agent_type="research")
        assert all(isinstance(a, ResearchAgent) for a in team)


# -------------------------------------------------------------------------
# create_specialized_team
# -------------------------------------------------------------------------


class TestCreateSpecializedTeam:
    def test_simple_composition(self, factory):
        team = factory.create_specialized_team("simple", seed=42)
        types = [type(a).__name__ for a in team]
        assert "ResearchAgent" in types
        assert "AnalysisAgent" in types
        assert len(team) == 2

    def test_moderate_composition(self, factory):
        team = factory.create_specialized_team("moderate", seed=42)
        types = [type(a).__name__ for a in team]
        assert "ResearchAgent" in types
        assert "AnalysisAgent" in types
        assert "CriticAgent" in types
        assert len(team) == 3

    def test_complex_composition(self, factory):
        team = factory.create_specialized_team("complex", seed=42)
        types = [type(a).__name__ for a in team]
        assert types.count("ResearchAgent") == 2
        assert "AnalysisAgent" in types
        assert "CriticAgent" in types
        assert len(team) == 5

    def test_spawn_times_sorted(self, factory):
        team = factory.create_specialized_team("complex", seed=42)
        spawn_times = [a.spawn_time for a in team]
        assert spawn_times == sorted(spawn_times)

    def test_invalid_complexity_raises(self, factory):
        with pytest.raises(ValueError, match="Unknown complexity"):
            factory.create_specialized_team("impossible")

    def test_deterministic_with_seed(self, factory):
        t1 = factory.create_specialized_team("moderate", seed=123)
        # Reset counter to get same IDs
        factory._agent_counter = 0
        t2 = factory.create_specialized_team("moderate", seed=123)
        times1 = [a.spawn_time for a in t1]
        times2 = [a.spawn_time for a in t2]
        assert times1 == times2


# -------------------------------------------------------------------------
# register_agent_type
# -------------------------------------------------------------------------


class TestRegisterAgentType:
    def test_register_custom_type(self, factory):
        class CustomAgent(LLMAgent):
            pass

        AgentFactory.register_agent_type("custom", CustomAgent)
        a = factory.create_agent("custom")
        assert isinstance(a, CustomAgent)
        # Clean up
        del AgentFactory._agent_types["custom"]

    def test_register_non_llmagent_raises(self):
        class NotAnAgent:
            pass

        with pytest.raises(TypeError, match="must be a subclass of LLMAgent"):
            AgentFactory.register_agent_type("bad", NotAnAgent)  # type: ignore[arg-type]
