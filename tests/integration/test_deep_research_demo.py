"""Smoke test for the deep research demo.

Runs the demo in --fast mode and verifies it completes without errors,
produces agent results, and generates a synthesis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add the example directory so sibling imports (_mock_research, helix_visualizer) resolve.
_example_dir = str(
    Path(__file__).resolve().parent.parent.parent / "examples" / "05_deep_research_live"
)
if _example_dir not in sys.path:
    sys.path.insert(0, _example_dir)

from _mock_research import make_research_mock_provider  # noqa: E402
from helix_visualizer import AgentSnapshot, HelixVisualizer  # noqa: E402

from felix_agent_sdk.agents.base import AgentState  # noqa: E402
from felix_agent_sdk.agents.factory import AgentFactory  # noqa: E402
from felix_agent_sdk.agents.llm_agent import LLMAgent, LLMResult, LLMTask  # noqa: E402
from felix_agent_sdk.communication.central_post import CentralPost  # noqa: E402
from felix_agent_sdk.communication.spoke import SpokeManager  # noqa: E402
from felix_agent_sdk.core.helix import HelixConfig  # noqa: E402
from felix_agent_sdk.workflows.config import SynthesisStrategy, WorkflowConfig  # noqa: E402
from felix_agent_sdk.workflows.context_builder import CollaborativeContextBuilder  # noqa: E402
from felix_agent_sdk.workflows.synthesizer import WorkflowSynthesizer  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def research_config() -> WorkflowConfig:
    return WorkflowConfig(
        helix_config=HelixConfig.research_heavy(),
        team_composition=[
            ("research", {}),
            ("research", {}),
            ("analysis", {}),
            ("critic", {}),
        ],
        confidence_threshold=0.78,
        max_rounds=4,
        synthesis_strategy=SynthesisStrategy.COMPRESSED_MERGE,
        max_agents=10,
    )


@pytest.fixture()
def mock_provider():
    return make_research_mock_provider()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeepResearchDemo:
    """End-to-end smoke tests for the demo workflow."""

    def test_mock_provider_returns_phase_aware_responses(self, mock_provider):
        """Mock provider returns different content based on agent type and phase."""
        from felix_agent_sdk.providers.types import ChatMessage, MessageRole

        msgs_research = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a research agent (exploration)"),
            ChatMessage(role=MessageRole.USER, content="Analyse AI frameworks"),
        ]
        msgs_critic = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a critic agent (synthesis)"),
            ChatMessage(role=MessageRole.USER, content="Analyse AI frameworks"),
        ]

        result_r = mock_provider.complete(msgs_research)
        result_c = mock_provider.complete(msgs_critic)

        assert result_r.content != result_c.content
        assert len(result_r.content) > 50
        assert len(result_c.content) > 50

    def test_full_workflow_completes(self, research_config, mock_provider):
        """The full orchestration loop runs to completion and produces results."""
        config = research_config
        factory = AgentFactory(mock_provider, helix_config=config.helix_config)
        hub = CentralPost(max_agents=config.max_agents)
        spoke_mgr = SpokeManager(hub=hub)
        builder = CollaborativeContextBuilder()
        all_results: list[LLMResult] = []

        # Create team
        agents: list[LLMAgent] = []
        for i, (agent_type, kwargs) in enumerate(config.team_composition):
            spawn_time = i / max(len(config.team_composition), 1)
            agent = factory.create_agent(agent_type=agent_type, spawn_time=spawn_time, **kwargs)
            spoke_mgr.create_spoke(agent.agent_id, agent=agent)
            agents.append(agent)

        assert len(agents) == 4

        # Run rounds
        try:
            for round_num in range(1, config.max_rounds + 1):
                current_time = round_num / max(config.max_rounds, 1)

                for agent in agents:
                    if agent.state == AgentState.WAITING and agent.can_spawn(current_time):
                        agent.spawn(current_time)
                    if agent.state != AgentState.ACTIVE:
                        continue

                    agent.update_position(current_time)
                    if not agent.should_process_at_checkpoint():
                        continue

                    task = LLMTask(
                        task_id=f"{agent.agent_id}-r{round_num}",
                        description="Analyse the state of AI agent frameworks",
                        context="",
                        context_history=builder.get_context_history(),
                    )
                    result = agent.process_task(task)
                    agent.mark_checkpoint_processed()
                    builder.add_from_result(result)
                    all_results.append(result)

                spoke_mgr.process_all_messages()
        finally:
            spoke_mgr.shutdown_all()
            hub.shutdown()

        assert len(all_results) > 0
        assert all(r.confidence > 0 for r in all_results)
        assert all(len(r.content) > 0 for r in all_results)

    def test_synthesis_produces_output(self, research_config, mock_provider):
        """WorkflowSynthesizer produces a non-empty string from mock results."""
        from felix_agent_sdk.providers.types import CompletionResult

        # Minimal fake results
        results = [
            LLMResult(
                agent_id="test-001",
                task_id="t1",
                content="Research finding about AI agent frameworks.",
                position_info={"phase": "exploration", "progress": 0.2},
                completion_result=CompletionResult(
                    content="Research finding about AI agent frameworks.",
                    model="mock",
                    usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
                ),
                processing_time=0.1,
                confidence=0.5,
                temperature_used=0.7,
                token_budget_used=70,
            ),
        ]

        synthesizer = WorkflowSynthesizer(mock_provider, research_config)
        synthesis = synthesizer.synthesize(results, "AI agent frameworks")
        assert len(synthesis) > 0

    def test_visualizer_renders_frame(self):
        """HelixVisualizer.render_frame returns a non-empty string with agent data."""
        helix = HelixConfig.research_heavy().to_geometry()
        viz = HelixVisualizer(helix)

        agents = [
            AgentSnapshot(
                agent_id="research-001",
                agent_type="research",
                progress=0.25,
                confidence=0.45,
                temperature=0.7,
                phase="exploration",
                content_preview="Exploring AI frameworks...",
            ),
            AgentSnapshot(
                agent_id="critic-001",
                agent_type="critic",
                progress=0.6,
                confidence=0.55,
                temperature=0.4,
                phase="analysis",
            ),
        ]

        frame = viz.render_frame(agents, round_num=2, max_rounds=4, status_line="Testing...")

        assert "F E L I X" in frame
        assert "EXPLORE" in frame
        assert "ANALYSE" in frame
        assert "SYNTHESISE" in frame
        assert "research-001" in frame
        assert "critic-001" in frame
        assert "Testing..." in frame
