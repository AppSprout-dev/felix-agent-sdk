"""Tests for DynamicSpawner integration."""

from __future__ import annotations

from unittest.mock import MagicMock

from felix_agent_sdk.agents.factory import AgentFactory
from felix_agent_sdk.agents.llm_agent import LLMResult
from felix_agent_sdk.communication.central_post import CentralPost
from felix_agent_sdk.communication.spoke import SpokeManager
from felix_agent_sdk.core.helix import HelixConfig
from felix_agent_sdk.events import EventBus, EventType
from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.providers.types import CompletionResult
from felix_agent_sdk.spawning import ConfidenceMonitor, DynamicSpawner
from felix_agent_sdk.workflows.config import WorkflowConfig
from felix_agent_sdk import run_felix_workflow


def _mock_provider() -> BaseProvider:
    provider = MagicMock(spec=BaseProvider)
    provider.complete.return_value = CompletionResult(
        content="Low confidence filler text.",
        model="mock",
        usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    )
    provider.count_tokens.return_value = 20
    return provider


def _make_low_confidence_result(agent_id: str = "agent-001") -> LLMResult:
    return LLMResult(
        agent_id=agent_id,
        task_id="t1",
        content="Uncertain findings with weak evidence.",
        position_info={"phase": "exploration", "agent_type": "research"},
        completion_result=CompletionResult(
            content="text", model="mock",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        ),
        processing_time=0.1,
        confidence=0.3,
        temperature_used=0.8,
        token_budget_used=20,
    )


class TestDynamicSpawner:
    def _setup(self, max_spawned: int = 3, event_bus: EventBus | None = None):
        provider = _mock_provider()
        hub = CentralPost(max_agents=20)
        spoke_mgr = SpokeManager(hub=hub)
        factory = AgentFactory(provider, helix_config=HelixConfig.default())
        monitor = ConfidenceMonitor(threshold=0.8)
        spawner = DynamicSpawner(
            factory=factory,
            spoke_mgr=spoke_mgr,
            monitor=monitor,
            max_spawned=max_spawned,
            event_bus=event_bus,
        )
        return spawner, spoke_mgr, hub

    def test_spawns_when_confidence_low(self):
        spawner, _, hub = self._setup()
        results = [_make_low_confidence_result("a1"), _make_low_confidence_result("a2")]
        # Large gap to threshold triggers spawn on first round
        new1 = spawner.check_and_spawn([], results, 0.3)
        assert len(new1) == 1
        new2 = spawner.check_and_spawn([], results, 0.5)
        assert len(new2) == 1
        assert spawner.total_spawned == 2

    def test_no_spawn_when_empty_results(self):
        spawner, _, _ = self._setup()
        new = spawner.check_and_spawn([], [], 0.3)
        assert len(new) == 0

    def test_respects_max_spawned(self):
        spawner, _, _ = self._setup(max_spawned=1)
        results = [_make_low_confidence_result()]
        # First round seeds the monitor
        spawner.check_and_spawn([], results, 0.3)
        # Second round triggers spawn
        spawner.check_and_spawn([], results, 0.5)
        assert spawner.total_spawned == 1
        # Third round — already at max
        new = spawner.check_and_spawn([], results, 0.7)
        assert len(new) == 0
        assert spawner.total_spawned == 1

    def test_emits_events(self):
        bus = EventBus()
        bus.enable_history()
        spawner, _, _ = self._setup(event_bus=bus)

        results = [_make_low_confidence_result()]
        spawner.check_and_spawn([], results, 0.3)
        spawner.check_and_spawn([], results, 0.5)

        types = [e.event_type for e in bus.history]
        assert "spawn.triggered" in types
        assert "spawn.completed" in types

    def test_spawn_completed_has_agent_id(self):
        bus = EventBus()
        bus.enable_history()
        spawner, _, _ = self._setup(event_bus=bus)

        results = [_make_low_confidence_result()]
        spawner.check_and_spawn([], results, 0.3)

        completed = [e for e in bus.history if e.event_type == "spawn.completed"]
        assert len(completed) == 1
        assert "agent_id" in completed[0].data
        assert "agent_type" in completed[0].data


class TestDynamicSpawningWorkflowIntegration:
    """Test that enable_dynamic_spawning works end-to-end in a workflow."""

    def test_workflow_with_dynamic_spawning(self):
        bus = EventBus()
        bus.enable_history()

        provider = _mock_provider()
        config = WorkflowConfig(
            max_rounds=3,
            confidence_threshold=0.99,  # unreachable → forces all rounds
            enable_dynamic_spawning=True,
            max_dynamic_agents=2,
        )

        result = run_felix_workflow(config, provider, "Test task", event_bus=bus)

        # Workflow should complete
        assert result.synthesis is not None
        assert result.total_rounds == 3

        # Check if spawn events were emitted (may or may not trigger
        # depending on confidence, but the machinery ran without error)
        types = [e.event_type for e in bus.history]
        assert "workflow.started" in types
        assert "workflow.completed" in types

    def test_workflow_without_dynamic_spawning(self):
        """Default config should not spawn — no errors."""
        provider = _mock_provider()
        config = WorkflowConfig(max_rounds=1)
        result = run_felix_workflow(config, provider, "Test task")
        assert result.synthesis is not None
