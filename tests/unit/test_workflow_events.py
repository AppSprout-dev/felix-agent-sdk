"""Tests verifying event emission from FelixWorkflow."""

from __future__ import annotations

from unittest.mock import MagicMock

from felix_agent_sdk import WorkflowConfig, run_felix_workflow
from felix_agent_sdk.events import EventBus, EventType, FelixEvent
from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.providers.types import CompletionResult


def _make_mock_provider() -> BaseProvider:
    """Provider that returns canned responses."""
    provider = MagicMock(spec=BaseProvider)
    counter = [0]
    responses = [
        "Research finding about renewable energy.",
        "Analysis of market trends and data.",
        "Critique: findings are sound but need more evidence.",
        "Synthesis of all findings into a coherent report.",
    ]

    def _complete(messages, **kwargs):
        idx = counter[0] % len(responses)
        counter[0] += 1
        return CompletionResult(
            content=responses[idx],
            model="mock",
            usage={"prompt_tokens": 40, "completion_tokens": 30, "total_tokens": 70},
        )

    provider.complete.side_effect = _complete
    provider.count_tokens.return_value = 40
    return provider


class TestWorkflowEventSequence:
    """Verify the exact sequence of events emitted during a workflow run."""

    def test_full_event_sequence(self):
        bus = EventBus()
        bus.enable_history()

        provider = _make_mock_provider()
        config = WorkflowConfig(max_rounds=2)

        run_felix_workflow(
            config, provider, "Test task", event_bus=bus
        )

        events = bus.history
        types = [e.event_type for e in events]

        # Must start with workflow.started
        assert types[0] == EventType.WORKFLOW_STARTED

        # Must end with workflow.completed
        assert types[-1] == EventType.WORKFLOW_COMPLETED

        # Must contain round starts and completions
        assert EventType.WORKFLOW_ROUND_STARTED in types
        assert EventType.WORKFLOW_ROUND_COMPLETED in types

        # Must contain task processing events
        assert EventType.TASK_STARTED in types
        assert EventType.TASK_COMPLETED in types

        # Must contain synthesis
        assert EventType.WORKFLOW_SYNTHESIS_STARTED in types

    def test_round_events_bracket_tasks(self):
        """Round start/end should bracket the task events within that round."""
        bus = EventBus()
        bus.enable_history()

        provider = _make_mock_provider()
        config = WorkflowConfig(max_rounds=1)

        run_felix_workflow(config, provider, "Test", event_bus=bus)

        events = bus.history
        types = [e.event_type for e in events]

        round_start_idx = types.index(EventType.WORKFLOW_ROUND_STARTED)
        round_end_idx = types.index(EventType.WORKFLOW_ROUND_COMPLETED)
        task_start_idx = types.index(EventType.TASK_STARTED)
        task_end_idx = types.index(EventType.TASK_COMPLETED)

        assert round_start_idx < task_start_idx
        assert task_end_idx < round_end_idx

    def test_workflow_started_has_metadata(self):
        bus = EventBus()
        bus.enable_history()

        config = WorkflowConfig(max_rounds=1)
        run_felix_workflow(config, _make_mock_provider(), "My task", event_bus=bus)

        started = [e for e in bus.history if e.event_type == EventType.WORKFLOW_STARTED]
        assert len(started) == 1
        assert started[0].data["task"] == "My task"
        assert started[0].data["max_rounds"] == 1
        assert started[0].data["agents_count"] == 3  # default team

    def test_workflow_completed_has_metadata(self):
        bus = EventBus()
        bus.enable_history()

        config = WorkflowConfig(max_rounds=1)
        run_felix_workflow(config, _make_mock_provider(), "Task", event_bus=bus)

        completed = [e for e in bus.history if e.event_type == EventType.WORKFLOW_COMPLETED]
        assert len(completed) == 1
        assert "rounds" in completed[0].data
        assert "final_confidence" in completed[0].data
        assert "total_tokens" in completed[0].data
        assert "elapsed_seconds" in completed[0].data

    def test_task_completed_has_agent_details(self):
        bus = EventBus()
        bus.enable_history()

        config = WorkflowConfig(max_rounds=1)
        run_felix_workflow(config, _make_mock_provider(), "Task", event_bus=bus)

        task_events = [e for e in bus.history if e.event_type == EventType.TASK_COMPLETED]
        assert len(task_events) > 0

        first = task_events[0]
        assert "task_id" in first.data
        assert "agent_type" in first.data
        assert "confidence" in first.data
        assert "temperature" in first.data
        assert "tokens" in first.data

    def test_no_bus_no_crash(self):
        """Workflow works fine without an event bus."""
        config = WorkflowConfig(max_rounds=1)
        result = run_felix_workflow(config, _make_mock_provider(), "Task")
        assert result.synthesis is not None

    def test_subscriber_receives_events_live(self):
        """Verify that subscribers are called during the workflow, not after."""
        bus = EventBus()
        received_during = []

        def on_round_start(event: FelixEvent):
            received_during.append(event.event_type)

        bus.subscribe(EventType.WORKFLOW_ROUND_STARTED, on_round_start)

        config = WorkflowConfig(max_rounds=2)
        run_felix_workflow(config, _make_mock_provider(), "Task", event_bus=bus)

        assert len(received_during) == 2

    def test_event_count_scales_with_rounds(self):
        """More rounds = more events."""
        bus1 = EventBus()
        bus1.enable_history()
        bus2 = EventBus()
        bus2.enable_history()

        run_felix_workflow(
            WorkflowConfig(max_rounds=1), _make_mock_provider(), "T", event_bus=bus1
        )
        run_felix_workflow(
            WorkflowConfig(max_rounds=3), _make_mock_provider(), "T", event_bus=bus2
        )

        assert len(bus2.history) > len(bus1.history)
