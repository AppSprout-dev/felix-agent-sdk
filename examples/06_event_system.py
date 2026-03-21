#!/usr/bin/env python3
"""Event system — subscribe to workflow events and print a timeline.

Demonstrates the EventBus, EventType, and event subscriptions added
in Phase 2. Runs entirely offline with a mock provider.

Usage:
    python examples/06_event_system.py
"""

from _mock import make_mock_provider

from felix_agent_sdk import EventBus, EventType, WorkflowConfig, run_felix_workflow


def main():
    bus = EventBus()

    # Subscribe to specific event types
    def on_round(event):
        print(f"  [ROUND] {event.data}")

    def on_task(event):
        agent = event.data.get("agent_type", "?")
        conf = event.data.get("confidence", "?")
        print(f"  [TASK]  {event.source} ({agent}) confidence={conf}")

    bus.subscribe(EventType.WORKFLOW_STARTED, lambda e: print(f">>> Workflow started: {e.data['task'][:60]}"))
    bus.subscribe("workflow.round.*", on_round)
    bus.subscribe(EventType.TASK_COMPLETED, on_task)
    bus.subscribe(EventType.WORKFLOW_CONVERGED, lambda e: print(f">>> Converged at round {e.data['round']}!"))
    bus.subscribe(EventType.WORKFLOW_COMPLETED, lambda e: print(f">>> Workflow done in {e.data['elapsed_seconds']}s"))

    provider = make_mock_provider()
    config = WorkflowConfig(max_rounds=2)

    result = run_felix_workflow(config, provider, "Evaluate renewable energy trends", event_bus=bus)
    print(f"\nFinal synthesis: {result.synthesis[:100]}...")


if __name__ == "__main__":
    main()
