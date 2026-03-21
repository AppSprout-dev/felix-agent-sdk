#!/usr/bin/env python3
"""Dynamic spawning — run a workflow with confidence-driven agent creation.

Demonstrates enable_dynamic_spawning in WorkflowConfig. When confidence
is low, the system automatically spawns additional agents. Uses event
bus to observe spawn decisions.

Usage:
    python examples/08_dynamic_spawning.py
"""

from _mock import make_mock_provider

from felix_agent_sdk import EventBus, EventType, WorkflowConfig, run_felix_workflow


def main():
    bus = EventBus()

    # Watch for spawn events
    def on_spawn(event):
        print(f"  >>> SPAWN: {event.data.get('agent_type', '?')} "
              f"agent {event.data.get('agent_id', '?')} "
              f"(reason: {event.data.get('reason', '?')})")

    def on_spawn_done(event):
        print(f"  >>> SPAWNED: {event.data.get('agent_id', '?')} "
              f"(total: {event.data.get('total_spawned', '?')})")

    bus.subscribe(EventType.SPAWN_TRIGGERED, on_spawn)
    bus.subscribe(EventType.SPAWN_COMPLETED, on_spawn_done)
    bus.subscribe(EventType.WORKFLOW_ROUND_COMPLETED, lambda e:
        print(f"  Round {e.data['round']} done — avg confidence: {e.data['avg_confidence']:.3f}"))

    # Low-confidence responses to trigger spawning
    responses = [
        "Preliminary findings are inconclusive.",
        "Weak evidence suggests possible correlation.",
        "Need more data to confirm hypothesis.",
        "Results remain uncertain after review.",
    ]
    provider = make_mock_provider(responses)

    config = WorkflowConfig(
        max_rounds=3,
        confidence_threshold=0.99,  # unreachable — forces all rounds
        enable_dynamic_spawning=True,
        max_dynamic_agents=2,
    )

    print("=== Dynamic Spawning Demo ===\n")
    result = run_felix_workflow(config, provider, "Investigate uncertain hypothesis", event_bus=bus)

    print(f"\nFinal team size: {result.metadata['agents_count']}")
    print(f"Rounds: {result.total_rounds}")
    print(f"Confidence: {result.final_confidence:.3f}")
    print(f"Synthesis: {result.synthesis[:100]}...")


if __name__ == "__main__":
    main()
