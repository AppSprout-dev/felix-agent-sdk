#!/usr/bin/env python3
"""Structured logging — run a workflow with JSON log output.

Demonstrates configure_logging() with JSON format and EventLogBridge
for automatic event-to-log bridging.

Usage:
    python examples/09_structured_logging.py
"""


from _mock import make_mock_provider

from felix_agent_sdk import EventBus, WorkflowConfig, configure_logging, run_felix_workflow
from felix_agent_sdk.utils.logging import EventLogBridge, FelixLogConfig


def main():
    # Configure JSON logging at INFO level (use DEBUG for more detail)
    configure_logging(FelixLogConfig(
        level="INFO",
        format="json",
    ))

    # Create event bus and bridge events to the logger
    bus = EventBus()
    bridge = EventLogBridge(bus)

    provider = make_mock_provider()
    config = WorkflowConfig(max_rounds=1)

    print("=== Running with JSON structured logging ===\n")
    result = run_felix_workflow(config, provider, "Test structured logging", event_bus=bus)

    print(f"\nSynthesis: {result.synthesis[:80]}...")

    # Clean up
    bridge.detach()

    # Show that text format also works
    print("\n=== Switching to text format ===\n")
    configure_logging(FelixLogConfig(level="INFO", format="text"))

    bus2 = EventBus()
    bridge2 = EventLogBridge(bus2)

    run_felix_workflow(config, make_mock_provider(), "Test text logging", event_bus=bus2)
    bridge2.detach()


if __name__ == "__main__":
    main()
