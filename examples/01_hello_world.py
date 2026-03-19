#!/usr/bin/env python3
"""Hello World — minimal Felix workflow with a mock provider.

This example runs entirely offline (no API key needed) by using a mock
provider. It demonstrates the core workflow loop: team creation,
helix-driven processing rounds, and synthesis.

Usage:
    python examples/01_hello_world.py
"""

from _mock import make_mock_provider

from felix_agent_sdk import WorkflowConfig, run_felix_workflow


def main():
    provider = make_mock_provider()
    config = WorkflowConfig(max_rounds=2)

    result = run_felix_workflow(
        config, provider, "Evaluate the impact of renewable energy on grid stability"
    )

    print("=== Felix Workflow Result ===")
    print(f"Rounds completed: {result.total_rounds}")
    print(f"Agents used: {result.metadata['agents_count']}")
    print(f"Confidence: {result.final_confidence:.3f}")
    print(f"Converged: {result.metadata['converged']}")
    print(f"\nSynthesis:\n{result.synthesis}")


if __name__ == "__main__":
    main()
