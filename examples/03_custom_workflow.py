#!/usr/bin/env python3
"""Custom workflow configuration.

Demonstrates building a WorkflowConfig from scratch with specific
helix parameters, team composition, and synthesis strategy. Uses a
mock provider so no API key is needed.

Usage:
    python examples/03_custom_workflow.py
"""

from _mock import make_mock_provider

from felix_agent_sdk import FelixWorkflow, WorkflowConfig
from felix_agent_sdk.core.helix import HelixConfig
from felix_agent_sdk.workflows.config import SynthesisStrategy


def main():
    responses = [
        "Initial research reveals three competing approaches to the problem, "
        "each with distinct trade-offs in scalability and maintainability.",
        "Deep analysis confirms approach B has the best balance of performance "
        "and developer experience based on benchmark data from similar projects.",
        "Critique: approach B's scalability claims need validation under load. "
        "Recommend stress testing before final recommendation.",
        "Second research pass focused on approach B confirms its suitability "
        "with specific caveats around connection pooling at scale.",
        "Final synthesis: approach B is recommended with a phased rollout "
        "strategy and connection pool monitoring as a key operational metric.",
    ]
    provider = make_mock_provider(responses)

    # Custom configuration: wide helix, extra research, higher threshold
    config = WorkflowConfig(
        helix_config=HelixConfig(top_radius=4.0, bottom_radius=0.3, height=10.0, turns=3),
        team_composition=[
            ("research", {}),
            ("research", {}),
            ("analysis", {}),
            ("critic", {}),
        ],
        confidence_threshold=0.85,
        max_rounds=4,
        synthesis_strategy=SynthesisStrategy.COMPRESSED_MERGE,
    )

    workflow = FelixWorkflow(config, provider)
    result = workflow.run("Evaluate database migration strategies for a high-traffic service")

    print("=== Custom Workflow Result ===")
    print(f"Team: {result.metadata['team_composition']}")
    print(f"Rounds: {result.total_rounds}/{config.max_rounds}")
    print(f"Confidence: {result.final_confidence:.3f}")
    print(f"Strategy: {config.synthesis_strategy.value}")
    print(f"\nSynthesis:\n{result.synthesis}")

    # Try BEST_RESULT strategy for comparison
    config_best = WorkflowConfig(
        team_composition=[("research", {}), ("analysis", {})],
        synthesis_strategy=SynthesisStrategy.BEST_RESULT,
        max_rounds=2,
    )
    result_best = FelixWorkflow(config_best, make_mock_provider(responses)).run(
        "Same task, different strategy"
    )
    print("\n--- BEST_RESULT strategy ---")
    print(f"Synthesis: {result_best.synthesis[:100]}...")


if __name__ == "__main__":
    main()
