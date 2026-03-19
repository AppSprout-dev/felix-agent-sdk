#!/usr/bin/env python3
"""Hello World — minimal Felix workflow with a mock provider.

This example runs entirely offline (no API key needed) by using a mock
provider. It demonstrates the core workflow loop: team creation,
helix-driven processing rounds, and synthesis.

Usage:
    python examples/01_hello_world.py
"""

from unittest.mock import MagicMock

from felix_agent_sdk import WorkflowConfig, run_felix_workflow
from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.providers.types import CompletionResult


def make_mock_provider() -> BaseProvider:
    """Create a mock provider that returns plausible content."""
    provider = MagicMock(spec=BaseProvider)
    call_count = [0]
    responses = [
        "Research indicates renewable energy adoption has accelerated globally, "
        "with solar capacity growing 40% year-over-year in key markets.",
        "Analysis of the data reveals a strong correlation between distributed "
        "generation and improved grid resilience across multiple regions.",
        "Critical review confirms the findings are well-supported. However, "
        "intermittency challenges require further attention to storage solutions.",
        "Synthesising all findings: the evidence strongly supports accelerating "
        "renewable energy deployment, with storage as the key enabling technology.",
    ]

    def _complete(messages, **kwargs):
        idx = call_count[0] % len(responses)
        call_count[0] += 1
        return CompletionResult(
            content=responses[idx],
            model="mock-model",
            usage={"prompt_tokens": 50, "completion_tokens": 40, "total_tokens": 90},
        )

    provider.complete.side_effect = _complete
    provider.count_tokens.return_value = 50
    return provider


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
