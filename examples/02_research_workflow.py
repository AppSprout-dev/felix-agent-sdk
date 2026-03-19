#!/usr/bin/env python3
"""Research workflow using a template.

Uses the research template (2 Research + 1 Analysis + 1 Critic) with
a real LLM provider. Requires an API key set via environment variable.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/02_research_workflow.py

    # Or with Anthropic:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/02_research_workflow.py --provider anthropic
"""

import argparse
import sys

from felix_agent_sdk import FelixWorkflow
from felix_agent_sdk.workflows.templates import research_config


def get_provider(provider_name: str):
    """Create a provider based on name. Returns None if deps missing."""
    if provider_name == "openai":
        try:
            from felix_agent_sdk.providers import OpenAIProvider

            return OpenAIProvider(model="gpt-4o-mini")
        except Exception as e:
            print(f"Could not create OpenAI provider: {e}")
            return None
    elif provider_name == "anthropic":
        try:
            from felix_agent_sdk.providers import AnthropicProvider

            return AnthropicProvider(model="claude-sonnet-4-5")
        except Exception as e:
            print(f"Could not create Anthropic provider: {e}")
            return None
    else:
        print(f"Unknown provider: {provider_name}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run a Felix research workflow")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    parser.add_argument(
        "--task",
        default="What are the key challenges and opportunities in quantum computing for cryptography?",
    )
    args = parser.parse_args()

    provider = get_provider(args.provider)
    if provider is None:
        print("No provider available. Set your API key and try again.")
        sys.exit(1)

    config = research_config(max_rounds=2)
    workflow = FelixWorkflow(config, provider)

    print(f"Running research workflow with {args.provider}...")
    print(f"Task: {args.task}\n")

    result = workflow.run(args.task)

    print("=== Research Workflow Result ===")
    print(f"Rounds: {result.total_rounds}")
    print(f"Agent results: {len(result.agent_results)}")
    print(f"Confidence: {result.final_confidence:.3f}")
    print(f"Converged: {result.metadata['converged']}")
    print(f"Tokens used: {result.metadata['total_tokens']}")
    print(f"\nSynthesis:\n{result.synthesis}")


if __name__ == "__main__":
    main()
