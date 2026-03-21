#!/usr/bin/env python3
"""Streaming output — process a single agent task with token-level streaming.

Demonstrates StreamHandler, CallbackStreamHandler, and
LLMAgent.process_task_streaming(). Runs offline with a mock provider.

Usage:
    python examples/07_streaming_output.py
"""

from unittest.mock import MagicMock

from felix_agent_sdk import LLMAgent, LLMTask
from felix_agent_sdk.core.helix import HelixConfig, HelixGeometry
from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.providers.types import CompletionResult, StreamChunk
from felix_agent_sdk.streaming import CallbackStreamHandler


def _make_streaming_provider():
    """Mock provider whose stream() yields word-by-word chunks."""
    text = (
        "Solar energy deployment has reached unprecedented levels globally. "
        "Key markets show 40% year-over-year capacity growth, driven by "
        "declining panel costs and supportive policy frameworks. Grid-scale "
        "battery storage is emerging as the critical enabler for intermittency "
        "management, with lithium-ion costs declining 85% since 2010."
    )
    words = text.split(" ")
    provider = MagicMock(spec=BaseProvider)

    def _stream(messages, **kwargs):
        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            yield StreamChunk(
                text=word + ("" if is_last else " "),
                is_final=is_last,
                usage={"prompt_tokens": 30, "completion_tokens": len(words), "total_tokens": 30 + len(words)} if is_last else {},
            )

    provider.stream.side_effect = _stream
    provider.complete.return_value = CompletionResult(
        content=text, model="mock",
        usage={"prompt_tokens": 30, "completion_tokens": len(words), "total_tokens": 30 + len(words)},
    )
    provider.count_tokens.return_value = 30
    return provider


def main():
    provider = _make_streaming_provider()
    cfg = HelixConfig.default()
    helix = HelixGeometry(cfg.top_radius, cfg.bottom_radius, cfg.height, cfg.turns)
    agent = LLMAgent("research-001", provider, helix, agent_type="research")
    agent.spawn(0.1)
    agent.update_position(0.5)

    print("=== Streaming Agent Output ===\n")

    token_count = [0]

    def on_token(event):
        token_count[0] += 1
        print(event.content, end="", flush=True)

    def on_result(event):
        print("\n\n--- Stream complete ---")
        print(f"Tokens: {event.token_index}")
        print(f"Length: {len(event.accumulated)} chars")

    handler = CallbackStreamHandler(on_token=on_token, on_result=on_result)

    task = LLMTask(task_id="demo-1", description="Analyse renewable energy trends")
    result = agent.process_task_streaming(task, handler)

    print(f"Confidence: {result.confidence:.3f}")
    print(f"Temperature: {result.temperature_used:.3f}")
    print(f"Phase: {result.position_info.get('phase', 'unknown')}")


if __name__ == "__main__":
    main()
