"""Tests for LLMAgent.process_task_streaming()."""

from __future__ import annotations

from unittest.mock import MagicMock

from felix_agent_sdk.agents.llm_agent import LLMAgent, LLMTask
from felix_agent_sdk.core.helix import HelixConfig, HelixGeometry
from felix_agent_sdk.events import EventBus
from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.providers.types import CompletionResult, StreamChunk
from felix_agent_sdk.streaming import CallbackStreamHandler, StreamHandler


def _make_helix() -> HelixGeometry:
    cfg = HelixConfig.default()
    return HelixGeometry(cfg.top_radius, cfg.bottom_radius, cfg.height, cfg.turns)


def _make_streaming_provider(text: str = "Hello streaming world!") -> BaseProvider:
    """Mock provider whose stream() yields word-by-word chunks."""
    provider = MagicMock(spec=BaseProvider)

    words = text.split(" ")
    chunks = []
    for i, word in enumerate(words):
        is_last = i == len(words) - 1
        suffix = "" if is_last else " "
        chunks.append(StreamChunk(
            text=word + suffix,
            is_final=is_last,
            usage={"prompt_tokens": 10, "completion_tokens": len(words), "total_tokens": 10 + len(words)} if is_last else {},
        ))

    provider.stream.return_value = iter(chunks)
    # Also set up complete() for comparison
    provider.complete.return_value = CompletionResult(
        content=text,
        model="mock",
        usage={"prompt_tokens": 10, "completion_tokens": len(words), "total_tokens": 10 + len(words)},
    )
    provider.count_tokens.return_value = 10
    return provider


class TestProcessTaskStreaming:
    def test_produces_same_content_as_process_task(self):
        text = "Research finds renewable energy growing rapidly."
        helix = _make_helix()

        # Streaming path
        provider_s = _make_streaming_provider(text)
        agent_s = LLMAgent("s-001", provider_s, helix, agent_type="research")
        agent_s.spawn(0.1)
        agent_s.update_position(0.5)
        task = LLMTask(task_id="t1", description="Test task")
        result_s = agent_s.process_task_streaming(task, StreamHandler())

        # Non-streaming path
        provider_c = _make_streaming_provider(text)
        agent_c = LLMAgent("c-001", provider_c, helix, agent_type="research")
        agent_c.spawn(0.1)
        agent_c.update_position(0.5)
        result_c = agent_c.process_task(LLMTask(task_id="t1", description="Test task"))

        assert result_s.content == result_c.content
        assert result_s.content == text

    def test_handler_receives_tokens(self):
        tokens = []
        handler = CallbackStreamHandler(on_token=lambda e: tokens.append(e.content))

        provider = _make_streaming_provider("one two three")
        helix = _make_helix()
        agent = LLMAgent("a-001", provider, helix, agent_type="research")
        agent.spawn(0.1)
        agent.update_position(0.5)

        result = agent.process_task_streaming(
            LLMTask(task_id="t1", description="Test"), handler
        )

        assert tokens == ["one ", "two "]  # "three" is final → dispatched as result
        assert result.content == "one two three"

    def test_handler_receives_final_result(self):
        results = []
        handler = CallbackStreamHandler(on_result=lambda e: results.append(e))

        provider = _make_streaming_provider("hello world")
        helix = _make_helix()
        agent = LLMAgent("a-001", provider, helix, agent_type="analysis")
        agent.spawn(0.1)
        agent.update_position(0.5)

        agent.process_task_streaming(
            LLMTask(task_id="t1", description="Test"), handler
        )

        assert len(results) == 1
        assert results[0].is_final is True
        assert results[0].accumulated == "hello world"

    def test_updates_agent_state(self):
        provider = _make_streaming_provider("test content")
        helix = _make_helix()
        agent = LLMAgent("a-001", provider, helix, agent_type="research")
        agent.spawn(0.1)
        agent.update_position(0.5)

        assert agent.total_tokens_used == 0
        assert agent.total_processing_time == 0.0
        assert len(agent.processing_results) == 0

        agent.process_task_streaming(
            LLMTask(task_id="t1", description="Test"), StreamHandler()
        )

        assert agent.total_tokens_used > 0
        assert agent.total_processing_time > 0.0
        assert len(agent.processing_results) == 1

    def test_emits_events(self):
        bus = EventBus()
        bus.enable_history()

        provider = _make_streaming_provider("data")
        helix = _make_helix()
        agent = LLMAgent("a-001", provider, helix, agent_type="research", event_bus=bus)
        agent.spawn(0.1)
        agent.update_position(0.5)

        agent.process_task_streaming(
            LLMTask(task_id="t1", description="Test"), StreamHandler()
        )

        types = [e.event_type for e in bus.history]
        assert "task.started" in types
        assert "task.completed" in types

        # Verify streaming flag in event data
        started = [e for e in bus.history if e.event_type == "task.started"][0]
        assert started.data["streaming"] is True

    def test_confidence_recorded(self):
        provider = _make_streaming_provider("a solid analysis with clear evidence")
        helix = _make_helix()
        agent = LLMAgent("a-001", provider, helix, agent_type="analysis")
        agent.spawn(0.1)
        agent.update_position(0.5)

        result = agent.process_task_streaming(
            LLMTask(task_id="t1", description="Test"), StreamHandler()
        )

        assert 0.0 < result.confidence <= 1.0
        assert len(agent._confidence_history) == 1
