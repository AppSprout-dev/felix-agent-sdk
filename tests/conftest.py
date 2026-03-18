"""Pytest configuration and shared fixtures for felix-agent-sdk test suite."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from felix_agent_sdk.agents.llm_agent import LLMTask
from felix_agent_sdk.core.helix import HelixConfig, HelixGeometry
from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.providers.types import (
    ChatMessage,
    CompletionResult,
    MessageRole,
    ProviderConfig,
    StreamChunk,
)


# ---------------------------------------------------------------------------
# Common messages used across test modules
# ---------------------------------------------------------------------------


@pytest.fixture
def user_message():
    """A simple user message."""
    return ChatMessage(role=MessageRole.USER, content="Hello, world!")


@pytest.fixture
def system_message():
    """A system message."""
    return ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant.")


@pytest.fixture
def assistant_message():
    """An assistant reply."""
    return ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!")


@pytest.fixture
def conversation(system_message, user_message):
    """System + user conversation sequence."""
    return [system_message, user_message]


# ---------------------------------------------------------------------------
# Provider configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config():
    """A ProviderConfig with sensible defaults."""
    return ProviderConfig(model="test-model", api_key="test-key")


@pytest.fixture
def anthropic_config():
    return ProviderConfig(model="claude-sonnet-4-5", api_key="sk-ant-test")


@pytest.fixture
def openai_config():
    return ProviderConfig(model="gpt-4o", api_key="sk-test")


@pytest.fixture
def local_config():
    return ProviderConfig(
        model="local-model",
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
    )


# ---------------------------------------------------------------------------
# Mock response factories
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_anthropic_response():
    """Factory for Anthropic-style mock responses."""

    def _make(content="Hello!", input_tokens=10, output_tokens=5, stop_reason="end_turn"):
        block = MagicMock()
        block.text = content
        block.type = "text"

        usage = MagicMock()
        usage.input_tokens = input_tokens
        usage.output_tokens = output_tokens

        response = MagicMock()
        response.content = [block]
        response.model = "claude-sonnet-4-5"
        response.usage = usage
        response.stop_reason = stop_reason
        return response

    return _make


@pytest.fixture
def mock_openai_response():
    """Factory for OpenAI-style mock responses."""

    def _make(content="Hello!", prompt_tokens=10, completion_tokens=5, finish_reason="stop"):
        message = MagicMock()
        message.content = content

        choice = MagicMock()
        choice.message = message
        choice.finish_reason = finish_reason

        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        usage.total_tokens = prompt_tokens + completion_tokens

        response = MagicMock()
        response.choices = [choice]
        response.model = "gpt-4o"
        response.usage = usage
        return response

    return _make


# ---------------------------------------------------------------------------
# Helix / Agent fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_helix():
    """Default HelixGeometry (top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)."""
    return HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)


@pytest.fixture
def default_helix_config():
    return HelixConfig.default()


@pytest.fixture
def mock_provider():
    """A BaseProvider mock that returns predictable CompletionResults."""
    provider = MagicMock(spec=BaseProvider)
    provider.complete.return_value = CompletionResult(
        content=(
            "The analysis reveals several key findings. First, the data indicates "
            "a strong correlation between renewable energy adoption and grid stability. "
            "Furthermore, evidence from multiple studies demonstrates that distributed "
            "generation improves resilience. However, intermittency challenges remain "
            "a specific concern requiring detailed attention."
        ),
        model="test-model",
        usage={"prompt_tokens": 100, "completion_tokens": 60, "total_tokens": 160},
    )
    provider.count_tokens.return_value = 100
    return provider


@pytest.fixture
def sample_task():
    """A sample LLMTask for agent tests."""
    return LLMTask(
        task_id="task-001",
        description="Analyse the impact of renewable energy on grid stability.",
    )
