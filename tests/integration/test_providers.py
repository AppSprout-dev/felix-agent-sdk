"""Integration tests for provider layer — requires real API keys.

These tests are skipped by default in local development. They run in CI
when the appropriate secrets are configured as environment variables.

Usage:
    # Run with Anthropic key
    ANTHROPIC_API_KEY=sk-ant-... pytest tests/integration/ -v

    # Run with OpenAI key
    OPENAI_API_KEY=sk-... pytest tests/integration/ -v

    # Run local provider tests (requires LM Studio / Ollama running)
    LOCAL_BASE_URL=http://localhost:1234/v1 pytest tests/integration/ -v -k local
"""

from __future__ import annotations

import os

import pytest

from felix_agent_sdk.providers.types import ChatMessage, CompletionResult, MessageRole, StreamChunk

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

HAS_ANTHROPIC_KEY = bool(os.getenv("ANTHROPIC_API_KEY"))
HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))
HAS_LOCAL_SERVER = bool(os.getenv("LOCAL_BASE_URL"))

SIMPLE_MESSAGES = [ChatMessage(role=MessageRole.USER, content="Say hello in exactly one word.")]


# ---------------------------------------------------------------------------
# Anthropic integration tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
class TestAnthropicIntegration:
    @pytest.fixture
    def provider(self):
        from felix_agent_sdk.providers.anthropic import AnthropicProvider
        return AnthropicProvider(model="claude-haiku-4-5-20251001")

    def test_complete(self, provider):
        result = provider.complete(SIMPLE_MESSAGES, max_tokens=10)
        assert isinstance(result, CompletionResult)
        assert len(result.content) > 0
        assert result.usage["total_tokens"] > 0

    def test_stream(self, provider):
        chunks = list(provider.stream(SIMPLE_MESSAGES, max_tokens=10))
        assert len(chunks) > 0
        final = [c for c in chunks if c.is_final]
        assert len(final) == 1
        assert final[0].usage["total_tokens"] > 0

    def test_count_tokens(self, provider):
        count = provider.count_tokens(SIMPLE_MESSAGES)
        assert isinstance(count, int)
        assert count > 0

    def test_validate(self, provider):
        assert provider.validate() is True


# ---------------------------------------------------------------------------
# OpenAI integration tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set")
class TestOpenAIIntegration:
    @pytest.fixture
    def provider(self):
        from felix_agent_sdk.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(model="gpt-4o-mini")

    def test_complete(self, provider):
        result = provider.complete(SIMPLE_MESSAGES, max_tokens=10)
        assert isinstance(result, CompletionResult)
        assert len(result.content) > 0
        assert result.usage["total_tokens"] > 0

    def test_stream(self, provider):
        chunks = list(provider.stream(SIMPLE_MESSAGES, max_tokens=10))
        assert len(chunks) > 0
        final = [c for c in chunks if c.is_final]
        assert len(final) == 1
        assert final[0].usage["total_tokens"] > 0

    def test_count_tokens(self, provider):
        count = provider.count_tokens(SIMPLE_MESSAGES)
        assert isinstance(count, int)
        assert count > 0

    def test_validate(self, provider):
        assert provider.validate() is True


# ---------------------------------------------------------------------------
# Local provider integration tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_LOCAL_SERVER,
    reason="LOCAL_BASE_URL not set (start LM Studio or Ollama first)",
)
class TestLocalIntegration:
    @pytest.fixture
    def provider(self):
        from felix_agent_sdk.providers.local import LocalProvider
        return LocalProvider()

    def test_validate(self, provider):
        assert provider.validate() is True

    def test_complete(self, provider):
        result = provider.complete(SIMPLE_MESSAGES, max_tokens=10)
        assert isinstance(result, CompletionResult)
        assert len(result.content) > 0

    def test_stream(self, provider):
        chunks = list(provider.stream(SIMPLE_MESSAGES, max_tokens=10))
        assert len(chunks) > 0

    def test_count_tokens(self, provider):
        count = provider.count_tokens(SIMPLE_MESSAGES)
        assert isinstance(count, int)
        assert count >= 0
