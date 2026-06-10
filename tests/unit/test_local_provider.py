"""Tests for LocalProvider — inherits from OpenAIProvider with local defaults."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from felix_agent_sdk.providers.local import LocalProvider
from felix_agent_sdk.providers.openai_provider import OpenAIProvider
from felix_agent_sdk.providers.types import ChatMessage, MessageRole


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestLocalProviderInheritance:
    def test_inherits_from_openai(self):
        assert issubclass(LocalProvider, OpenAIProvider)

    def test_isinstance_of_openai(self):
        p = LocalProvider()
        assert isinstance(p, OpenAIProvider)


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------


class TestLocalProviderDefaults:
    def test_default_model(self):
        p = LocalProvider()
        assert p.model == "local-model"

    def test_default_base_url(self):
        p = LocalProvider()
        assert p.config.base_url == "http://localhost:1234/v1"

    def test_default_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            p = LocalProvider()
            assert p.config.api_key == "lm-studio"

    def test_custom_model(self):
        p = LocalProvider(model="llama-3.2")
        assert p.model == "llama-3.2"

    def test_custom_base_url(self):
        p = LocalProvider(base_url="http://localhost:11434/v1")
        assert p.config.base_url == "http://localhost:11434/v1"

    def test_custom_api_key(self):
        p = LocalProvider(api_key="ollama")
        assert p.config.api_key == "ollama"

    def test_base_url_from_env(self):
        with patch.dict("os.environ", {"LOCAL_BASE_URL": "http://custom:8080/v1"}):
            p = LocalProvider()
            assert p.config.base_url == "http://custom:8080/v1"

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"LOCAL_API_KEY": "env-local-key"}):
            p = LocalProvider()
            assert p.config.api_key == "env-local-key"

    def test_explicit_base_url_overrides_env(self):
        with patch.dict("os.environ", {"LOCAL_BASE_URL": "http://env:9999/v1"}):
            p = LocalProvider(base_url="http://explicit:7777/v1")
            assert p.config.base_url == "http://explicit:7777/v1"

    def test_default_base_url_constant(self):
        assert LocalProvider.DEFAULT_BASE_URL == "http://localhost:1234/v1"


# ---------------------------------------------------------------------------
# provider_name override
# ---------------------------------------------------------------------------


class TestLocalProviderName:
    def test_provider_name_returns_local(self):
        p = LocalProvider()
        assert p.provider_name == "local"

    def test_provider_name_not_openai(self):
        """Ensure it overrides the default class-name-based logic."""
        p = LocalProvider()
        assert p.provider_name != "openai"
        assert p.provider_name != "localprovider"


# ---------------------------------------------------------------------------
# validate() override — calls client.models.list()
# ---------------------------------------------------------------------------


class TestLocalValidate:
    def test_validate_calls_models_list(self):
        p = LocalProvider()
        p._client = MagicMock()
        p._client.models.list.return_value = MagicMock()

        result = p.validate()
        assert result is True
        p._client.models.list.assert_called_once()

    def test_validate_returns_false_on_connection_error(self):
        p = LocalProvider()
        p._client = MagicMock()
        p._client.models.list.side_effect = ConnectionError("refused")

        result = p.validate()
        assert result is False

    def test_validate_returns_false_on_any_exception(self):
        p = LocalProvider()
        p._client = MagicMock()
        p._client.models.list.side_effect = Exception("timeout")

        result = p.validate()
        assert result is False

    def test_validate_does_not_use_base_ping(self):
        """LocalProvider.validate() should NOT call complete() like BaseProvider."""
        p = LocalProvider()
        p._client = MagicMock()
        p._client.models.list.return_value = MagicMock()
        p.complete = MagicMock()

        p.validate()
        p.complete.assert_not_called()


# ---------------------------------------------------------------------------
# count_tokens() override — always char/4 heuristic
# ---------------------------------------------------------------------------


class TestLocalCountTokens:
    def test_char_heuristic(self):
        p = LocalProvider()
        p._client = MagicMock()
        messages = [ChatMessage(role=MessageRole.USER, content="a" * 100)]
        assert p.count_tokens(messages) == 25

    def test_multiple_messages(self):
        p = LocalProvider()
        p._client = MagicMock()
        messages = [
            ChatMessage(role=MessageRole.USER, content="a" * 40),
            ChatMessage(role=MessageRole.ASSISTANT, content="b" * 60),
        ]
        assert p.count_tokens(messages) == 25  # 100 // 4

    def test_empty_message(self):
        p = LocalProvider()
        p._client = MagicMock()
        messages = [ChatMessage(role=MessageRole.USER, content="")]
        assert p.count_tokens(messages) == 0

    def test_short_message(self):
        p = LocalProvider()
        p._client = MagicMock()
        messages = [ChatMessage(role=MessageRole.USER, content="Hi")]
        # 2 chars // 4 = 0
        assert p.count_tokens(messages) == 0

    def test_does_not_use_tiktoken(self):
        """LocalProvider should never attempt tiktoken import."""
        p = LocalProvider()
        p._client = MagicMock()
        messages = [ChatMessage(role=MessageRole.USER, content="test" * 50)]

        with patch("builtins.__import__") as mock_import:
            # If tiktoken is imported, fail the test
            original = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

            def side_effect(name, *args, **kwargs):
                if name == "tiktoken":
                    raise AssertionError("LocalProvider should not import tiktoken")
                return original(name, *args, **kwargs)

            mock_import.side_effect = side_effect
            # Just verify char heuristic works directly
            result = p.count_tokens(messages)
            assert result == 200 // 4


# ---------------------------------------------------------------------------
# Async interface (inherited from OpenAIProvider, exercised on LocalProvider)
# ---------------------------------------------------------------------------


def _mock_chat_response(content="hi", prompt=5, completion=3):
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    usage.total_tokens = prompt + completion
    response = MagicMock()
    response.choices = [choice]
    response.model = "local-model"
    response.usage = usage
    return response


class _AsyncChunkStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _content_chunk(text, usage=None, finish_reason=None):
    delta = MagicMock()
    delta.content = text
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = usage
    return chunk


def _usage(prompt=5, completion=3):
    u = MagicMock()
    u.prompt_tokens = prompt
    u.completion_tokens = completion
    u.total_tokens = prompt + completion
    return u


class TestLocalProviderAsync:
    @pytest.mark.asyncio
    async def test_acomplete(self):
        from unittest.mock import AsyncMock

        p = LocalProvider()
        p._async_client = MagicMock()
        p._async_client.chat.completions.create = AsyncMock(
            return_value=_mock_chat_response("local async")
        )
        msgs = [ChatMessage(role=MessageRole.USER, content="hi")]
        result = await p.acomplete(msgs)
        assert result.content == "local async"

    @pytest.mark.asyncio
    async def test_astream(self):
        from unittest.mock import AsyncMock

        p = LocalProvider()
        p._async_client = MagicMock()
        final = MagicMock()
        final.choices = []
        final.usage = _usage()
        p._async_client.chat.completions.create = AsyncMock(
            return_value=_AsyncChunkStream([_content_chunk("tok"), final])
        )
        msgs = [ChatMessage(role=MessageRole.USER, content="hi")]
        out = [c async for c in p.astream(msgs)]
        assert [c.text for c in out if not c.is_final] == ["tok"]
        assert out[-1].is_final is True

    def test_async_client_uses_async_openai(self):
        mock_mod = MagicMock()
        mock_mod.AsyncOpenAI.return_value = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_mod}):
            p = LocalProvider()
            p._get_async_client()
            mock_mod.AsyncOpenAI.assert_called_once()
            # local default base_url is forwarded to the async client
            assert mock_mod.AsyncOpenAI.call_args[1]["base_url"] == p.config.base_url
