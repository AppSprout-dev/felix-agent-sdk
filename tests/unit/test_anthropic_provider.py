"""Tests for AnthropicProvider — all Anthropic SDK calls are mocked."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from felix_agent_sdk.providers.anthropic import AnthropicProvider
from felix_agent_sdk.providers.errors import (
    AuthenticationError,
    ContextLengthError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)
from felix_agent_sdk.providers.types import (
    ChatMessage,
    CompletionResult,
    MessageRole,
    StreamChunk,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(api_key="sk-ant-test", model="claude-sonnet-4-5", **kwargs):
    """Create an AnthropicProvider with a pre-injected mock client."""
    p = AnthropicProvider(model=model, api_key=api_key, **kwargs)
    p._client = MagicMock()
    return p


def _mock_response(content="Hello!", input_tokens=10, output_tokens=5,
                    stop_reason="end_turn", model="claude-sonnet-4-5"):
    block = MagicMock()
    block.text = content
    block.type = "text"
    setattr(block, "__class__", type("TextBlock", (), {"text": content}))

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    response = MagicMock()
    response.content = [block]
    response.model = model
    response.usage = usage
    response.stop_reason = stop_reason
    return response


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestAnthropicProviderInit:
    def test_default_model(self):
        p = AnthropicProvider(api_key="k")
        assert p.model == "claude-sonnet-4-5"

    def test_custom_model(self):
        p = AnthropicProvider(model="claude-opus-4-5", api_key="k")
        assert p.model == "claude-opus-4-5"

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            p = AnthropicProvider()
            assert p.config.api_key == "env-key"

    def test_explicit_api_key_overrides_env(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            p = AnthropicProvider(api_key="explicit-key")
            assert p.config.api_key == "explicit-key"

    def test_base_url_stored(self):
        p = AnthropicProvider(api_key="k", base_url="https://proxy.example.com")
        assert p.config.base_url == "https://proxy.example.com"

    def test_provider_name(self):
        p = AnthropicProvider(api_key="k")
        assert p.provider_name == "anthropic"


# ---------------------------------------------------------------------------
# Lazy client initialization
# ---------------------------------------------------------------------------


class TestAnthropicClientInit:
    def test_missing_package_raises_import_error(self):
        p = AnthropicProvider(api_key="k")
        p._client = None
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic"):
                p._get_client()

    def test_client_created_once(self):
        mock_mod = MagicMock()
        mock_mod.Anthropic.return_value = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_mod}):
            p = AnthropicProvider(api_key="k")
            p._client = None
            c1 = p._get_client()
            c2 = p._get_client()
            assert c1 is c2
            mock_mod.Anthropic.assert_called_once()

    def test_client_passes_api_key(self):
        mock_mod = MagicMock()
        mock_mod.Anthropic.return_value = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_mod}):
            p = AnthropicProvider(api_key="my-key")
            p._client = None
            p._get_client()
            call_kwargs = mock_mod.Anthropic.call_args[1]
            assert call_kwargs["api_key"] == "my-key"

    def test_client_passes_base_url(self):
        mock_mod = MagicMock()
        mock_mod.Anthropic.return_value = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_mod}):
            p = AnthropicProvider(api_key="k", base_url="https://proxy.test")
            p._client = None
            p._get_client()
            call_kwargs = mock_mod.Anthropic.call_args[1]
            assert call_kwargs["base_url"] == "https://proxy.test"

    def test_client_omits_base_url_when_none(self):
        mock_mod = MagicMock()
        mock_mod.Anthropic.return_value = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_mod}):
            p = AnthropicProvider(api_key="k")
            p._client = None
            p._get_client()
            call_kwargs = mock_mod.Anthropic.call_args[1]
            assert "base_url" not in call_kwargs


# ---------------------------------------------------------------------------
# _format_messages
# ---------------------------------------------------------------------------


class TestFormatMessages:
    def test_system_extracted(self, system_message, user_message):
        p = _make_provider()
        system, msgs = p._format_messages([system_message, user_message])
        assert system == "You are a helpful assistant."
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_no_system_message(self, user_message):
        p = _make_provider()
        system, msgs = p._format_messages([user_message])
        assert system is None
        assert len(msgs) == 1

    def test_multiple_non_system_preserved(self, user_message, assistant_message):
        p = _make_provider()
        system, msgs = p._format_messages([user_message, assistant_message])
        assert system is None
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_message_content_preserved(self, user_message):
        p = _make_provider()
        _, msgs = p._format_messages([user_message])
        assert msgs[0]["content"] == "Hello, world!"


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------


class TestAnthropicComplete:
    def test_returns_completion_result(self, conversation):
        p = _make_provider()
        p._client.messages.create.return_value = _mock_response()
        result = p.complete(conversation)

        assert isinstance(result, CompletionResult)
        assert result.content == "Hello!"
        assert result.model == "claude-sonnet-4-5"

    def test_usage_populated(self, conversation):
        p = _make_provider()
        p._client.messages.create.return_value = _mock_response(
            input_tokens=20, output_tokens=10
        )
        result = p.complete(conversation)

        assert result.usage["prompt_tokens"] == 20
        assert result.usage["completion_tokens"] == 10
        assert result.usage["total_tokens"] == 30

    def test_finish_reason(self, conversation):
        p = _make_provider()
        p._client.messages.create.return_value = _mock_response(stop_reason="max_tokens")
        result = p.complete(conversation)
        assert result.finish_reason == "max_tokens"

    def test_finish_reason_defaults_to_stop(self, conversation):
        p = _make_provider()
        p._client.messages.create.return_value = _mock_response(stop_reason=None)
        result = p.complete(conversation)
        assert result.finish_reason == "stop"

    def test_system_passed_as_kwarg(self, conversation):
        p = _make_provider()
        p._client.messages.create.return_value = _mock_response()
        p.complete(conversation)

        call_kwargs = p._client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a helpful assistant."

    def test_temperature_and_max_tokens_passed(self, user_message):
        p = _make_provider()
        p._client.messages.create.return_value = _mock_response()
        p.complete([user_message], temperature=0.5, max_tokens=100)

        call_kwargs = p._client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    def test_temperature_falls_back_to_config_default(self, user_message):
        p = _make_provider()
        p._client.messages.create.return_value = _mock_response()
        p.complete([user_message])

        call_kwargs = p._client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == p.config.default_temperature

    def test_stop_sequences_passed(self, user_message):
        p = _make_provider()
        p._client.messages.create.return_value = _mock_response()
        p.complete([user_message], stop_sequences=["STOP", "END"])

        call_kwargs = p._client.messages.create.call_args[1]
        assert call_kwargs["stop_sequences"] == ["STOP", "END"]

    def test_stop_sequences_omitted_when_none(self, user_message):
        p = _make_provider()
        p._client.messages.create.return_value = _mock_response()
        p.complete([user_message])

        call_kwargs = p._client.messages.create.call_args[1]
        assert "stop_sequences" not in call_kwargs

    def test_extra_kwargs_forwarded(self, user_message):
        p = _make_provider()
        p._client.messages.create.return_value = _mock_response()
        p.complete([user_message], top_p=0.9)

        call_kwargs = p._client.messages.create.call_args[1]
        assert call_kwargs["top_p"] == 0.9

    def test_multi_block_content_concatenated(self, user_message):
        p = _make_provider()
        block1 = MagicMock()
        block1.text = "Hello "
        block2 = MagicMock()
        block2.text = "world!"
        response = _mock_response()
        response.content = [block1, block2]
        p._client.messages.create.return_value = response
        result = p.complete([user_message])
        assert result.content == "Hello world!"

    def test_raw_response_attached(self, user_message):
        p = _make_provider()
        mock_resp = _mock_response()
        p._client.messages.create.return_value = mock_resp
        result = p.complete([user_message])
        assert result.raw_response is mock_resp


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


class TestAnthropicStream:
    def test_yields_stream_chunks(self, user_message):
        p = _make_provider()

        mock_stream = MagicMock()
        mock_stream.text_stream = iter(["Hello", " world"])
        final_msg = _mock_response(input_tokens=10, output_tokens=5)
        mock_stream.get_final_message.return_value = final_msg
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        p._client.messages.stream.return_value = mock_stream

        chunks = list(p.stream([user_message]))
        texts = [c.text for c in chunks if not c.is_final]
        assert texts == ["Hello", " world"]

    def test_final_chunk_has_usage(self, user_message):
        p = _make_provider()

        mock_stream = MagicMock()
        mock_stream.text_stream = iter(["tok"])
        final_msg = _mock_response(input_tokens=15, output_tokens=8)
        mock_stream.get_final_message.return_value = final_msg
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        p._client.messages.stream.return_value = mock_stream

        chunks = list(p.stream([user_message]))
        final = chunks[-1]
        assert final.is_final is True
        assert final.text == ""
        assert final.usage["prompt_tokens"] == 15
        assert final.usage["completion_tokens"] == 8
        assert final.usage["total_tokens"] == 23

    def test_stream_passes_system_kwarg(self, conversation):
        p = _make_provider()

        mock_stream = MagicMock()
        mock_stream.text_stream = iter([])
        mock_stream.get_final_message.return_value = _mock_response()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        p._client.messages.stream.return_value = mock_stream

        list(p.stream(conversation))
        call_kwargs = p._client.messages.stream.call_args[1]
        assert call_kwargs["system"] == "You are a helpful assistant."


# ---------------------------------------------------------------------------
# count_tokens()
# ---------------------------------------------------------------------------


class TestAnthropicCountTokens:
    def test_uses_api_when_available(self, user_message):
        p = _make_provider()
        token_resp = MagicMock()
        token_resp.input_tokens = 42
        p._client.messages.count_tokens.return_value = token_resp

        result = p.count_tokens([user_message])
        assert result == 42

    def test_fallback_on_api_error(self, user_message):
        p = _make_provider()
        p._client.messages.count_tokens.side_effect = Exception("not supported")

        result = p.count_tokens([user_message])
        # "Hello, world!" = 13 chars → 13 // 4 = 3
        assert result == len("Hello, world!") // 4

    def test_fallback_char_heuristic(self):
        p = _make_provider()
        p._client.messages.count_tokens.side_effect = Exception("fail")

        messages = [
            ChatMessage(role=MessageRole.USER, content="a" * 100),
            ChatMessage(role=MessageRole.ASSISTANT, content="b" * 200),
        ]
        result = p.count_tokens(messages)
        assert result == 300 // 4


# ---------------------------------------------------------------------------
# _translate_error()
# ---------------------------------------------------------------------------


class TestAnthropicTranslateError:
    def test_authentication_error(self):
        p = _make_provider()
        err = Exception("Invalid authentication credentials")
        result = p._translate_error(err)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "anthropic"

    def test_api_key_error(self):
        p = _make_provider()
        err = Exception("Invalid api_key provided")
        result = p._translate_error(err)
        assert isinstance(result, AuthenticationError)

    def test_rate_limit_from_type(self):
        p = _make_provider()

        # Class name must contain "rate_limit" (with underscore) to match error_type check
        class rate_limit_error(Exception):
            pass

        err = rate_limit_error("slow down")
        result = p._translate_error(err)
        assert isinstance(result, RateLimitError)

    def test_rate_limit_from_429(self):
        p = _make_provider()
        err = Exception("Error code: 429 Too Many Requests")
        result = p._translate_error(err)
        assert isinstance(result, RateLimitError)

    def test_model_not_found_from_type(self):
        p = _make_provider()

        class NotFoundError(Exception):
            pass

        err = NotFoundError("model does not exist")
        result = p._translate_error(err)
        assert isinstance(result, ModelNotFoundError)

    def test_model_not_found_from_message(self):
        p = _make_provider()
        err = Exception("The model claude-bad is not available")
        result = p._translate_error(err)
        assert isinstance(result, ModelNotFoundError)

    def test_context_length_error(self):
        p = _make_provider()
        err = Exception("prompt is too long for context window")
        result = p._translate_error(err)
        assert isinstance(result, ContextLengthError)

    def test_context_too_long_error(self):
        p = _make_provider()
        err = Exception("input is too long")
        result = p._translate_error(err)
        assert isinstance(result, ContextLengthError)

    def test_generic_error(self):
        p = _make_provider()
        err = Exception("something unknown")
        result = p._translate_error(err)
        assert isinstance(result, ProviderError)
        assert not isinstance(result, AuthenticationError)
        assert result.provider == "anthropic"


# ---------------------------------------------------------------------------
# Error propagation in complete/stream
# ---------------------------------------------------------------------------


class TestAnthropicErrorPropagation:
    def test_complete_translates_exceptions(self, user_message):
        p = _make_provider()
        p._client.messages.create.side_effect = Exception("authentication failed")
        with pytest.raises(AuthenticationError):
            p.complete([user_message])

    def test_stream_translates_exceptions(self, user_message):
        p = _make_provider()
        p._client.messages.stream.side_effect = Exception("Error 429: rate limit exceeded")
        with pytest.raises(RateLimitError):
            list(p.stream([user_message]))
