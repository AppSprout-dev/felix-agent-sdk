"""Tests for OpenAIProvider — all OpenAI SDK calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from felix_agent_sdk.providers.errors import (
    AuthenticationError,
    ContextLengthError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)
from felix_agent_sdk.providers.openai_provider import OpenAIProvider
from felix_agent_sdk.providers.types import (
    ChatMessage,
    CompletionResult,
    MessageRole,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(api_key="sk-test", model="gpt-4o", **kwargs):
    p = OpenAIProvider(model=model, api_key=api_key, **kwargs)
    p._client = MagicMock()
    return p


def _mock_response(content="Hello!", prompt_tokens=10, completion_tokens=5,
                    finish_reason="stop", model="gpt-4o"):
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
    response.model = model
    response.usage = usage
    return response


def _mock_stream_chunks(texts, prompt_tokens=10, completion_tokens=5):
    """Create a list of mock streaming chunks with a final usage chunk."""
    chunks = []
    for text in texts:
        chunk = MagicMock()
        delta = MagicMock()
        delta.content = text
        choice = MagicMock()
        choice.delta = delta
        chunk.choices = [choice]
        chunk.usage = None
        chunks.append(chunk)

    # Final chunk with usage
    final = MagicMock()
    final.choices = []
    final_usage = MagicMock()
    final_usage.prompt_tokens = prompt_tokens
    final_usage.completion_tokens = completion_tokens
    final_usage.total_tokens = prompt_tokens + completion_tokens
    final.usage = final_usage
    chunks.append(final)

    return chunks


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestOpenAIProviderInit:
    def test_default_model(self):
        p = OpenAIProvider(api_key="k")
        assert p.model == "gpt-4o"

    def test_custom_model(self):
        p = OpenAIProvider(model="gpt-4-turbo", api_key="k")
        assert p.model == "gpt-4-turbo"

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            p = OpenAIProvider()
            assert p.config.api_key == "env-key"

    def test_explicit_api_key_overrides_env(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            p = OpenAIProvider(api_key="explicit-key")
            assert p.config.api_key == "explicit-key"

    def test_base_url_stored(self):
        p = OpenAIProvider(api_key="k", base_url="https://azure.example.com")
        assert p.config.base_url == "https://azure.example.com"

    def test_provider_name(self):
        p = OpenAIProvider(api_key="k")
        assert p.provider_name == "openai"


# ---------------------------------------------------------------------------
# Lazy client initialization
# ---------------------------------------------------------------------------


class TestOpenAIClientInit:
    def test_missing_package_raises_import_error(self):
        p = OpenAIProvider(api_key="k")
        p._client = None
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                p._get_client()

    def test_client_created_once(self):
        mock_mod = MagicMock()
        mock_mod.OpenAI.return_value = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_mod}):
            p = OpenAIProvider(api_key="k")
            p._client = None
            c1 = p._get_client()
            c2 = p._get_client()
            assert c1 is c2
            mock_mod.OpenAI.assert_called_once()

    def test_client_passes_api_key(self):
        mock_mod = MagicMock()
        mock_mod.OpenAI.return_value = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_mod}):
            p = OpenAIProvider(api_key="my-key")
            p._client = None
            p._get_client()
            call_kwargs = mock_mod.OpenAI.call_args[1]
            assert call_kwargs["api_key"] == "my-key"

    def test_client_passes_base_url(self):
        mock_mod = MagicMock()
        mock_mod.OpenAI.return_value = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_mod}):
            p = OpenAIProvider(api_key="k", base_url="https://custom.api")
            p._client = None
            p._get_client()
            call_kwargs = mock_mod.OpenAI.call_args[1]
            assert call_kwargs["base_url"] == "https://custom.api"


# ---------------------------------------------------------------------------
# _format_messages
# ---------------------------------------------------------------------------


class TestOpenAIFormatMessages:
    def test_system_stays_inline(self, system_message, user_message):
        p = _make_provider()
        msgs = p._format_messages([system_message, user_message])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_all_content_preserved(self, user_message, assistant_message):
        p = _make_provider()
        msgs = p._format_messages([user_message, assistant_message])
        assert msgs[0]["content"] == "Hello, world!"
        assert msgs[1]["content"] == "Hi there!"


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------


class TestOpenAIComplete:
    def test_returns_completion_result(self, user_message):
        p = _make_provider()
        p._client.chat.completions.create.return_value = _mock_response()
        result = p.complete([user_message])

        assert isinstance(result, CompletionResult)
        assert result.content == "Hello!"
        assert result.model == "gpt-4o"

    def test_usage_populated(self, user_message):
        p = _make_provider()
        p._client.chat.completions.create.return_value = _mock_response(
            prompt_tokens=20, completion_tokens=15
        )
        result = p.complete([user_message])

        assert result.usage["prompt_tokens"] == 20
        assert result.usage["completion_tokens"] == 15
        assert result.usage["total_tokens"] == 35

    def test_finish_reason(self, user_message):
        p = _make_provider()
        p._client.chat.completions.create.return_value = _mock_response(
            finish_reason="length"
        )
        result = p.complete([user_message])
        assert result.finish_reason == "length"

    def test_none_content_becomes_empty_string(self, user_message):
        p = _make_provider()
        resp = _mock_response()
        resp.choices[0].message.content = None
        p._client.chat.completions.create.return_value = resp
        result = p.complete([user_message])
        assert result.content == ""

    def test_none_usage_handled(self, user_message):
        p = _make_provider()
        resp = _mock_response()
        resp.usage = None
        p._client.chat.completions.create.return_value = resp
        result = p.complete([user_message])
        assert result.usage["prompt_tokens"] == 0
        assert result.usage["completion_tokens"] == 0
        assert result.usage["total_tokens"] == 0

    def test_stop_sequences_mapped_to_stop(self, user_message):
        p = _make_provider()
        p._client.chat.completions.create.return_value = _mock_response()
        p.complete([user_message], stop_sequences=["STOP"])

        call_kwargs = p._client.chat.completions.create.call_args[1]
        assert call_kwargs["stop"] == ["STOP"]
        assert "stop_sequences" not in call_kwargs

    def test_stop_not_set_when_none(self, user_message):
        p = _make_provider()
        p._client.chat.completions.create.return_value = _mock_response()
        p.complete([user_message])

        call_kwargs = p._client.chat.completions.create.call_args[1]
        assert "stop" not in call_kwargs

    def test_temperature_and_max_tokens_passed(self, user_message):
        p = _make_provider()
        p._client.chat.completions.create.return_value = _mock_response()
        p.complete([user_message], temperature=0.2, max_tokens=200)

        call_kwargs = p._client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 200

    def test_extra_kwargs_forwarded(self, user_message):
        p = _make_provider()
        p._client.chat.completions.create.return_value = _mock_response()
        p.complete([user_message], top_p=0.95, frequency_penalty=0.5)

        call_kwargs = p._client.chat.completions.create.call_args[1]
        assert call_kwargs["top_p"] == 0.95
        assert call_kwargs["frequency_penalty"] == 0.5

    def test_raw_response_attached(self, user_message):
        p = _make_provider()
        mock_resp = _mock_response()
        p._client.chat.completions.create.return_value = mock_resp
        result = p.complete([user_message])
        assert result.raw_response is mock_resp


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


class TestOpenAIStream:
    def test_yields_text_chunks(self, user_message):
        p = _make_provider()
        stream_data = _mock_stream_chunks(["Hello", " world"])
        p._client.chat.completions.create.return_value = iter(stream_data)

        chunks = list(p.stream([user_message]))
        texts = [c.text for c in chunks if not c.is_final]
        assert texts == ["Hello", " world"]

    def test_final_chunk_has_usage(self, user_message):
        p = _make_provider()
        stream_data = _mock_stream_chunks(
            ["tok"], prompt_tokens=12, completion_tokens=3
        )
        p._client.chat.completions.create.return_value = iter(stream_data)

        chunks = list(p.stream([user_message]))
        final = [c for c in chunks if c.is_final]
        assert len(final) == 1
        assert final[0].usage["prompt_tokens"] == 12
        assert final[0].usage["completion_tokens"] == 3
        assert final[0].usage["total_tokens"] == 15

    def test_stream_options_include_usage(self, user_message):
        p = _make_provider()
        p._client.chat.completions.create.return_value = iter(
            _mock_stream_chunks([])
        )
        list(p.stream([user_message]))

        call_kwargs = p._client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True
        assert call_kwargs["stream_options"] == {"include_usage": True}

    def test_stream_stop_sequences_mapped(self, user_message):
        p = _make_provider()
        p._client.chat.completions.create.return_value = iter(
            _mock_stream_chunks([])
        )
        list(p.stream([user_message], stop_sequences=["END"]))

        call_kwargs = p._client.chat.completions.create.call_args[1]
        assert call_kwargs["stop"] == ["END"]


# ---------------------------------------------------------------------------
# count_tokens()
# ---------------------------------------------------------------------------


class TestOpenAICountTokens:
    def test_tiktoken_used_when_available(self, user_message):
        p = _make_provider()
        mock_encoding = MagicMock()
        mock_encoding.encode.side_effect = lambda s: list(range(len(s)))

        mock_tiktoken = MagicMock()
        mock_tiktoken.encoding_for_model.return_value = mock_encoding
        with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
            result = p.count_tokens([user_message])

        # 4 (overhead) + len("Hello, world!") + len("user") + 2 (reply priming)
        expected = 4 + 13 + 4 + 2
        assert result == expected

    def test_fallback_on_key_error(self, user_message):
        """When tiktoken doesn't know the model, falls back to char heuristic."""
        p = _make_provider(model="unknown-model-xyz")
        mock_tiktoken = MagicMock()
        mock_tiktoken.encoding_for_model.side_effect = KeyError("unknown model")

        with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
            result = p.count_tokens([user_message])

        assert result == len("Hello, world!") // 4

    def test_fallback_char_heuristic_multiple_messages(self):
        p = _make_provider()
        messages = [
            ChatMessage(role=MessageRole.USER, content="a" * 100),
            ChatMessage(role=MessageRole.ASSISTANT, content="b" * 200),
        ]
        mock_tiktoken = MagicMock()
        mock_tiktoken.encoding_for_model.side_effect = KeyError("bad model")

        with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
            result = p.count_tokens(messages)

        assert result == 300 // 4

    def test_tiktoken_import_error_fallback(self):
        """When tiktoken is not installed, falls back to char heuristic."""
        p = _make_provider()
        messages = [ChatMessage(role=MessageRole.USER, content="a" * 80)]

        with patch.dict("sys.modules", {"tiktoken": None}):
            result = p.count_tokens(messages)

        assert result == 80 // 4


# ---------------------------------------------------------------------------
# _translate_error()
# ---------------------------------------------------------------------------


class TestOpenAITranslateError:
    def test_authentication_error(self):
        p = _make_provider()
        err = Exception("Incorrect API key provided: sk-test. authentication failed.")
        result = p._translate_error(err)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "openai"

    def test_api_key_error(self):
        p = _make_provider()
        err = Exception("Invalid api_key")
        result = p._translate_error(err)
        assert isinstance(result, AuthenticationError)

    def test_rate_limit_from_type(self):
        p = _make_provider()

        # Class name must contain "rate_limit" (with underscore) to match error_type check
        class rate_limit_error(Exception):
            pass

        err = rate_limit_error("too many requests")
        result = p._translate_error(err)
        assert isinstance(result, RateLimitError)

    def test_rate_limit_from_429(self):
        p = _make_provider()
        err = Exception("Error code: 429")
        result = p._translate_error(err)
        assert isinstance(result, RateLimitError)

    def test_model_not_found(self):
        p = _make_provider()

        # Class name must contain "not_found" (with underscore) to match error_type check
        class not_found_error(Exception):
            pass

        err = not_found_error("model not found")
        result = p._translate_error(err)
        assert isinstance(result, ModelNotFoundError)

    def test_context_length_error(self):
        p = _make_provider()
        err = Exception("This model's maximum context length is 8192")
        result = p._translate_error(err)
        assert isinstance(result, ContextLengthError)

    def test_context_length_from_keyword(self):
        p = _make_provider()
        err = Exception("context_length_exceeded")
        result = p._translate_error(err)
        assert isinstance(result, ContextLengthError)

    def test_generic_error(self):
        p = _make_provider()
        err = Exception("server error")
        result = p._translate_error(err)
        assert isinstance(result, ProviderError)
        assert not isinstance(result, AuthenticationError)
        assert result.provider == "openai"


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


class TestOpenAIErrorPropagation:
    def test_complete_translates_exceptions(self, user_message):
        p = _make_provider()
        p._client.chat.completions.create.side_effect = Exception("authentication error")
        with pytest.raises(AuthenticationError):
            p.complete([user_message])

    def test_stream_translates_exceptions(self, user_message):
        p = _make_provider()
        p._client.chat.completions.create.side_effect = Exception("429 rate limit")
        with pytest.raises(RateLimitError):
            list(p.stream([user_message]))


# ---------------------------------------------------------------------------
# Status-code / retry-after aware error translation
# ---------------------------------------------------------------------------


def _status_error(message, status_code, retry_after=None):
    """Mimic a vendor SDK APIStatusError carrying status_code and response."""
    err = Exception(message)
    err.status_code = status_code
    headers = {}
    if retry_after is not None:
        headers["retry-after"] = retry_after
    response = MagicMock()
    response.headers = headers
    err.response = response
    return err


class TestOpenAITypedErrorTranslation:
    def test_status_401_maps_to_authentication(self):
        p = _make_provider()
        result = p._translate_error(_status_error("nope", 401))
        assert isinstance(result, AuthenticationError)
        assert result.status_code == 401

    def test_status_429_maps_to_rate_limit_with_retry_after(self):
        p = _make_provider()
        result = p._translate_error(_status_error("slow down", 429, retry_after="2.5"))
        assert isinstance(result, RateLimitError)
        assert result.status_code == 429
        assert result.retry_after == 2.5

    def test_retry_after_absent_is_none(self):
        p = _make_provider()
        result = p._translate_error(_status_error("slow down", 429))
        assert isinstance(result, RateLimitError)
        assert result.retry_after is None

    def test_non_numeric_retry_after_is_none(self):
        p = _make_provider()
        result = p._translate_error(
            _status_error("slow down", 429, retry_after="Wed, 21 Oct 2026 07:28:00 GMT")
        )
        assert isinstance(result, RateLimitError)
        assert result.retry_after is None

    def test_generic_error_carries_status_code(self):
        p = _make_provider()
        result = p._translate_error(_status_error("boom", 500))
        assert isinstance(result, ProviderError)
        assert result.status_code == 500


# ---------------------------------------------------------------------------
# Streaming stops after the final usage chunk
# ---------------------------------------------------------------------------


class TestOpenAIStreamTermination:
    def test_stream_stops_after_final_chunk(self, user_message):
        p = _make_provider()
        chunks = _mock_stream_chunks(["a", "b"])

        # A chunk after the usage chunk must never be consumed
        poisoned = MagicMock()
        poisoned.choices = []
        poisoned.usage = None

        consumed = []

        def chunk_iter():
            for c in chunks:
                consumed.append(c)
                yield c
            consumed.append(poisoned)
            yield poisoned

        p._client.chat.completions.create.return_value = chunk_iter()
        out = list(p.stream([user_message]))

        assert out[-1].is_final is True
        assert poisoned not in consumed


# ---------------------------------------------------------------------------
# Async interface
# ---------------------------------------------------------------------------


class _AsyncChunkStream:
    """Async-iterable wrapper over a list of mock chunks."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class TestOpenAIAsync:
    @pytest.mark.asyncio
    async def test_acomplete_returns_completion_result(self, user_message):
        from unittest.mock import AsyncMock

        p = _make_provider()
        p._async_client = MagicMock()
        p._async_client.chat.completions.create = AsyncMock(return_value=_mock_response())

        result = await p.acomplete([user_message])
        assert isinstance(result, CompletionResult)
        assert result.content == "Hello!"

    @pytest.mark.asyncio
    async def test_astream_yields_chunks_with_final_usage(self, user_message):
        from unittest.mock import AsyncMock

        p = _make_provider()
        p._async_client = MagicMock()
        p._async_client.chat.completions.create = AsyncMock(
            return_value=_AsyncChunkStream(_mock_stream_chunks(["Hello", " world"]))
        )

        out = [chunk async for chunk in p.astream([user_message])]
        texts = [c.text for c in out if not c.is_final]
        assert texts == ["Hello", " world"]
        assert out[-1].is_final is True
        assert out[-1].usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_acomplete_translates_errors(self, user_message):
        from unittest.mock import AsyncMock

        p = _make_provider()
        p._async_client = MagicMock()
        p._async_client.chat.completions.create = AsyncMock(
            side_effect=Exception("429 rate limit")
        )
        with pytest.raises(RateLimitError):
            await p.acomplete([user_message])

    def test_async_client_created_once(self):
        mock_mod = MagicMock()
        mock_mod.AsyncOpenAI.return_value = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_mod}):
            p = OpenAIProvider(api_key="k")
            c1 = p._get_async_client()
            c2 = p._get_async_client()
            assert c1 is c2
            mock_mod.AsyncOpenAI.assert_called_once()


# ---------------------------------------------------------------------------
# Continuous usage stats (vLLM-style) must not truncate the stream
# ---------------------------------------------------------------------------


def _content_chunk(text, usage=None, finish_reason=None):
    """A content chunk with explicit usage/finish_reason (no MagicMock auto-attrs)."""
    delta = MagicMock()
    delta.content = text
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = usage
    return chunk


def _usage(prompt=10, completion=5):
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    usage.total_tokens = prompt + completion
    return usage


class TestOpenAIContinuousUsageStats:
    def test_mid_stream_usage_does_not_truncate(self, user_message):
        """Regression: servers like vLLM (continuous_usage_stats) attach
        running usage to every content chunk; the stream must not stop at
        the first one."""
        p = _make_provider()
        chunks = [
            _content_chunk("The ", usage=_usage(10, 1)),
            _content_chunk("quick ", usage=_usage(10, 2)),
            _content_chunk("fox", usage=_usage(10, 3), finish_reason="stop"),
        ]
        p._client.chat.completions.create.return_value = iter(chunks)

        out = list(p.stream([user_message]))
        texts = [c.text for c in out if not c.is_final]
        assert texts == ["The ", "quick ", "fox"]
        # finish_reason chunk carries usage -> terminal
        assert out[-1].is_final is True
        assert out[-1].usage["completion_tokens"] == 3

    def test_usage_only_final_chunk_still_terminal(self, user_message):
        """Stock OpenAI shape: dedicated choices-empty usage chunk."""
        p = _make_provider()
        final = MagicMock()
        final.choices = []
        final.usage = _usage()
        p._client.chat.completions.create.return_value = iter(
            [_content_chunk("hi"), final]
        )
        out = list(p.stream([user_message]))
        assert out[-1].is_final is True


class TestOpenAIAstreamTermination:
    @pytest.mark.asyncio
    async def test_astream_stops_after_final_chunk(self, user_message):
        """Async twin of the poisoned-chunk guard: chunks after the terminal
        usage chunk must never be consumed."""
        from unittest.mock import AsyncMock

        consumed = []
        final = MagicMock()
        final.choices = []
        final.usage = _usage()
        poisoned = MagicMock()
        poisoned.choices = []
        poisoned.usage = None

        class _TrackingAsyncStream:
            def __init__(self, chunks):
                self._chunks = list(chunks)

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._chunks:
                    raise StopAsyncIteration
                chunk = self._chunks.pop(0)
                consumed.append(chunk)
                return chunk

        p = _make_provider()
        p._async_client = MagicMock()
        p._async_client.chat.completions.create = AsyncMock(
            return_value=_TrackingAsyncStream([_content_chunk("a"), final, poisoned])
        )

        out = [c async for c in p.astream([user_message])]
        assert out[-1].is_final is True
        assert poisoned not in consumed


class TestOpenAIModelNotFoundTranslation:
    def test_status_404_maps_to_model_not_found(self):
        p = _make_provider()
        result = p._translate_error(_status_error("missing", 404))
        assert isinstance(result, ModelNotFoundError)
        assert result.status_code == 404

    def test_sdk_notfounderror_class_name_maps(self):
        """Regression: 'not_found' substring never matched the SDK's
        NotFoundError class name."""
        p = _make_provider()

        class NotFoundError(Exception):
            pass

        result = p._translate_error(NotFoundError("model gone"))
        assert isinstance(result, ModelNotFoundError)


class _ClosableStream:
    """Iterable stream whose close() raises, mimicking a teardown failure."""

    def __init__(self, chunks, raise_on_close=True):
        self._chunks = list(chunks)
        self._raise_on_close = raise_on_close
        self.closed = False

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        self.closed = True
        if self._raise_on_close:
            raise RuntimeError("connection reset during teardown")


class TestOpenAIStreamCloseErrors:
    def test_close_failure_does_not_mask_successful_stream(self, user_message):
        """A close() raising after the final chunk must not turn a complete
        stream into a ProviderError."""
        p = _make_provider()
        final = MagicMock()
        final.choices = []
        final.usage = _usage()
        stream = _ClosableStream([_content_chunk("hi"), final])
        p._client.chat.completions.create.return_value = stream

        out = list(p.stream([user_message]))  # must not raise
        assert out[-1].is_final is True
        assert stream.closed is True
