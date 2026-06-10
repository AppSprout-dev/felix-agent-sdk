"""OpenAI and OpenAI-compatible API provider for Felix.

Requires: pip install felix-agent-sdk[openai]

Note: This file is named openai_provider.py (not openai.py) to avoid
shadowing the openai package import within this module.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence

from .base import BaseProvider
from .errors import (
    AuthenticationError,
    ContextLengthError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    extract_retry_after,
    extract_status_code,
)
from .types import ChatMessage, CompletionResult, ProviderConfig, StreamChunk


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI models and OpenAI-compatible APIs.

    Requires: pip install felix-agent-sdk[openai]

    Supported models: gpt-4o, gpt-4o-mini, gpt-4-turbo, o1, o1-mini,
    and any model available through the OpenAI API.

    This provider also works with OpenAI-compatible APIs (e.g., Azure OpenAI,
    Together AI, Fireworks) by setting a custom base_url.

    Configuration:
        - api_key: Set via constructor or OPENAI_API_KEY env var.
        - base_url: Defaults to OpenAI's API. Override for compatible APIs.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        config = ProviderConfig(
            model=model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            **kwargs,
        )
        super().__init__(config)

    def _import_sdk(self) -> Any:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI provider requires the 'openai' package. "
                "Install with: pip install felix-agent-sdk[openai]"
            )
        return openai

    def _client_kwargs(self) -> Dict[str, Any]:
        client_kwargs: Dict[str, Any] = {}
        if self.config.api_key:
            client_kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        client_kwargs["max_retries"] = self.config.max_retries
        client_kwargs["timeout"] = self.config.timeout
        return client_kwargs

    def _get_client(self) -> Any:
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            openai = self._import_sdk()
            self._client = openai.OpenAI(**self._client_kwargs())
        return self._client

    def _get_async_client(self) -> Any:
        """Lazy-initialize the async OpenAI client."""
        if self._async_client is None:
            openai = self._import_sdk()
            self._async_client = openai.AsyncOpenAI(**self._client_kwargs())
        return self._async_client

    def _format_messages(self, messages: Sequence[ChatMessage]) -> List[Dict[str, str]]:
        """Convert ChatMessages to OpenAI's format (system stays inline)."""
        return [{"role": msg.role.value, "content": msg.content} for msg in messages]

    def _build_create_kwargs(
        self,
        messages: Sequence[ChatMessage],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stop_sequences: Optional[List[str]],
        extra: Dict[str, Any],
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Assemble the keyword arguments for a chat.completions.create call."""
        create_kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": self._format_messages(messages),
            "temperature": self._resolve_temperature(temperature),
            "max_tokens": self._resolve_max_tokens(max_tokens),
        }
        if stream:
            create_kwargs["stream"] = True
            create_kwargs["stream_options"] = {"include_usage": True}
        if stop_sequences:
            create_kwargs["stop"] = stop_sequences
        create_kwargs.update(extra)
        return create_kwargs

    @staticmethod
    def _to_completion_result(response: Any) -> CompletionResult:
        choice = response.choices[0]
        usage = response.usage
        return CompletionResult(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            finish_reason=choice.finish_reason or "stop",
            raw_response=response,
        )

    @staticmethod
    def _chunk_to_stream_chunks(chunk: Any) -> Iterator[StreamChunk]:
        """Translate one raw OpenAI stream chunk into StreamChunks."""
        if chunk.choices and chunk.choices[0].delta.content:
            yield StreamChunk(text=chunk.choices[0].delta.content)

        # Final chunk with usage (sent last when include_usage is set)
        if chunk.usage:
            yield StreamChunk(
                text="",
                is_final=True,
                usage={
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                },
            )

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> CompletionResult:
        client = self._get_client()
        create_kwargs = self._build_create_kwargs(
            messages, temperature, max_tokens, stop_sequences, kwargs
        )

        try:
            response = client.chat.completions.create(**create_kwargs)
            return self._to_completion_result(response)
        except Exception as e:
            raise self._translate_error(e)

    async def acomplete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> CompletionResult:
        client = self._get_async_client()
        create_kwargs = self._build_create_kwargs(
            messages, temperature, max_tokens, stop_sequences, kwargs
        )

        try:
            response = await client.chat.completions.create(**create_kwargs)
            return self._to_completion_result(response)
        except Exception as e:
            raise self._translate_error(e)

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        client = self._get_client()
        create_kwargs = self._build_create_kwargs(
            messages, temperature, max_tokens, stop_sequences, kwargs, stream=True
        )

        try:
            response = client.chat.completions.create(**create_kwargs)
            for chunk in response:
                done = False
                for stream_chunk in self._chunk_to_stream_chunks(chunk):
                    done = done or stream_chunk.is_final
                    yield stream_chunk
                if done:
                    break
        except Exception as e:
            raise self._translate_error(e)

    async def astream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        client = self._get_async_client()
        create_kwargs = self._build_create_kwargs(
            messages, temperature, max_tokens, stop_sequences, kwargs, stream=True
        )

        try:
            response = await client.chat.completions.create(**create_kwargs)
            async for chunk in response:
                done = False
                for stream_chunk in self._chunk_to_stream_chunks(chunk):
                    done = done or stream_chunk.is_final
                    yield stream_chunk
                if done:
                    break
        except Exception as e:
            raise self._translate_error(e)

    def count_tokens(self, messages: Sequence[ChatMessage]) -> int:
        """Estimate tokens using tiktoken when available.

        Falls back to character-based heuristic.
        """
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(self.config.model)
            total = 0
            for msg in messages:
                total += 4  # message overhead
                total += len(encoding.encode(msg.content))
                total += len(encoding.encode(msg.role.value))
            total += 2  # reply priming
            return total
        except (ImportError, KeyError):
            total_chars = sum(len(m.content) for m in messages)
            return total_chars // 4

    def _translate_error(self, error: Exception) -> ProviderError:
        """Translate OpenAI-specific exceptions to Felix provider errors.

        Prefers the HTTP status code carried by the SDK's typed exceptions;
        falls back to message/type-name matching for plain exceptions.
        """
        error_str = str(error)
        error_type = type(error).__name__
        lowered = error_str.lower()
        status_code = extract_status_code(error)

        if status_code in (401, 403) or "authentication" in lowered or "api_key" in lowered:
            return AuthenticationError(error_str, provider="openai", status_code=status_code)
        if status_code == 429 or "rate_limit" in error_type.lower() or "429" in error_str:
            return RateLimitError(
                error_str,
                retry_after=extract_retry_after(error),
                provider="openai",
                status_code=status_code,
            )
        if "not_found" in error_type.lower():
            return ModelNotFoundError(error_str, provider="openai", status_code=status_code)
        if "context_length" in lowered or "maximum context" in lowered:
            return ContextLengthError(error_str, provider="openai", status_code=status_code)
        return ProviderError(error_str, provider="openai", status_code=status_code)


__all__ = ["OpenAIProvider"]
