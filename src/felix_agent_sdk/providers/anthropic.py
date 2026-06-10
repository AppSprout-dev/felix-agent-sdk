"""Anthropic Claude provider for Felix.

Requires: pip install felix-agent-sdk[anthropic]
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence, Tuple

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
from .types import ChatMessage, CompletionResult, MessageRole, ProviderConfig, StreamChunk


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude models.

    Requires: pip install felix-agent-sdk[anthropic]

    Supported models: claude-fable-5, claude-opus-4-8, claude-sonnet-4-6,
    claude-haiku-4-5, and all previous Claude model versions.

    Note: Fable-class and Opus 4.7+ models reject sampling parameters
    (temperature/top_p/top_k) with HTTP 400; for those models the provider
    omits temperature entirely and helix temperature adaptation has no effect.

    Configuration:
        - api_key: Set via constructor or ANTHROPIC_API_KEY env var.
        - base_url: Defaults to Anthropic's API. Override for proxies.
    """

    # Model families that reject sampling parameters with HTTP 400.
    _SAMPLING_UNSUPPORTED_PREFIXES = (
        "claude-fable",
        "claude-opus-4-7",
        "claude-opus-4-8",
    )

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        config = ProviderConfig(
            model=model,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url,
            **kwargs,
        )
        super().__init__(config)

    def _import_sdk(self) -> Any:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' package. "
                "Install with: pip install felix-agent-sdk[anthropic]"
            )
        return anthropic

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
        """Lazy-initialize the Anthropic client."""
        if self._client is None:
            anthropic = self._import_sdk()
            self._client = anthropic.Anthropic(**self._client_kwargs())
        return self._client

    def _get_async_client(self) -> Any:
        """Lazy-initialize the async Anthropic client."""
        if self._async_client is None:
            anthropic = self._import_sdk()
            self._async_client = anthropic.AsyncAnthropic(**self._client_kwargs())
        return self._async_client

    def _supports_sampling_params(self) -> bool:
        """Whether the configured model accepts temperature/top_p/top_k."""
        return not self.config.model.startswith(self._SAMPLING_UNSUPPORTED_PREFIXES)

    def _format_messages(
        self, messages: Sequence[ChatMessage]
    ) -> Tuple[Optional[str], List[Dict[str, str]]]:
        """Convert ChatMessages to Anthropic's format.

        Anthropic uses a separate 'system' parameter rather than a system message
        in the messages list, so we extract it here.
        """
        system_content: Optional[str] = None
        api_messages: List[Dict[str, str]] = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_content = msg.content
            else:
                api_messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        return system_content, api_messages

    def _build_create_kwargs(
        self,
        messages: Sequence[ChatMessage],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stop_sequences: Optional[List[str]],
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assemble the keyword arguments for a messages.create/stream call."""
        system_content, api_messages = self._format_messages(messages)
        create_kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": api_messages,
            "max_tokens": self._resolve_max_tokens(max_tokens),
        }
        if self._supports_sampling_params():
            create_kwargs["temperature"] = self._resolve_temperature(temperature)
        if system_content:
            create_kwargs["system"] = system_content
        if stop_sequences:
            create_kwargs["stop_sequences"] = stop_sequences
        create_kwargs.update(extra)
        return create_kwargs

    @staticmethod
    def _to_completion_result(response: Any) -> CompletionResult:
        content = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        return CompletionResult(
            content=content,
            model=response.model,
            usage=AnthropicProvider._usage_dict(response),
            finish_reason=response.stop_reason or "stop",
            raw_response=response,
        )

    @staticmethod
    def _usage_dict(response: Any) -> Dict[str, int]:
        return {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

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
            response = client.messages.create(**create_kwargs)
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
            response = await client.messages.create(**create_kwargs)
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
            messages, temperature, max_tokens, stop_sequences, kwargs
        )

        try:
            with client.messages.stream(**create_kwargs) as stream:
                for text in stream.text_stream:
                    yield StreamChunk(text=text)
                # Final chunk with usage
                final = stream.get_final_message()
                yield StreamChunk(text="", is_final=True, usage=self._usage_dict(final))
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
            messages, temperature, max_tokens, stop_sequences, kwargs
        )

        try:
            async with client.messages.stream(**create_kwargs) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(text=text)
                # Final chunk with usage
                final = await stream.get_final_message()
                yield StreamChunk(text="", is_final=True, usage=self._usage_dict(final))
        except Exception as e:
            raise self._translate_error(e)

    def count_tokens(self, messages: Sequence[ChatMessage]) -> int:
        """Count tokens using Anthropic's token counting API when available.

        Falls back to a character-based heuristic (1 token ~ 4 characters).
        """
        try:
            client = self._get_client()
            _, api_messages = self._format_messages(messages)
            response = client.messages.count_tokens(
                model=self.config.model,
                messages=api_messages,
            )
            return int(response.input_tokens)
        except Exception:
            # Fallback: rough approximation
            total_chars = sum(len(m.content) for m in messages)
            return total_chars // 4

    def _translate_error(self, error: Exception) -> ProviderError:
        """Translate Anthropic-specific exceptions to Felix provider errors.

        Prefers the HTTP status code carried by the SDK's typed exceptions;
        falls back to message/type-name matching for plain exceptions.
        """
        error_str = str(error)
        error_type = type(error).__name__
        lowered = error_str.lower()
        status_code = extract_status_code(error)

        if status_code in (401, 403) or "authentication" in lowered or "api_key" in lowered:
            return AuthenticationError(error_str, provider="anthropic", status_code=status_code)
        if status_code == 429 or "rate_limit" in error_type.lower() or "429" in error_str:
            return RateLimitError(
                error_str,
                retry_after=extract_retry_after(error),
                provider="anthropic",
                status_code=status_code,
            )
        if "context" in lowered or "too long" in lowered:
            return ContextLengthError(error_str, provider="anthropic", status_code=status_code)
        # Bare "model" in the message is not enough: param-rejection 400s
        # ("temperature is not supported on this model") must stay generic.
        if (
            status_code == 404
            or "notfound" in error_type.lower().replace("_", "")
            or (
                "model" in lowered
                and any(
                    phrase in lowered
                    for phrase in ("not exist", "not found", "not available", "unknown")
                )
            )
        ):
            return ModelNotFoundError(error_str, provider="anthropic", status_code=status_code)
        return ProviderError(error_str, provider="anthropic", status_code=status_code)


__all__ = ["AnthropicProvider"]
