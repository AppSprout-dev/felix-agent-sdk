"""Anthropic Claude provider for Felix.

Requires: pip install felix-agent-sdk[anthropic]
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .base import BaseProvider
from .errors import (
    AuthenticationError,
    ContextLengthError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)
from .types import ChatMessage, CompletionResult, MessageRole, ProviderConfig, StreamChunk


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude models.

    Requires: pip install felix-agent-sdk[anthropic]

    Supported models: claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5,
    and all previous Claude model versions.

    Configuration:
        - api_key: Set via constructor or ANTHROPIC_API_KEY env var.
        - base_url: Defaults to Anthropic's API. Override for proxies.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
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

    def _get_client(self):
        """Lazy-initialize the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "Anthropic provider requires the 'anthropic' package. "
                    "Install with: pip install felix-agent-sdk[anthropic]"
                )
            client_kwargs = {}
            if self.config.api_key:
                client_kwargs["api_key"] = self.config.api_key
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            client_kwargs["max_retries"] = self.config.max_retries
            client_kwargs["timeout"] = self.config.timeout
            self._client = anthropic.Anthropic(**client_kwargs)
        return self._client

    def _format_messages(self, messages: Sequence[ChatMessage]):
        """Convert ChatMessages to Anthropic's format.

        Anthropic uses a separate 'system' parameter rather than a system message
        in the messages list, so we extract it here.
        """
        system_content = None
        api_messages = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_content = msg.content
            else:
                api_messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        return system_content, api_messages

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
        system_content, api_messages = self._format_messages(messages)

        create_kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": api_messages,
            "temperature": self._resolve_temperature(temperature),
            "max_tokens": self._resolve_max_tokens(max_tokens),
        }
        if system_content:
            create_kwargs["system"] = system_content
        if stop_sequences:
            create_kwargs["stop_sequences"] = stop_sequences
        create_kwargs.update(kwargs)

        try:
            response = client.messages.create(**create_kwargs)
            content = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            return CompletionResult(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                finish_reason=response.stop_reason or "stop",
                raw_response=response,
            )
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
        system_content, api_messages = self._format_messages(messages)

        create_kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": api_messages,
            "temperature": self._resolve_temperature(temperature),
            "max_tokens": self._resolve_max_tokens(max_tokens),
        }
        if system_content:
            create_kwargs["system"] = system_content
        if stop_sequences:
            create_kwargs["stop_sequences"] = stop_sequences
        create_kwargs.update(kwargs)

        try:
            with client.messages.stream(**create_kwargs) as stream:
                for text in stream.text_stream:
                    yield StreamChunk(text=text)
                # Final chunk with usage
                final = stream.get_final_message()
                yield StreamChunk(
                    text="",
                    is_final=True,
                    usage={
                        "prompt_tokens": final.usage.input_tokens,
                        "completion_tokens": final.usage.output_tokens,
                        "total_tokens": final.usage.input_tokens + final.usage.output_tokens,
                    },
                )
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
            return response.input_tokens
        except Exception:
            # Fallback: rough approximation
            total_chars = sum(len(m.content) for m in messages)
            return total_chars // 4

    def _translate_error(self, error: Exception) -> ProviderError:
        """Translate Anthropic-specific exceptions to Felix provider errors."""
        error_str = str(error)
        error_type = type(error).__name__

        if "authentication" in error_str.lower() or "api_key" in error_str.lower():
            return AuthenticationError(error_str, provider="anthropic")
        if "rate_limit" in error_type.lower() or "429" in error_str:
            return RateLimitError(error_str, provider="anthropic")
        if "not_found" in error_type.lower() or "model" in error_str.lower():
            return ModelNotFoundError(error_str, provider="anthropic")
        if "context" in error_str.lower() or "too long" in error_str.lower():
            return ContextLengthError(error_str, provider="anthropic")
        return ProviderError(error_str, provider="anthropic")


__all__ = ["AnthropicProvider"]
