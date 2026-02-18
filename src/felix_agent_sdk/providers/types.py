"""Provider-agnostic data types for Felix's LLM abstraction layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class MessageRole(str, Enum):
    """Standard message roles across all providers."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ChatMessage:
    """Provider-agnostic chat message.

    All Felix agents communicate through this format. Provider implementations
    are responsible for translating to/from their native message types.
    """

    role: MessageRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompletionResult:
    """Standardized result returned by all providers.

    Attributes:
        content: The generated text content.
        model: Model identifier used for generation.
        usage: Token usage breakdown (prompt_tokens, completion_tokens, total_tokens).
        finish_reason: Why generation stopped (stop, length, etc.).
        raw_response: Provider-specific raw response for advanced use cases.
    """

    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    raw_response: Optional[Any] = None

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", self.prompt_tokens + self.completion_tokens)


@dataclass(frozen=True)
class StreamChunk:
    """A single chunk from a streaming response.

    Attributes:
        text: The incremental text content.
        is_final: Whether this is the last chunk in the stream.
        usage: Token usage (typically only populated on the final chunk).
    """

    text: str
    is_final: bool = False
    usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class ProviderConfig:
    """Configuration for a provider instance.

    Attributes:
        model: Model identifier (e.g., "claude-sonnet-4-5", "gpt-4o").
        api_key: API key for authentication. If None, reads from environment.
        base_url: Override the default API endpoint (useful for proxies, local servers).
        max_retries: Number of retries on transient failures.
        timeout: Request timeout in seconds.
        default_temperature: Default temperature for completions.
        default_max_tokens: Default max tokens for completions.
        extra: Provider-specific additional configuration.
    """

    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_retries: int = 3
    timeout: float = 120.0
    default_temperature: float = 0.7
    default_max_tokens: int = 1024
    extra: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "MessageRole",
    "ChatMessage",
    "CompletionResult",
    "StreamChunk",
    "ProviderConfig",
]
