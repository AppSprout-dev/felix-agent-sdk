"""Felix provider abstraction layer.

Exposes all built-in LLM providers, the base class, data types,
error hierarchy, and the auto-detection registry.

Usage:
    from felix_agent_sdk.providers import AnthropicProvider
    from felix_agent_sdk.providers import auto_detect_provider

    provider = AnthropicProvider(model="claude-sonnet-4-5")
    # or
    provider = auto_detect_provider()
"""

from felix_agent_sdk.providers.types import (
    ChatMessage,
    CompletionResult,
    MessageRole,
    ProviderConfig,
    StreamChunk,
)
from felix_agent_sdk.providers.errors import (
    AuthenticationError,
    ContextLengthError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)
from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.providers.anthropic import AnthropicProvider
from felix_agent_sdk.providers.openai_provider import OpenAIProvider
from felix_agent_sdk.providers.local import LocalProvider
from felix_agent_sdk.providers.registry import ProviderRegistry, auto_detect_provider

__all__ = [
    # Base
    "BaseProvider",
    "ProviderConfig",
    "ChatMessage",
    "MessageRole",
    "CompletionResult",
    "StreamChunk",
    # Providers
    "AnthropicProvider",
    "OpenAIProvider",
    "LocalProvider",
    # Registry
    "ProviderRegistry",
    "auto_detect_provider",
    # Errors
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotFoundError",
    "ContextLengthError",
]
