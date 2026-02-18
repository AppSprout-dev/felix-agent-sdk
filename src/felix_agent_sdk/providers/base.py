"""BaseProvider abstract class — the contract all Felix LLM providers must fulfill."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator, List, Optional, Sequence

from .errors import ProviderError  # noqa: F401 — used in docstrings
from .types import ChatMessage, CompletionResult, MessageRole, ProviderConfig, StreamChunk

logger = logging.getLogger("felix.providers")


class BaseProvider(ABC):
    """Abstract base class for all LLM providers.

    Every provider must implement three core methods:
    - complete(): Synchronous completion
    - stream(): Synchronous streaming completion
    - count_tokens(): Estimate token count for a message sequence

    Providers may optionally implement:
    - acomplete(): Async completion
    - astream(): Async streaming completion
    - validate(): Test connectivity and authentication
    """

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._client: Optional[Any] = None  # Lazy-initialized provider client

    @property
    def model(self) -> str:
        """The model identifier this provider is configured to use."""
        return self.config.model

    @property
    def provider_name(self) -> str:
        """Human-readable provider name (e.g., 'anthropic', 'openai')."""
        return self.__class__.__name__.replace("Provider", "").lower()

    # --- Core interface (required) ---

    @abstractmethod
    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Generate a completion for the given message sequence.

        Args:
            messages: Ordered sequence of chat messages.
            temperature: Sampling temperature. Falls back to config default.
            max_tokens: Maximum tokens to generate. Falls back to config default.
            stop_sequences: Optional sequences that halt generation.
            **kwargs: Provider-specific parameters passed through.

        Returns:
            CompletionResult with generated content and usage metadata.

        Raises:
            ProviderError: On API errors, authentication failures, rate limits.
        """
        ...

    @abstractmethod
    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Generate a streaming completion.

        Yields StreamChunk objects as tokens are generated. The final chunk
        has is_final=True and includes usage statistics.

        Args:
            messages: Ordered sequence of chat messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stop_sequences: Optional sequences that halt generation.
            **kwargs: Provider-specific parameters.

        Yields:
            StreamChunk objects with incremental text.
        """
        ...

    @abstractmethod
    def count_tokens(self, messages: Sequence[ChatMessage]) -> int:
        """Estimate the token count for a message sequence.

        This is used by the TokenBudgetManager to track and enforce budgets.
        Implementations should use the provider's native tokenizer when
        available, or fall back to a reasonable approximation.

        Args:
            messages: Message sequence to count.

        Returns:
            Estimated token count.
        """
        ...

    # --- Async interface (optional, default raises) ---

    async def acomplete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Async version of complete(). Override for true async support."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async completions. "
            "Use complete() or implement acomplete()."
        )

    async def astream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Async version of stream(). Override for true async support."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async streaming. "
            "Use stream() or implement astream()."
        )
        yield  # Make this a proper async generator for type checking

    # --- Lifecycle ---

    def validate(self) -> bool:
        """Test that the provider is correctly configured and reachable.

        Returns True if a test request succeeds, False otherwise.
        Implementations should make a lightweight API call (e.g., list models).
        """
        try:
            result = self.complete(
                [ChatMessage(role=MessageRole.USER, content="ping")],
                max_tokens=5,
            )
            return bool(result.content)
        except Exception as e:
            logger.warning(f"Provider validation failed for {self.provider_name}: {e}")
            return False

    def _resolve_temperature(self, temperature: Optional[float]) -> float:
        """Resolve temperature with fallback to config default."""
        return temperature if temperature is not None else self.config.default_temperature

    def _resolve_max_tokens(self, max_tokens: Optional[int]) -> int:
        """Resolve max_tokens with fallback to config default."""
        return max_tokens if max_tokens is not None else self.config.default_max_tokens

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model!r})"


__all__ = ["BaseProvider"]
