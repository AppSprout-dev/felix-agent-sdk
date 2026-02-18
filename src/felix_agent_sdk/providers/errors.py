"""Provider error hierarchy for Felix's LLM abstraction layer."""

from __future__ import annotations

from typing import Optional


class ProviderError(Exception):
    """Base exception for provider-related errors."""

    def __init__(self, message: str, provider: str = "", status_code: Optional[int] = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(message)


class AuthenticationError(ProviderError):
    """Raised when API key is invalid or missing."""

    pass


class RateLimitError(ProviderError):
    """Raised when the provider's rate limit is hit.

    Attributes:
        retry_after: Suggested wait time in seconds before retrying.
    """

    def __init__(self, message: str, retry_after: Optional[float] = None, **kwargs):  # type: ignore[override]
        self.retry_after = retry_after
        super().__init__(message, **kwargs)


class ModelNotFoundError(ProviderError):
    """Raised when the requested model is not available."""

    pass


class ContextLengthError(ProviderError):
    """Raised when the input exceeds the model's context window."""

    pass


__all__ = [
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotFoundError",
    "ContextLengthError",
]
