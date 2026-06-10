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

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        provider: str = "",
        status_code: Optional[int] = None,
    ):
        self.retry_after = retry_after
        super().__init__(message, provider=provider, status_code=status_code)


class ModelNotFoundError(ProviderError):
    """Raised when the requested model is not available."""

    pass


class ContextLengthError(ProviderError):
    """Raised when the input exceeds the model's context window."""

    pass


def extract_status_code(error: Exception) -> Optional[int]:
    """Pull an HTTP status code off a vendor SDK exception, if present.

    Both the ``anthropic`` and ``openai`` SDKs expose ``status_code`` on
    their ``APIStatusError`` hierarchy.
    """
    code = getattr(error, "status_code", None)
    return code if isinstance(code, int) else None


def extract_retry_after(error: Exception) -> Optional[float]:
    """Pull a Retry-After value (seconds) off a vendor SDK exception.

    Vendor SDK status errors carry the httpx response; the header is
    optional and may be a date string, in which case None is returned.
    """
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("retry-after")
    except (AttributeError, TypeError):
        return None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotFoundError",
    "ContextLengthError",
    "extract_status_code",
    "extract_retry_after",
]
