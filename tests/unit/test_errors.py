"""Tests for the provider error hierarchy."""

from __future__ import annotations

import pytest

from felix_agent_sdk.providers.errors import (
    AuthenticationError,
    ContextLengthError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)


# ---------------------------------------------------------------------------
# Inheritance chain
# ---------------------------------------------------------------------------


class TestInheritanceChain:
    def test_provider_error_is_exception(self):
        assert issubclass(ProviderError, Exception)

    def test_authentication_error_inherits_provider_error(self):
        assert issubclass(AuthenticationError, ProviderError)

    def test_rate_limit_error_inherits_provider_error(self):
        assert issubclass(RateLimitError, ProviderError)

    def test_model_not_found_inherits_provider_error(self):
        assert issubclass(ModelNotFoundError, ProviderError)

    def test_context_length_inherits_provider_error(self):
        assert issubclass(ContextLengthError, ProviderError)

    def test_all_catchable_as_provider_error(self):
        """All specific errors can be caught with except ProviderError."""
        for cls in (AuthenticationError, RateLimitError, ModelNotFoundError, ContextLengthError):
            err = cls("test")
            assert isinstance(err, ProviderError)

    def test_all_catchable_as_exception(self):
        for cls in (ProviderError, AuthenticationError, RateLimitError,
                    ModelNotFoundError, ContextLengthError):
            err = cls("test")
            assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# ProviderError attributes
# ---------------------------------------------------------------------------


class TestProviderErrorAttributes:
    def test_message_stored(self):
        err = ProviderError("something broke")
        assert str(err) == "something broke"

    def test_provider_attribute_default(self):
        err = ProviderError("msg")
        assert err.provider == ""

    def test_provider_attribute_set(self):
        err = ProviderError("msg", provider="anthropic")
        assert err.provider == "anthropic"

    def test_status_code_default_none(self):
        err = ProviderError("msg")
        assert err.status_code is None

    def test_status_code_set(self):
        err = ProviderError("msg", status_code=500)
        assert err.status_code == 500

    def test_provider_and_status_code(self):
        err = ProviderError("msg", provider="openai", status_code=429)
        assert err.provider == "openai"
        assert err.status_code == 429


# ---------------------------------------------------------------------------
# AuthenticationError
# ---------------------------------------------------------------------------


class TestAuthenticationError:
    def test_basic(self):
        err = AuthenticationError("bad key", provider="anthropic")
        assert str(err) == "bad key"
        assert err.provider == "anthropic"

    def test_has_status_code(self):
        err = AuthenticationError("bad key", provider="openai", status_code=401)
        assert err.status_code == 401


# ---------------------------------------------------------------------------
# RateLimitError
# ---------------------------------------------------------------------------


class TestRateLimitError:
    def test_retry_after_default_none(self):
        err = RateLimitError("slow down")
        assert err.retry_after is None

    def test_retry_after_set(self):
        err = RateLimitError("slow down", retry_after=30.0)
        assert err.retry_after == 30.0

    def test_retry_after_with_provider(self):
        err = RateLimitError("slow down", retry_after=5.0, provider="anthropic")
        assert err.retry_after == 5.0
        assert err.provider == "anthropic"

    def test_retry_after_int(self):
        err = RateLimitError("wait", retry_after=60)
        assert err.retry_after == 60

    def test_message_preserved(self):
        err = RateLimitError("Too many requests")
        assert str(err) == "Too many requests"

    def test_inherits_provider_error_attrs(self):
        err = RateLimitError("limit", provider="openai", status_code=429)
        assert err.provider == "openai"
        assert err.status_code == 429


# ---------------------------------------------------------------------------
# ModelNotFoundError
# ---------------------------------------------------------------------------


class TestModelNotFoundError:
    def test_basic(self):
        err = ModelNotFoundError("model 'gpt-5' not found", provider="openai")
        assert "gpt-5" in str(err)
        assert err.provider == "openai"

    def test_status_code(self):
        err = ModelNotFoundError("not found", status_code=404)
        assert err.status_code == 404


# ---------------------------------------------------------------------------
# ContextLengthError
# ---------------------------------------------------------------------------


class TestContextLengthError:
    def test_basic(self):
        err = ContextLengthError("too many tokens", provider="anthropic")
        assert "too many tokens" in str(err)
        assert err.provider == "anthropic"

    def test_status_code(self):
        err = ContextLengthError("exceeded", status_code=400)
        assert err.status_code == 400


# ---------------------------------------------------------------------------
# Exception handling patterns
# ---------------------------------------------------------------------------


class TestExceptionHandling:
    def test_catch_specific_before_generic(self):
        """Specific errors should be catchable before ProviderError."""
        err = RateLimitError("limit hit")
        caught_specific = False
        try:
            raise err
        except RateLimitError:
            caught_specific = True
        except ProviderError:
            caught_specific = False
        assert caught_specific

    def test_catch_generic_catches_all(self):
        """ProviderError handler catches all specific errors."""
        for cls in (AuthenticationError, RateLimitError, ModelNotFoundError, ContextLengthError):
            try:
                raise cls("test")
            except ProviderError:
                pass  # expected
            else:
                pytest.fail(f"{cls.__name__} not caught by ProviderError handler")
