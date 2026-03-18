"""Unit tests for BaseProvider abstract class and interface contract."""

from __future__ import annotations

import pytest

from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.providers.types import (
    ChatMessage,
    CompletionResult,
    MessageRole,
    ProviderConfig,
    StreamChunk,
)


# ---------------------------------------------------------------------------
# Concrete subclass for testing BaseProvider's non-abstract behavior
# ---------------------------------------------------------------------------


class StubProvider(BaseProvider):
    """Minimal concrete provider for testing base class methods."""

    def __init__(self, config=None, complete_result=None):
        super().__init__(config or ProviderConfig(model="stub-model"))
        self._complete_result = complete_result

    def complete(self, messages, *, temperature=None, max_tokens=None,
                 stop_sequences=None, **kwargs):
        if self._complete_result is not None:
            return self._complete_result
        return CompletionResult(content="stub", model=self.config.model)

    def stream(self, messages, *, temperature=None, max_tokens=None,
               stop_sequences=None, **kwargs):
        yield StreamChunk(text="stub")
        yield StreamChunk(text="", is_final=True)

    def count_tokens(self, messages):
        return sum(len(m.content) for m in messages)


class FailingProvider(BaseProvider):
    """Provider whose complete() always raises."""

    def complete(self, messages, **kwargs):
        raise ConnectionError("server unreachable")

    def stream(self, messages, **kwargs):
        raise NotImplementedError

    def count_tokens(self, messages):
        return 0


# ---------------------------------------------------------------------------
# Abstract method enforcement
# ---------------------------------------------------------------------------


class TestBaseProviderAbstract:
    def test_cannot_instantiate_abstract_class(self):
        """BaseProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract method"):
            BaseProvider(ProviderConfig(model="x"))

    def test_must_implement_complete(self):
        """Subclass missing complete() cannot be instantiated."""

        class Incomplete(BaseProvider):
            def stream(self, messages, **kwargs):
                yield StreamChunk(text="")

            def count_tokens(self, messages):
                return 0

        with pytest.raises(TypeError):
            Incomplete(ProviderConfig(model="x"))

    def test_must_implement_stream(self):
        """Subclass missing stream() cannot be instantiated."""

        class Incomplete(BaseProvider):
            def complete(self, messages, **kwargs):
                return CompletionResult(content="", model="x")

            def count_tokens(self, messages):
                return 0

        with pytest.raises(TypeError):
            Incomplete(ProviderConfig(model="x"))

    def test_must_implement_count_tokens(self):
        """Subclass missing count_tokens() cannot be instantiated."""

        class Incomplete(BaseProvider):
            def complete(self, messages, **kwargs):
                return CompletionResult(content="", model="x")

            def stream(self, messages, **kwargs):
                yield StreamChunk(text="")

        with pytest.raises(TypeError):
            Incomplete(ProviderConfig(model="x"))


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProviderProperties:
    def test_model_property(self, default_config):
        p = StubProvider(default_config)
        assert p.model == "test-model"

    def test_provider_name_strips_provider_suffix(self):
        p = StubProvider()
        assert p.provider_name == "stub"

    def test_provider_name_lowercased(self):
        """provider_name should be lowercase."""

        class MyCustomProvider(BaseProvider):
            def complete(self, messages, **kwargs):
                return CompletionResult(content="", model="x")

            def stream(self, messages, **kwargs):
                yield StreamChunk(text="")

            def count_tokens(self, messages):
                return 0

        p = MyCustomProvider(ProviderConfig(model="m"))
        assert p.provider_name == "mycustom"

    def test_repr(self, default_config):
        p = StubProvider(default_config)
        assert repr(p) == "StubProvider(model='test-model')"


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------


class TestResolveHelpers:
    def test_resolve_temperature_explicit(self, default_config):
        p = StubProvider(default_config)
        assert p._resolve_temperature(0.3) == 0.3

    def test_resolve_temperature_none_uses_config(self, default_config):
        p = StubProvider(default_config)
        assert p._resolve_temperature(None) == default_config.default_temperature

    def test_resolve_temperature_zero_is_not_none(self, default_config):
        """0.0 is a valid explicit temperature, should not fall back to default."""
        p = StubProvider(default_config)
        assert p._resolve_temperature(0.0) == 0.0

    def test_resolve_max_tokens_explicit(self, default_config):
        p = StubProvider(default_config)
        assert p._resolve_max_tokens(500) == 500

    def test_resolve_max_tokens_none_uses_config(self, default_config):
        p = StubProvider(default_config)
        assert p._resolve_max_tokens(None) == default_config.default_max_tokens


# ---------------------------------------------------------------------------
# Default validate() behavior
# ---------------------------------------------------------------------------


class TestValidate:
    def test_validate_returns_true_on_success(self):
        result = CompletionResult(content="pong", model="stub")
        p = StubProvider(complete_result=result)
        assert p.validate() is True

    def test_validate_returns_false_on_empty_content(self):
        result = CompletionResult(content="", model="stub")
        p = StubProvider(complete_result=result)
        assert p.validate() is False

    def test_validate_returns_false_on_exception(self):
        p = FailingProvider(ProviderConfig(model="fail"))
        assert p.validate() is False


# ---------------------------------------------------------------------------
# Async stubs
# ---------------------------------------------------------------------------


class TestAsyncStubs:
    @pytest.mark.asyncio
    async def test_acomplete_raises_not_implemented(self):
        p = StubProvider()
        with pytest.raises(NotImplementedError, match="does not support async completions"):
            await p.acomplete([ChatMessage(role=MessageRole.USER, content="hi")])

    @pytest.mark.asyncio
    async def test_astream_raises_not_implemented(self):
        p = StubProvider()
        with pytest.raises(NotImplementedError, match="does not support async streaming"):
            async for _ in p.astream([ChatMessage(role=MessageRole.USER, content="hi")]):
                pass


# ---------------------------------------------------------------------------
# Lazy client init
# ---------------------------------------------------------------------------


class TestLazyClient:
    def test_client_initially_none(self):
        p = StubProvider()
        assert p._client is None
