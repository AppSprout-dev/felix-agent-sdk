"""Tests for ProviderRegistry and auto_detect_provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.providers.errors import ProviderError
from felix_agent_sdk.providers.registry import ProviderRegistry, auto_detect_provider
from felix_agent_sdk.providers.types import CompletionResult, ProviderConfig, StreamChunk


# ---------------------------------------------------------------------------
# Concrete test provider
# ---------------------------------------------------------------------------


class _TestProvider(BaseProvider):
    def __init__(self, model="test", **kwargs):
        super().__init__(ProviderConfig(model=model))

    def complete(self, messages, **kw):
        return CompletionResult(content="", model=self.config.model)

    def stream(self, messages, **kw):
        yield StreamChunk(text="", is_final=True)

    def count_tokens(self, messages):
        return 0


class _NotAProvider:
    """Not a BaseProvider subclass — should be rejected."""
    pass


# ---------------------------------------------------------------------------
# Fixtures to isolate registry state
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Save and restore registry state around each test."""
    original_providers = ProviderRegistry._providers.copy()
    original_order = ProviderRegistry._detection_order.copy()
    yield
    ProviderRegistry._providers = original_providers
    ProviderRegistry._detection_order = original_order


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_valid_provider(self):
        ProviderRegistry.register("test", _TestProvider)
        assert "test" in ProviderRegistry._providers
        assert ProviderRegistry._providers["test"] is _TestProvider

    def test_register_rejects_non_provider(self):
        with pytest.raises(TypeError, match="must inherit from BaseProvider"):
            ProviderRegistry.register("bad", _NotAProvider)

    def test_register_overwrites_existing(self):
        ProviderRegistry.register("dup", _TestProvider, detection_priority=50)
        ProviderRegistry.register("dup", _TestProvider, detection_priority=50)
        assert ProviderRegistry._providers["dup"] is _TestProvider

    def test_register_adds_to_detection_order(self):
        ProviderRegistry.register("mytest", _TestProvider)
        assert "mytest" in ProviderRegistry._detection_order


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_registered_provider(self):
        ProviderRegistry.register("myp", _TestProvider)
        assert ProviderRegistry.get("myp") is _TestProvider

    def test_get_unknown_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown provider 'nonexistent'"):
            ProviderRegistry.get("nonexistent")

    def test_get_error_lists_available(self):
        ProviderRegistry.register("avail", _TestProvider)
        with pytest.raises(KeyError, match="avail"):
            ProviderRegistry.get("nonexistent")


# ---------------------------------------------------------------------------
# list_available()
# ---------------------------------------------------------------------------


class TestListAvailable:
    def test_lists_registered_providers(self):
        ProviderRegistry.register("p1", _TestProvider)
        ProviderRegistry.register("p2", _TestProvider)
        available = ProviderRegistry.list_available()
        assert "p1" in available
        assert "p2" in available

    def test_returns_list_type(self):
        assert isinstance(ProviderRegistry.list_available(), list)

    def test_builtin_providers_present(self):
        """After importing registry, anthropic/openai/local should be registered."""
        available = ProviderRegistry.list_available()
        assert "anthropic" in available
        assert "openai" in available
        assert "local" in available


# ---------------------------------------------------------------------------
# auto_detect() — environment variable priority
# ---------------------------------------------------------------------------


class TestAutoDetect:
    def test_felix_provider_explicit_override(self):
        """FELIX_PROVIDER takes highest priority."""
        ProviderRegistry.register("test", _TestProvider, detection_priority=50)
        with patch.dict("os.environ", {"FELIX_PROVIDER": "test"}, clear=True):
            provider = ProviderRegistry.auto_detect()
            assert isinstance(provider, _TestProvider)

    def test_felix_provider_with_model(self):
        ProviderRegistry.register("test", _TestProvider, detection_priority=50)
        with patch.dict(
            "os.environ",
            {"FELIX_PROVIDER": "test", "FELIX_MODEL": "custom-model"},
            clear=True,
        ):
            provider = ProviderRegistry.auto_detect()
            assert provider.model == "custom-model"

    def test_felix_provider_unknown_raises(self):
        with patch.dict(
            "os.environ", {"FELIX_PROVIDER": "nonexistent_xyz"}, clear=True
        ):
            with pytest.raises(KeyError):
                ProviderRegistry.auto_detect()

    def test_anthropic_key_detected_first(self):
        """ANTHROPIC_API_KEY should be checked before OPENAI_API_KEY."""
        with patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_KEY": "sk-ant-test",
                "OPENAI_API_KEY": "sk-test",
            },
            clear=True,
        ):
            provider = ProviderRegistry.auto_detect()
            from felix_agent_sdk.providers.anthropic import AnthropicProvider
            assert isinstance(provider, AnthropicProvider)

    def test_openai_key_detected(self):
        with patch.dict(
            "os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True
        ):
            provider = ProviderRegistry.auto_detect()
            from felix_agent_sdk.providers.openai_provider import OpenAIProvider
            assert isinstance(provider, OpenAIProvider)

    def test_local_fallback(self):
        """When no API keys are set, falls back to local provider."""
        with patch.dict("os.environ", {}, clear=True):
            provider = ProviderRegistry.auto_detect()
            from felix_agent_sdk.providers.local import LocalProvider
            assert isinstance(provider, LocalProvider)

    def test_local_fallback_uses_felix_model(self):
        with patch.dict("os.environ", {"FELIX_MODEL": "my-local-llm"}, clear=True):
            provider = ProviderRegistry.auto_detect()
            assert provider.model == "my-local-llm"

    def test_local_fallback_default_model(self):
        with patch.dict("os.environ", {}, clear=True):
            provider = ProviderRegistry.auto_detect()
            assert provider.model == "local-model"

    def test_no_provider_found_raises(self):
        """When local is not registered and no keys found, should raise."""
        # Remove all providers
        ProviderRegistry._providers.clear()
        ProviderRegistry._detection_order.clear()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ProviderError, match="No provider could be auto-detected"):
                ProviderRegistry.auto_detect()


# ---------------------------------------------------------------------------
# auto_detect_provider() convenience function
# ---------------------------------------------------------------------------


class TestAutoDetectFunction:
    def test_delegates_to_registry(self):
        with patch.dict("os.environ", {}, clear=True):
            provider = auto_detect_provider()
            from felix_agent_sdk.providers.local import LocalProvider
            assert isinstance(provider, LocalProvider)
