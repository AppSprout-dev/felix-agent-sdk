"""Provider registry and auto-detection for Felix.

Handles runtime provider resolution from environment variables.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from .base import BaseProvider
from .errors import ProviderError
from .types import ProviderConfig  # noqa: F401 — re-exported

logger = logging.getLogger("felix.providers")


class ProviderRegistry:
    """Registry for discovering and instantiating providers.

    Supports both built-in providers and user-registered custom providers.
    Auto-detection reads from environment variables to determine which
    provider to instantiate.

    Environment variables:
        FELIX_PROVIDER: Provider name (anthropic, openai, local, bedrock, vertex)
        FELIX_MODEL: Model identifier
        ANTHROPIC_API_KEY: Anthropic API key
        OPENAI_API_KEY: OpenAI API key
        LOCAL_BASE_URL: Local server URL
    """

    _providers: Dict[str, type] = {}
    _detection_order: List[str] = []

    @classmethod
    def register(cls, name: str, provider_class: type, detection_priority: int = 100) -> None:
        """Register a provider class.

        Args:
            name: Provider identifier (e.g., 'anthropic', 'openai').
            provider_class: The provider class (must inherit from BaseProvider).
            detection_priority: Lower numbers are checked first during auto-detection.
        """
        if not issubclass(provider_class, BaseProvider):
            raise TypeError(f"{provider_class} must inherit from BaseProvider")
        cls._providers[name] = provider_class
        # Maintain sorted detection order
        cls._detection_order.append(name)
        cls._detection_order.sort(
            key=lambda n: detection_priority if n == name else 100
        )
        logger.debug(f"Registered provider: {name}")

    @classmethod
    def get(cls, name: str) -> type:
        """Get a provider class by name.

        Raises:
            KeyError: If the provider name is not registered.
        """
        if name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise KeyError(
                f"Unknown provider '{name}'. Available: {available}. "
                f"Register custom providers with ProviderRegistry.register()."
            )
        return cls._providers[name]

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered provider names."""
        return list(cls._providers.keys())

    @classmethod
    def auto_detect(cls) -> BaseProvider:
        """Auto-detect and instantiate a provider from environment variables.

        Detection priority:
        1. FELIX_PROVIDER env var (explicit override)
        2. Check for provider-specific API keys in priority order
        3. Check for local server availability

        Returns:
            An instantiated provider.

        Raises:
            ProviderError: If no provider can be detected.
        """
        # Explicit override
        explicit = os.getenv("FELIX_PROVIDER")
        if explicit:
            model = os.getenv("FELIX_MODEL")
            provider_class = cls.get(explicit)
            kwargs: Dict[str, Any] = {"model": model} if model else {}
            return provider_class(**kwargs)

        # Auto-detect from available keys
        detection_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "bedrock": "AWS_ACCESS_KEY_ID",
            "vertex": "GOOGLE_APPLICATION_CREDENTIALS",
        }

        for provider_name, env_var in detection_map.items():
            if os.getenv(env_var) and provider_name in cls._providers:
                model = os.getenv("FELIX_MODEL")
                provider_class = cls._providers[provider_name]
                kwargs = {"model": model} if model else {}
                logger.info(
                    f"Auto-detected provider '{provider_name}' from {env_var}"
                )
                return provider_class(**kwargs)

        # Fall back to local
        if "local" in cls._providers:
            logger.info("No cloud API keys found, falling back to local provider")
            model_name = os.getenv("FELIX_MODEL", "local-model")
            return cls._providers["local"](model=model_name)

        raise ProviderError(
            "No provider could be auto-detected. Set FELIX_PROVIDER or "
            "configure an API key (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)."
        )


def auto_detect_provider() -> BaseProvider:
    """Auto-detect and return a configured provider.

    Reads from environment variables to determine which LLM provider
    to use. See ProviderRegistry.auto_detect() for detection logic.

    Returns:
        A configured BaseProvider instance.

    Example:
        export ANTHROPIC_API_KEY=sk-ant-...
        export FELIX_MODEL=claude-sonnet-4-5

        >>> from felix_agent_sdk.providers import auto_detect_provider
        >>> provider = auto_detect_provider()
        >>> print(provider)
        AnthropicProvider(model='claude-sonnet-4-5')
    """
    return ProviderRegistry.auto_detect()


# ---------------------------------------------------------------------------
# Register built-in providers
# ---------------------------------------------------------------------------

from .anthropic import AnthropicProvider  # noqa: E402
from .local import LocalProvider  # noqa: E402
from .openai_provider import OpenAIProvider  # noqa: E402

ProviderRegistry.register("anthropic", AnthropicProvider, detection_priority=10)
ProviderRegistry.register("openai", OpenAIProvider, detection_priority=20)
ProviderRegistry.register("local", LocalProvider, detection_priority=90)


__all__ = [
    "ProviderRegistry",
    "auto_detect_provider",
]
