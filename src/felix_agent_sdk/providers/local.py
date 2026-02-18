"""Local model provider for LM Studio, Ollama, vLLM, and any OpenAI-compatible server.

Requires: pip install felix-agent-sdk[local]
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

from .openai_provider import OpenAIProvider
from .types import ChatMessage

logger = logging.getLogger("felix.providers")


class LocalProvider(OpenAIProvider):
    """Provider for locally-hosted models via OpenAI-compatible APIs.

    Supports LM Studio, Ollama, vLLM, text-generation-inference, and any
    server that exposes an OpenAI-compatible /v1/chat/completions endpoint.

    Requires: pip install felix-agent-sdk[local]

    Configuration:
        - base_url: The local server URL (default: http://localhost:1234/v1).
        - model: The model name as loaded on the server.
        - api_key: Usually "lm-studio" or "ollama" (not validated locally).
    """

    DEFAULT_BASE_URL = "http://localhost:1234/v1"

    def __init__(
        self,
        model: str = "local-model",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            api_key=api_key or os.getenv("LOCAL_API_KEY", "lm-studio"),
            base_url=base_url or os.getenv("LOCAL_BASE_URL", self.DEFAULT_BASE_URL),
            **kwargs,
        )

    @property
    def provider_name(self) -> str:
        return "local"

    def validate(self) -> bool:
        """Validate by checking if the local server is reachable."""
        try:
            client = self._get_client()
            client.models.list()
            return True
        except Exception as e:
            logger.warning(f"Local provider validation failed: {e}")
            return False

    def count_tokens(self, messages: Sequence[ChatMessage]) -> int:
        """Local models rarely support token counting APIs.

        Uses character-based approximation: 1 token ~ 4 characters.
        """
        total_chars = sum(len(m.content) for m in messages)
        return total_chars // 4


__all__ = ["LocalProvider"]
