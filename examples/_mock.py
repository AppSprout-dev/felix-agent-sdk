"""Shared mock provider for examples that run without an API key."""

from unittest.mock import MagicMock

from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.providers.types import CompletionResult

DEFAULT_RESPONSES = [
    "Research indicates renewable energy adoption has accelerated globally, "
    "with solar capacity growing 40% year-over-year in key markets.",
    "Analysis of the data reveals a strong correlation between distributed "
    "generation and improved grid resilience across multiple regions.",
    "Critical review confirms the findings are well-supported. However, "
    "intermittency challenges require further attention to storage solutions.",
    "Synthesising all findings: the evidence strongly supports accelerating "
    "renewable energy deployment, with storage as the key enabling technology.",
]


def make_mock_provider(responses: list[str] | None = None) -> BaseProvider:
    """Create a mock provider that cycles through canned responses.

    Args:
        responses: Custom response strings. Uses DEFAULT_RESPONSES if None.
    """
    responses = responses or DEFAULT_RESPONSES
    provider = MagicMock(spec=BaseProvider)
    call_count = [0]

    def _complete(messages, **kwargs):
        idx = call_count[0] % len(responses)
        call_count[0] += 1
        return CompletionResult(
            content=responses[idx],
            model="mock-model",
            usage={"prompt_tokens": 50, "completion_tokens": 40, "total_tokens": 90},
        )

    provider.complete.side_effect = _complete
    provider.count_tokens.return_value = 50
    return provider
