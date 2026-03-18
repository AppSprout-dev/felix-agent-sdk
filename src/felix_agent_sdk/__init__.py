"""Felix Agent SDK — Helical multi-agent orchestration for convergent collaboration.

Felix models multi-agent collaboration as movement along a helical geometry,
where agents start wide (broad exploration) and spiral inward toward consensus.

Quickstart:
    from felix_agent_sdk.providers import AnthropicProvider

    provider = AnthropicProvider(model="claude-sonnet-4-5")

Full documentation: https://github.com/AppSprout-dev/felix-agent-sdk
"""

from felix_agent_sdk._version import __version__
from felix_agent_sdk.agents import (
    Agent,
    AgentFactory,
    AgentState,
    AnalysisAgent,
    CriticAgent,
    LLMAgent,
    LLMResult,
    LLMTask,
    ResearchAgent,
)
from felix_agent_sdk.providers import (
    AnthropicProvider,
    BaseProvider,
    LocalProvider,
    OpenAIProvider,
    auto_detect_provider,
)
from felix_agent_sdk.tokens import TokenBudget

__all__ = [
    "__version__",
    # Providers
    "BaseProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "LocalProvider",
    "auto_detect_provider",
    # Agents
    "Agent",
    "AgentState",
    "LLMAgent",
    "LLMTask",
    "LLMResult",
    "ResearchAgent",
    "AnalysisAgent",
    "CriticAgent",
    "AgentFactory",
    # Tokens
    "TokenBudget",
]
