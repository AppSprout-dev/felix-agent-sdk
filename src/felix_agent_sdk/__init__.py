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
from felix_agent_sdk.communication import CentralPost, Message, MessageType, Spoke
from felix_agent_sdk.events import EventBus, EventType, FelixEvent
from felix_agent_sdk.memory import ContextCompressor, KnowledgeStore, TaskMemory
from felix_agent_sdk.spawning import ConfidenceMonitor, DynamicSpawner
from felix_agent_sdk.streaming import StreamEvent, StreamHandler
from felix_agent_sdk.tokens import TokenBudget
from felix_agent_sdk.utils import configure_logging
from felix_agent_sdk.workflows import FelixWorkflow, WorkflowConfig, run_felix_workflow
from felix_agent_sdk.visualization import HelixVisualizer

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
    # Communication
    "CentralPost",
    "Message",
    "MessageType",
    "Spoke",
    # Memory
    "KnowledgeStore",
    "TaskMemory",
    "ContextCompressor",
    # Workflows
    "FelixWorkflow",
    "run_felix_workflow",
    "WorkflowConfig",
    # Events
    "EventBus",
    "EventType",
    "FelixEvent",
    # Spawning
    "DynamicSpawner",
    "ConfidenceMonitor",
    # Streaming
    "StreamEvent",
    "StreamHandler",
    # Logging
    "configure_logging",
    # Tokens
    "TokenBudget",
    # Visualization
    "HelixVisualizer",
]
