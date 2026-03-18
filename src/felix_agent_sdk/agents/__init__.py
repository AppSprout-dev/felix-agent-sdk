"""Felix agent system.

Position-aware agents that traverse the helical geometry. Temperature,
token budget, and prompting strategy adapt based on helix position.
"""

from felix_agent_sdk.agents.base import Agent, AgentState
from felix_agent_sdk.agents.factory import AgentFactory
from felix_agent_sdk.agents.llm_agent import LLMAgent, LLMResult, LLMTask
from felix_agent_sdk.agents.specialized import AnalysisAgent, CriticAgent, ResearchAgent

__all__ = [
    "Agent",
    "AgentState",
    "LLMAgent",
    "LLMTask",
    "LLMResult",
    "ResearchAgent",
    "AnalysisAgent",
    "CriticAgent",
    "AgentFactory",
]
