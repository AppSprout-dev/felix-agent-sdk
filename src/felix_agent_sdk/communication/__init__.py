"""Felix hub-spoke communication system.

All agent communication routes through CentralPost at O(N) complexity.
"""

from felix_agent_sdk.communication.central_post import (
    AgentLifecycleEvent,
    CentralPost,
    HubCapacityError,
)
from felix_agent_sdk.communication.messages import Message, MessageType
from felix_agent_sdk.communication.registry import AgentRegistry
from felix_agent_sdk.communication.spoke import Spoke, SpokeConnection, SpokeManager

__all__ = [
    "CentralPost",
    "AgentLifecycleEvent",
    "HubCapacityError",
    "Message",
    "MessageType",
    "AgentRegistry",
    "Spoke",
    "SpokeConnection",
    "SpokeManager",
]
