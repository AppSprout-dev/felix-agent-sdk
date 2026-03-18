"""Message types and structures for Felix hub-spoke communication.

Ported from CalebisGross/felix src/communication/message_types.py.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class MessageType(Enum):
    """Types of messages in the communication system."""

    # Core task lifecycle
    TASK_REQUEST = "task_request"
    TASK_ASSIGNMENT = "task_assignment"
    STATUS_UPDATE = "status_update"
    TASK_COMPLETE = "task_complete"
    ERROR_REPORT = "error_report"

    # Phase-aware coordination
    PHASE_ANNOUNCE = "phase_announce"
    CONVERGENCE_SIGNAL = "convergence_signal"
    COLLABORATION_REQUEST = "collaboration_request"
    SYNTHESIS_READY = "synthesis_ready"
    AGENT_QUERY = "agent_query"
    AGENT_DISCOVERY = "agent_discovery"

    # Feedback integration
    SYNTHESIS_FEEDBACK = "synthesis_feedback"
    CONTRIBUTION_EVALUATION = "contribution_evaluation"
    IMPROVEMENT_REQUEST = "improvement_request"


@dataclass
class Message:
    """Message structure for communication between agents and CentralPost."""

    sender_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    receiver_id: Optional[str] = None
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
