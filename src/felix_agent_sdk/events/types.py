"""Event type definitions for the Felix observability system.

Provides a unified event model that all SDK components can emit into.
Events are frozen dataclasses — immutable records of things that happened.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class EventType(str, Enum):
    """All observable events in the Felix SDK.

    Naming convention: ``component.action`` with dotted hierarchy
    so subscribers can match by prefix (e.g. ``"agent.*"``).
    """

    # Agent lifecycle — SPAWNED/COMPLETED/FAILED bridged from CentralPost
    AGENT_SPAWNED = "agent.spawned"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    # TODO: emit from Agent.update_position() and should_process_at_checkpoint()
    AGENT_POSITION_UPDATED = "agent.position_updated"
    AGENT_CHECKPOINT = "agent.checkpoint"

    # Task processing — emitted from LLMAgent.process_task()
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"

    # Workflow lifecycle — emitted from FelixWorkflow.run()
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_ROUND_STARTED = "workflow.round.started"
    WORKFLOW_ROUND_COMPLETED = "workflow.round.completed"
    WORKFLOW_CONVERGED = "workflow.converged"
    WORKFLOW_SYNTHESIS_STARTED = "workflow.synthesis.started"
    WORKFLOW_COMPLETED = "workflow.completed"

    # Communication
    # TODO: emit from CentralPost.queue_message() and _handle_message()
    MESSAGE_QUEUED = "message.queued"
    MESSAGE_PROCESSED = "message.processed"

    # Streaming — wired in feat/streaming PR
    STREAM_TOKEN = "stream.token"
    STREAM_COMPLETED = "stream.completed"

    # Dynamic spawning — wired in feat/dynamic-spawning PR
    SPAWN_TRIGGERED = "spawn.triggered"
    SPAWN_COMPLETED = "spawn.completed"


@dataclass(frozen=True)
class FelixEvent:
    """An immutable record of something that happened in the SDK.

    Attributes:
        event_type: The event category (from :class:`EventType` or a custom string).
        source: Identifies the emitter, e.g. ``"workflow"`` or ``"agent:research-001"``.
        data: Arbitrary payload. Contents vary by event type.
        timestamp: Monotonic timestamp (seconds) when the event was created.
    """

    event_type: str
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)
