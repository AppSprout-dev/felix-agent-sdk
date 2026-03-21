"""Streaming event types for real-time agent output observation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class StreamEventType(str, Enum):
    """Types of streaming events."""

    TOKEN = "token"
    CHUNK = "chunk"
    RESULT = "result"
    ERROR = "error"


@dataclass(frozen=True)
class StreamEvent:
    """A single event from a streaming agent response.

    Attributes:
        agent_id: The agent producing the stream.
        event_type: The kind of streaming event.
        content: Incremental text content.
        accumulated: Full text accumulated so far.
        token_index: Number of tokens/chunks received so far.
        is_final: Whether this is the last event in the stream.
        usage: Token usage stats (populated on final event).
        timestamp: Monotonic timestamp.
    """

    agent_id: str
    event_type: StreamEventType
    content: str
    accumulated: str = ""
    token_index: int = 0
    is_final: bool = False
    usage: Dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)
