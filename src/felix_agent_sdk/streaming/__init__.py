"""Felix real-time output streaming.

Stream event types, handlers, and accumulator for token-by-token output
observation during agent processing.
"""

from felix_agent_sdk.streaming.accumulator import StreamAccumulator
from felix_agent_sdk.streaming.handler import (
    CallbackStreamHandler,
    EventBusStreamHandler,
    StreamHandler,
)
from felix_agent_sdk.streaming.types import StreamEvent, StreamEventType

__all__ = [
    "StreamAccumulator",
    "StreamHandler",
    "CallbackStreamHandler",
    "EventBusStreamHandler",
    "StreamEvent",
    "StreamEventType",
]
