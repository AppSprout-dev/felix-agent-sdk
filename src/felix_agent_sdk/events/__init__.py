"""Felix event system for SDK observability.

Provides a synchronous pub/sub event bus, typed events, and a mixin
for emitting events from any component.
"""

from felix_agent_sdk.events.bus import EventBus, EventCallback
from felix_agent_sdk.events.mixins import EventEmitterMixin
from felix_agent_sdk.events.types import EventType, FelixEvent

__all__ = [
    "EventBus",
    "EventCallback",
    "EventEmitterMixin",
    "EventType",
    "FelixEvent",
]
