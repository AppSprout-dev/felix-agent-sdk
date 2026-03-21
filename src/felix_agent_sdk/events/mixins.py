"""Mixin for classes that emit events onto an EventBus."""

from __future__ import annotations

from typing import Any, Dict, Optional

from felix_agent_sdk.events.bus import EventBus
from felix_agent_sdk.events.types import FelixEvent


class EventEmitterMixin:
    """Opt-in mixin that gives any class event-emission capability.

    When an :class:`EventBus` is attached via :meth:`set_event_bus`, calls
    to :meth:`emit_event` dispatch a :class:`FelixEvent` onto the bus.
    When no bus is attached, :meth:`emit_event` is a silent no-op —
    zero overhead for users who don't enable observability.
    """

    _event_bus: Optional[EventBus] = None

    def set_event_bus(self, bus: Optional[EventBus]) -> None:
        """Attach or detach an event bus."""
        self._event_bus = bus

    def emit_event(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        source: Optional[str] = None,
    ) -> None:
        """Emit a :class:`FelixEvent` if a bus is attached.

        Args:
            event_type: Event category (e.g. ``EventType.TASK_STARTED``).
            data: Arbitrary payload dict.
            source: Override the default source identifier.
        """
        if self._event_bus is None:
            return
        event = FelixEvent(
            event_type=event_type,
            source=source or self._default_event_source(),
            data=data or {},
        )
        self._event_bus.emit(event)

    def _default_event_source(self) -> str:
        """Return a default source string. Override for richer identification."""
        return type(self).__name__
