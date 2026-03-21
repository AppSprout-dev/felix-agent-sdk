"""Synchronous publish/subscribe event bus.

The ``EventBus`` is the central nervous system of Felix observability.
Components emit events; observers subscribe by exact type or prefix pattern.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Callable, Dict, List

from felix_agent_sdk.events.types import FelixEvent

logger = logging.getLogger(__name__)

# Type alias for event callbacks
EventCallback = Callable[[FelixEvent], None]


# Enough to capture a full workflow run (~20 rounds × ~10 agents × ~3 events)
# without unbounded growth in long-running processes.
_DEFAULT_MAX_HISTORY = 10_000


class EventBus:
    """Synchronous event bus with prefix-pattern subscriptions.

    Thread-safe. Callbacks are invoked inline during :meth:`emit`.
    If a callback raises, the exception is logged and the remaining
    callbacks still execute (same isolation pattern as
    ``CentralPost.emit_lifecycle_event``).

    Examples::

        bus = EventBus()

        # Exact match
        bus.subscribe("workflow.started", my_handler)

        # Prefix match — catches all agent.* events
        bus.subscribe("agent.*", my_agent_handler)

        # Catch-all
        bus.subscribe_all(my_logger)
    """

    def __init__(self) -> None:
        self._exact: Dict[str, List[EventCallback]] = defaultdict(list)
        self._prefix: Dict[str, List[EventCallback]] = defaultdict(list)
        self._catch_all: List[EventCallback] = []
        self._lock = threading.Lock()
        self._history: List[FelixEvent] = []
        self._record_history = False
        self._max_history_size = _DEFAULT_MAX_HISTORY

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def enable_history(self, enabled: bool = True, max_size: int = _DEFAULT_MAX_HISTORY) -> None:
        """Toggle event recording. When enabled, all emitted events are
        stored in :attr:`history` for later inspection.

        Args:
            enabled: Whether to record events.
            max_size: Maximum number of events to retain. Oldest events
                are discarded when the limit is reached.
        """
        with self._lock:
            self._record_history = enabled
            self._max_history_size = max_size
            if not enabled:
                self._history.clear()

    @property
    def history(self) -> List[FelixEvent]:
        """Read-only copy of recorded events (empty if history is disabled)."""
        with self._lock:
            return list(self._history)

    # ------------------------------------------------------------------
    # Subscribe / unsubscribe
    # ------------------------------------------------------------------

    def subscribe(self, event_type: str, callback: EventCallback) -> None:
        """Register *callback* for events matching *event_type*.

        If *event_type* ends with ``".*"``, the callback receives every
        event whose type starts with the prefix (e.g. ``"agent.*"``
        matches ``"agent.spawned"``, ``"agent.completed"``, etc.).
        """
        with self._lock:
            if event_type.endswith(".*"):
                prefix = event_type[:-1]  # "agent.*" → "agent."
                self._prefix[prefix].append(callback)
            else:
                self._exact[event_type].append(callback)

    def subscribe_all(self, callback: EventCallback) -> None:
        """Register *callback* for **every** event."""
        with self._lock:
            self._catch_all.append(callback)

    def unsubscribe(self, event_type: str, callback: EventCallback) -> None:
        """Remove *callback* from *event_type* subscriptions."""
        with self._lock:
            if event_type.endswith(".*"):
                prefix = event_type[:-1]
                cbs = self._prefix.get(prefix, [])
            else:
                cbs = self._exact.get(event_type, [])
            if callback in cbs:
                cbs.remove(callback)

    def unsubscribe_all(self, callback: EventCallback) -> None:
        """Remove *callback* from the catch-all list."""
        with self._lock:
            if callback in self._catch_all:
                self._catch_all.remove(callback)

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def emit(self, event: FelixEvent) -> None:
        """Dispatch *event* to all matching subscribers.

        Matching order: exact → prefix → catch-all.
        Exceptions in callbacks are logged and swallowed.
        """
        with self._lock:
            if self._record_history:
                self._history.append(event)
                if len(self._history) > self._max_history_size:
                    self._history = self._history[-self._max_history_size:]

            targets: List[EventCallback] = []

            # Exact match
            targets.extend(self._exact.get(event.event_type, []))

            # Prefix match
            for prefix, cbs in self._prefix.items():
                if event.event_type.startswith(prefix):
                    targets.extend(cbs)

            # Catch-all
            targets.extend(self._catch_all)

        # Invoke outside the lock to avoid deadlocks in callbacks
        for cb in targets:
            try:
                cb(event)
            except Exception:
                logger.exception(
                    "EventBus callback error for %s from %s",
                    event.event_type,
                    event.source,
                )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all subscriptions and recorded history."""
        with self._lock:
            self._exact.clear()
            self._prefix.clear()
            self._catch_all.clear()
            self._history.clear()

    @property
    def subscriber_count(self) -> int:
        """Total number of registered callbacks (exact + prefix + catch-all)."""
        with self._lock:
            exact_count = sum(len(cbs) for cbs in self._exact.values())
            prefix_count = sum(len(cbs) for cbs in self._prefix.values())
            return exact_count + prefix_count + len(self._catch_all)
