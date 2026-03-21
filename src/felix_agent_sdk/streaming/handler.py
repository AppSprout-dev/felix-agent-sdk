"""Stream handler base class and built-in implementations."""

from __future__ import annotations

from typing import Callable, Optional

from felix_agent_sdk.events.bus import EventBus
from felix_agent_sdk.events.types import EventType, FelixEvent
from felix_agent_sdk.streaming.types import StreamEvent, StreamEventType


class StreamHandler:
    """Base class for consuming stream events.

    Override the ``on_*`` methods to handle specific event types.
    Default implementations are no-ops.

    Examples::

        class PrintHandler(StreamHandler):
            def on_token(self, event: StreamEvent) -> None:
                print(event.content, end="", flush=True)

            def on_result(self, event: StreamEvent) -> None:
                print()  # newline after stream completes
    """

    def on_token(self, event: StreamEvent) -> None:
        """Called for each token/chunk received."""

    def on_result(self, event: StreamEvent) -> None:
        """Called when the stream completes with the full result."""

    def on_error(self, event: StreamEvent) -> None:
        """Called if an error occurs during streaming."""

    def dispatch(self, event: StreamEvent) -> None:
        """Route *event* to the appropriate ``on_*`` method."""
        if event.event_type == StreamEventType.TOKEN:
            self.on_token(event)
        elif event.event_type == StreamEventType.CHUNK:
            self.on_token(event)  # chunks are batched tokens
        elif event.event_type == StreamEventType.RESULT:
            self.on_result(event)
        elif event.event_type == StreamEventType.ERROR:
            self.on_error(event)


class CallbackStreamHandler(StreamHandler):
    """Stream handler backed by simple callables.

    Examples::

        handler = CallbackStreamHandler(
            on_token=lambda e: print(e.content, end=""),
            on_result=lambda e: print(f"\\nDone: {len(e.accumulated)} chars"),
        )
    """

    def __init__(
        self,
        on_token: Optional[Callable[[StreamEvent], None]] = None,
        on_result: Optional[Callable[[StreamEvent], None]] = None,
        on_error: Optional[Callable[[StreamEvent], None]] = None,
    ) -> None:
        self._on_token = on_token
        self._on_result = on_result
        self._on_error = on_error

    def on_token(self, event: StreamEvent) -> None:
        if self._on_token is not None:
            self._on_token(event)

    def on_result(self, event: StreamEvent) -> None:
        if self._on_result is not None:
            self._on_result(event)

    def on_error(self, event: StreamEvent) -> None:
        if self._on_error is not None:
            self._on_error(event)


class EventBusStreamHandler(StreamHandler):
    """Stream handler that re-emits stream events onto an EventBus."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    def on_token(self, event: StreamEvent) -> None:
        self._bus.emit(FelixEvent(
            event_type=EventType.STREAM_TOKEN,
            source=f"agent:{event.agent_id}",
            data={"content": event.content, "token_index": event.token_index},
        ))

    def on_result(self, event: StreamEvent) -> None:
        self._bus.emit(FelixEvent(
            event_type=EventType.STREAM_COMPLETED,
            source=f"agent:{event.agent_id}",
            data={
                "length": len(event.accumulated),
                "token_count": event.token_index,
                "usage": event.usage,
            },
        ))
