"""Stream accumulator — bridges provider StreamChunks to SDK StreamEvents."""

from __future__ import annotations

from typing import Iterator

from felix_agent_sdk.providers.types import CompletionResult, StreamChunk
from felix_agent_sdk.streaming.handler import StreamHandler
from felix_agent_sdk.streaming.types import StreamEvent, StreamEventType


class StreamAccumulator:
    """Consume :class:`StreamChunk` objects from a provider and produce
    :class:`StreamEvent` objects dispatched to a :class:`StreamHandler`.

    After iteration completes, :meth:`to_completion_result` returns the
    equivalent :class:`CompletionResult` so the caller can treat the
    stream result identically to a non-streaming ``complete()`` call.

    Args:
        agent_id: Agent producing the stream.
        handler: Handler to dispatch events to.
        model: Model identifier for the completion result.
    """

    def __init__(self, agent_id: str, handler: StreamHandler, model: str = "") -> None:
        self._agent_id = agent_id
        self._handler = handler
        self._model = model
        self._accumulated = ""
        self._token_index = 0
        self._usage: dict[str, int] = {}

    def feed(self, chunk: StreamChunk) -> None:
        """Process a single :class:`StreamChunk`."""
        self._accumulated += chunk.text
        self._token_index += 1

        if chunk.is_final:
            self._usage = dict(chunk.usage)
            event = StreamEvent(
                agent_id=self._agent_id,
                event_type=StreamEventType.RESULT,
                content=chunk.text,
                accumulated=self._accumulated,
                token_index=self._token_index,
                is_final=True,
                usage=self._usage,
            )
            self._handler.dispatch(event)
        else:
            event = StreamEvent(
                agent_id=self._agent_id,
                event_type=StreamEventType.TOKEN,
                content=chunk.text,
                accumulated=self._accumulated,
                token_index=self._token_index,
                is_final=False,
            )
            self._handler.dispatch(event)

    def feed_all(self, chunks: Iterator[StreamChunk]) -> None:
        """Consume all chunks from an iterator."""
        for chunk in chunks:
            self.feed(chunk)

    def to_completion_result(self) -> CompletionResult:
        """Build a :class:`CompletionResult` from the accumulated stream."""
        return CompletionResult(
            content=self._accumulated,
            model=self._model,
            usage=self._usage,
        )

    @property
    def accumulated_text(self) -> str:
        """The full text accumulated so far."""
        return self._accumulated

    @property
    def token_count(self) -> int:
        """Number of chunks received."""
        return self._token_index
