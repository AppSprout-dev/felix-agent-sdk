"""Tests for the streaming module — types, handlers, and accumulator."""

from __future__ import annotations

import pytest

from felix_agent_sdk.events import EventBus, EventType
from felix_agent_sdk.providers.types import StreamChunk
from felix_agent_sdk.streaming import (
    CallbackStreamHandler,
    EventBusStreamHandler,
    StreamAccumulator,
    StreamEvent,
    StreamEventType,
    StreamHandler,
)


# ---------------------------------------------------------------------------
# StreamEvent
# ---------------------------------------------------------------------------


class TestStreamEvent:
    def test_construction(self):
        evt = StreamEvent(
            agent_id="research-001",
            event_type=StreamEventType.TOKEN,
            content="hello",
        )
        assert evt.agent_id == "research-001"
        assert evt.event_type == StreamEventType.TOKEN
        assert evt.content == "hello"
        assert evt.accumulated == ""
        assert evt.token_index == 0
        assert evt.is_final is False
        assert isinstance(evt.timestamp, float)

    def test_frozen(self):
        evt = StreamEvent(agent_id="a", event_type=StreamEventType.TOKEN, content="x")
        with pytest.raises(AttributeError):
            evt.content = "y"  # type: ignore[misc]


class TestStreamEventType:
    def test_values(self):
        assert StreamEventType.TOKEN.value == "token"
        assert StreamEventType.CHUNK.value == "chunk"
        assert StreamEventType.RESULT.value == "result"
        assert StreamEventType.ERROR.value == "error"


# ---------------------------------------------------------------------------
# StreamHandler
# ---------------------------------------------------------------------------


class TestStreamHandler:
    def test_base_is_noop(self):
        handler = StreamHandler()
        evt = StreamEvent(agent_id="a", event_type=StreamEventType.TOKEN, content="x")
        # Should not raise
        handler.on_token(evt)
        handler.on_result(evt)
        handler.on_error(evt)

    def test_dispatch_routes_token(self):
        received = []

        class MyHandler(StreamHandler):
            def on_token(self, event: StreamEvent) -> None:
                received.append(("token", event.content))

        handler = MyHandler()
        handler.dispatch(StreamEvent(
            agent_id="a", event_type=StreamEventType.TOKEN, content="hello"
        ))
        assert received == [("token", "hello")]

    def test_dispatch_routes_chunk_to_on_token(self):
        received = []

        class MyHandler(StreamHandler):
            def on_token(self, event: StreamEvent) -> None:
                received.append(event.content)

        handler = MyHandler()
        handler.dispatch(StreamEvent(
            agent_id="a", event_type=StreamEventType.CHUNK, content="batch"
        ))
        assert received == ["batch"]

    def test_dispatch_routes_result(self):
        received = []

        class MyHandler(StreamHandler):
            def on_result(self, event: StreamEvent) -> None:
                received.append(event.accumulated)

        handler = MyHandler()
        handler.dispatch(StreamEvent(
            agent_id="a", event_type=StreamEventType.RESULT,
            content="last", accumulated="full text", is_final=True,
        ))
        assert received == ["full text"]

    def test_dispatch_routes_error(self):
        received = []

        class MyHandler(StreamHandler):
            def on_error(self, event: StreamEvent) -> None:
                received.append(event.content)

        handler = MyHandler()
        handler.dispatch(StreamEvent(
            agent_id="a", event_type=StreamEventType.ERROR, content="boom"
        ))
        assert received == ["boom"]


# ---------------------------------------------------------------------------
# CallbackStreamHandler
# ---------------------------------------------------------------------------


class TestCallbackStreamHandler:
    def test_token_callback(self):
        tokens = []
        handler = CallbackStreamHandler(on_token=lambda e: tokens.append(e.content))
        handler.on_token(StreamEvent(
            agent_id="a", event_type=StreamEventType.TOKEN, content="hi"
        ))
        assert tokens == ["hi"]

    def test_result_callback(self):
        results = []
        handler = CallbackStreamHandler(on_result=lambda e: results.append(e.accumulated))
        handler.on_result(StreamEvent(
            agent_id="a", event_type=StreamEventType.RESULT,
            content="", accumulated="done", is_final=True,
        ))
        assert results == ["done"]

    def test_none_callbacks_are_noop(self):
        handler = CallbackStreamHandler()
        # Should not raise
        handler.on_token(StreamEvent(agent_id="a", event_type=StreamEventType.TOKEN, content="x"))
        handler.on_result(StreamEvent(agent_id="a", event_type=StreamEventType.RESULT, content="x"))
        handler.on_error(StreamEvent(agent_id="a", event_type=StreamEventType.ERROR, content="x"))


# ---------------------------------------------------------------------------
# EventBusStreamHandler
# ---------------------------------------------------------------------------


class TestEventBusStreamHandler:
    def test_token_emits_to_bus(self):
        bus = EventBus()
        bus.enable_history()
        handler = EventBusStreamHandler(bus)

        handler.on_token(StreamEvent(
            agent_id="research-001", event_type=StreamEventType.TOKEN,
            content="hello", token_index=1,
        ))

        assert len(bus.history) == 1
        assert bus.history[0].event_type == "stream.token"
        assert bus.history[0].data["content"] == "hello"

    def test_result_emits_to_bus(self):
        bus = EventBus()
        bus.enable_history()
        handler = EventBusStreamHandler(bus)

        handler.on_result(StreamEvent(
            agent_id="research-001", event_type=StreamEventType.RESULT,
            content="last", accumulated="hello world", token_index=5,
            is_final=True, usage={"total_tokens": 50},
        ))

        assert len(bus.history) == 1
        assert bus.history[0].event_type == "stream.completed"
        assert bus.history[0].data["length"] == 11
        assert bus.history[0].data["token_count"] == 5


# ---------------------------------------------------------------------------
# StreamAccumulator
# ---------------------------------------------------------------------------


class TestStreamAccumulator:
    def test_accumulates_text(self):
        received = []
        handler = CallbackStreamHandler(on_token=lambda e: received.append(e.content))

        acc = StreamAccumulator("agent-001", handler, model="test-model")
        acc.feed(StreamChunk(text="Hello "))
        acc.feed(StreamChunk(text="world"))
        acc.feed(StreamChunk(text="!", is_final=True, usage={"total_tokens": 10}))

        assert acc.accumulated_text == "Hello world!"
        assert acc.token_count == 3
        # 2 token events + 1 result event dispatched through handler
        assert received == ["Hello ", "world"]

    def test_to_completion_result(self):
        handler = StreamHandler()  # no-op
        acc = StreamAccumulator("agent-001", handler, model="gpt-4o")

        acc.feed(StreamChunk(text="abc"))
        acc.feed(StreamChunk(text="def", is_final=True, usage={"total_tokens": 5}))

        result = acc.to_completion_result()
        assert result.content == "abcdef"
        assert result.model == "gpt-4o"
        assert result.usage == {"total_tokens": 5}

    def test_feed_all(self):
        handler = StreamHandler()
        acc = StreamAccumulator("agent-001", handler)

        chunks = [
            StreamChunk(text="a"),
            StreamChunk(text="b"),
            StreamChunk(text="c", is_final=True, usage={"total_tokens": 3}),
        ]
        acc.feed_all(iter(chunks))

        assert acc.accumulated_text == "abc"
        assert acc.token_count == 3

    def test_result_event_dispatched_on_final(self):
        results = []
        handler = CallbackStreamHandler(on_result=lambda e: results.append(e))

        acc = StreamAccumulator("agent-001", handler)
        acc.feed(StreamChunk(text="hello"))
        assert len(results) == 0

        acc.feed(StreamChunk(text="!", is_final=True, usage={"total_tokens": 2}))
        assert len(results) == 1
        assert results[0].is_final is True
        assert results[0].accumulated == "hello!"
        assert results[0].usage == {"total_tokens": 2}
