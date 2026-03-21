"""Tests for the Felix event system — bus, types, and mixins."""

from __future__ import annotations

import threading

import pytest

from felix_agent_sdk.events import EventBus, EventType, FelixEvent
from felix_agent_sdk.events.mixins import EventEmitterMixin


# ---------------------------------------------------------------------------
# FelixEvent
# ---------------------------------------------------------------------------


class TestFelixEvent:
    def test_construction(self):
        evt = FelixEvent(event_type="test.event", source="test")
        assert evt.event_type == "test.event"
        assert evt.source == "test"
        assert evt.data == {}
        assert isinstance(evt.timestamp, float)

    def test_frozen(self):
        evt = FelixEvent(event_type="x", source="y")
        with pytest.raises(AttributeError):
            evt.event_type = "z"  # type: ignore[misc]

    def test_with_data(self):
        evt = FelixEvent(event_type="x", source="y", data={"key": "val"})
        assert evt.data == {"key": "val"}


# ---------------------------------------------------------------------------
# EventType enum
# ---------------------------------------------------------------------------


class TestEventType:
    def test_agent_events(self):
        assert EventType.AGENT_SPAWNED.value == "agent.spawned"
        assert EventType.AGENT_COMPLETED.value == "agent.completed"
        assert EventType.AGENT_FAILED.value == "agent.failed"

    def test_workflow_events(self):
        assert EventType.WORKFLOW_STARTED.value == "workflow.started"
        assert EventType.WORKFLOW_ROUND_STARTED.value == "workflow.round.started"
        assert EventType.WORKFLOW_COMPLETED.value == "workflow.completed"

    def test_task_events(self):
        assert EventType.TASK_STARTED.value == "task.started"
        assert EventType.TASK_COMPLETED.value == "task.completed"

    def test_string_enum(self):
        # EventType values can be used as plain strings
        assert EventType.WORKFLOW_STARTED == "workflow.started"


# ---------------------------------------------------------------------------
# EventBus — subscription and dispatch
# ---------------------------------------------------------------------------


class TestEventBusSubscription:
    def test_exact_match(self):
        bus = EventBus()
        received = []
        bus.subscribe("workflow.started", received.append)

        bus.emit(FelixEvent(event_type="workflow.started", source="test"))
        bus.emit(FelixEvent(event_type="workflow.completed", source="test"))

        assert len(received) == 1
        assert received[0].event_type == "workflow.started"

    def test_prefix_match(self):
        bus = EventBus()
        received = []
        bus.subscribe("agent.*", received.append)

        bus.emit(FelixEvent(event_type="agent.spawned", source="test"))
        bus.emit(FelixEvent(event_type="agent.completed", source="test"))
        bus.emit(FelixEvent(event_type="workflow.started", source="test"))

        assert len(received) == 2
        assert received[0].event_type == "agent.spawned"
        assert received[1].event_type == "agent.completed"

    def test_catch_all(self):
        bus = EventBus()
        received = []
        bus.subscribe_all(received.append)

        bus.emit(FelixEvent(event_type="agent.spawned", source="a"))
        bus.emit(FelixEvent(event_type="workflow.started", source="b"))

        assert len(received) == 2

    def test_multiple_subscribers(self):
        bus = EventBus()
        r1, r2 = [], []
        bus.subscribe("test.event", r1.append)
        bus.subscribe("test.event", r2.append)

        bus.emit(FelixEvent(event_type="test.event", source="test"))

        assert len(r1) == 1
        assert len(r2) == 1

    def test_unsubscribe_exact(self):
        bus = EventBus()
        received = []
        bus.subscribe("test.event", received.append)
        bus.unsubscribe("test.event", received.append)

        bus.emit(FelixEvent(event_type="test.event", source="test"))
        assert len(received) == 0

    def test_unsubscribe_prefix(self):
        bus = EventBus()
        received = []
        bus.subscribe("agent.*", received.append)
        bus.unsubscribe("agent.*", received.append)

        bus.emit(FelixEvent(event_type="agent.spawned", source="test"))
        assert len(received) == 0

    def test_unsubscribe_all(self):
        bus = EventBus()
        received = []
        bus.subscribe_all(received.append)
        bus.unsubscribe_all(received.append)

        bus.emit(FelixEvent(event_type="test.event", source="test"))
        assert len(received) == 0


# ---------------------------------------------------------------------------
# EventBus — exception isolation
# ---------------------------------------------------------------------------


class TestEventBusExceptionIsolation:
    def test_bad_callback_does_not_block_others(self):
        bus = EventBus()
        received = []

        def bad_callback(event: FelixEvent) -> None:
            raise RuntimeError("boom")

        bus.subscribe("test.event", bad_callback)
        bus.subscribe("test.event", received.append)

        # Should not raise
        bus.emit(FelixEvent(event_type="test.event", source="test"))

        # Good callback still ran
        assert len(received) == 1


# ---------------------------------------------------------------------------
# EventBus — history
# ---------------------------------------------------------------------------


class TestEventBusHistory:
    def test_history_disabled_by_default(self):
        bus = EventBus()
        bus.emit(FelixEvent(event_type="test", source="test"))
        assert len(bus.history) == 0

    def test_history_enabled(self):
        bus = EventBus()
        bus.enable_history()

        bus.emit(FelixEvent(event_type="a", source="test"))
        bus.emit(FelixEvent(event_type="b", source="test"))

        assert len(bus.history) == 2
        assert bus.history[0].event_type == "a"

    def test_history_returns_copy(self):
        bus = EventBus()
        bus.enable_history()
        bus.emit(FelixEvent(event_type="x", source="test"))

        h1 = bus.history
        h2 = bus.history
        assert h1 is not h2

    def test_disable_clears_history(self):
        bus = EventBus()
        bus.enable_history()
        bus.emit(FelixEvent(event_type="x", source="test"))
        bus.enable_history(False)
        assert len(bus.history) == 0


# ---------------------------------------------------------------------------
# EventBus — utilities
# ---------------------------------------------------------------------------


class TestEventBusUtilities:
    def test_clear(self):
        bus = EventBus()
        bus.subscribe("test", lambda e: None)
        bus.subscribe_all(lambda e: None)
        bus.enable_history()
        bus.emit(FelixEvent(event_type="test", source="test"))

        bus.clear()
        assert bus.subscriber_count == 0
        assert len(bus.history) == 0

    def test_subscriber_count(self):
        bus = EventBus()
        assert bus.subscriber_count == 0

        bus.subscribe("a", lambda e: None)
        bus.subscribe("b.*", lambda e: None)
        bus.subscribe_all(lambda e: None)

        assert bus.subscriber_count == 3

    def test_thread_safety(self):
        """Concurrent emit/subscribe should not crash."""
        bus = EventBus()
        received = []
        bus.subscribe_all(received.append)

        def emitter():
            for _ in range(50):
                bus.emit(FelixEvent(event_type="test", source="thread"))

        threads = [threading.Thread(target=emitter) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 200


# ---------------------------------------------------------------------------
# EventEmitterMixin
# ---------------------------------------------------------------------------


class TestEventEmitterMixin:
    def test_no_bus_is_silent(self):
        class MyComponent(EventEmitterMixin):
            pass

        comp = MyComponent()
        # Should not raise even with no bus
        comp.emit_event("test.event", {"key": "val"})

    def test_emit_with_bus(self):
        class MyComponent(EventEmitterMixin):
            pass

        bus = EventBus()
        received = []
        bus.subscribe_all(received.append)

        comp = MyComponent()
        comp.set_event_bus(bus)
        comp.emit_event("test.event", {"key": "val"})

        assert len(received) == 1
        assert received[0].event_type == "test.event"
        assert received[0].data == {"key": "val"}

    def test_default_source(self):
        class MyComponent(EventEmitterMixin):
            pass

        bus = EventBus()
        received = []
        bus.subscribe_all(received.append)

        comp = MyComponent()
        comp.set_event_bus(bus)
        comp.emit_event("test.event")

        assert received[0].source == "MyComponent"

    def test_custom_source(self):
        class MyComponent(EventEmitterMixin):
            pass

        bus = EventBus()
        received = []
        bus.subscribe_all(received.append)

        comp = MyComponent()
        comp.set_event_bus(bus)
        comp.emit_event("test.event", source="custom:src")

        assert received[0].source == "custom:src"

    def test_detach_bus(self):
        class MyComponent(EventEmitterMixin):
            pass

        bus = EventBus()
        received = []
        bus.subscribe_all(received.append)

        comp = MyComponent()
        comp.set_event_bus(bus)
        comp.emit_event("a")
        comp.set_event_bus(None)
        comp.emit_event("b")

        assert len(received) == 1
