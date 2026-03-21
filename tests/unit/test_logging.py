"""Tests for structured logging configuration and EventLogBridge."""

from __future__ import annotations

import json
import logging


from felix_agent_sdk.events import EventBus, EventType, FelixEvent
from felix_agent_sdk.utils.logging import (
    EventLogBridge,
    FelixLogConfig,
    JSONFormatter,
    configure_logging,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_handler() -> logging.Handler:
    """Return a handler that stores formatted records in handler.records."""
    handler = logging.StreamHandler()
    handler.records: list[str] = []  # type: ignore[attr-defined]

    def capturing_emit(record: logging.LogRecord) -> None:
        handler.records.append(handler.format(record))  # type: ignore[attr-defined]

    handler.emit = capturing_emit  # type: ignore[method-assign]
    return handler


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    def setup_method(self):
        # Clean up any managed handlers from previous tests
        root = logging.getLogger("felix_agent_sdk")
        for h in list(root.handlers):
            if getattr(h, "_felix_managed", False):
                root.removeHandler(h)

    def test_default_config(self):
        configure_logging()
        root = logging.getLogger("felix_agent_sdk")
        assert root.level == logging.INFO
        managed = [h for h in root.handlers if getattr(h, "_felix_managed", False)]
        assert len(managed) == 1

    def test_custom_level(self):
        configure_logging(FelixLogConfig(level="DEBUG"))
        root = logging.getLogger("felix_agent_sdk")
        assert root.level == logging.DEBUG

    def test_idempotent(self):
        configure_logging()
        configure_logging()
        root = logging.getLogger("felix_agent_sdk")
        managed = [h for h in root.handlers if getattr(h, "_felix_managed", False)]
        assert len(managed) == 1  # not 2

    def test_subsystem_levels(self):
        configure_logging(
            FelixLogConfig(subsystem_levels={"felix_agent_sdk.providers": "WARNING"})
        )
        sub = logging.getLogger("felix_agent_sdk.providers")
        assert sub.level == logging.WARNING

    def test_custom_handler(self):
        handler = _capture_handler()
        configure_logging(FelixLogConfig(output_handler=handler))
        logger = logging.getLogger("felix_agent_sdk.test_custom")
        logger.info("hello")
        assert any("hello" in r for r in handler.records)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# JSONFormatter
# ---------------------------------------------------------------------------


class TestJSONFormatter:
    def test_valid_json(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="felix_agent_sdk.test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "felix_agent_sdk.test"
        assert parsed["message"] == "test message"
        assert "timestamp" in parsed

    def test_extra_fields(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=(), exc_info=None,
        )
        record.event_type = "workflow.started"  # type: ignore[attr-defined]
        record.event_source = "workflow"  # type: ignore[attr-defined]
        record.event_data = {"task": "test"}  # type: ignore[attr-defined]

        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["event_type"] == "workflow.started"
        assert parsed["event_source"] == "workflow"
        assert parsed["event_data"] == {"task": "test"}

    def test_json_format_via_configure(self):
        handler = _capture_handler()
        configure_logging(FelixLogConfig(format="json", output_handler=handler))
        logger = logging.getLogger("felix_agent_sdk.test_json")
        logger.info("json test")
        assert len(handler.records) > 0  # type: ignore[attr-defined]
        parsed = json.loads(handler.records[0])  # type: ignore[attr-defined]
        assert parsed["message"] == "json test"


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------


class TestTextFormatter:
    def test_with_timestamps(self):
        handler = _capture_handler()
        configure_logging(
            FelixLogConfig(format="text", include_timestamps=True, output_handler=handler)
        )
        logger = logging.getLogger("felix_agent_sdk.test_text_ts")
        logger.info("ts test")
        # Should contain a timestamp-like pattern (year)
        assert any("202" in r for r in handler.records)  # type: ignore[attr-defined]

    def test_without_timestamps(self):
        handler = _capture_handler()
        configure_logging(
            FelixLogConfig(format="text", include_timestamps=False, output_handler=handler)
        )
        logger = logging.getLogger("felix_agent_sdk.test_text_nots")
        logger.info("no ts")
        # First token should be the level, not a timestamp
        assert handler.records[0].startswith("INFO")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# EventLogBridge
# ---------------------------------------------------------------------------


class TestEventLogBridge:
    def test_logs_events(self):
        handler = _capture_handler()
        configure_logging(FelixLogConfig(level="DEBUG", output_handler=handler))

        bus = EventBus()
        EventLogBridge(bus)

        bus.emit(FelixEvent(
            event_type=EventType.WORKFLOW_STARTED,
            source="workflow",
            data={"task": "test"},
        ))

        assert any("workflow.started" in r for r in handler.records)  # type: ignore[attr-defined]

    def test_json_bridge(self):
        handler = _capture_handler()
        configure_logging(FelixLogConfig(level="DEBUG", format="json", output_handler=handler))

        bus = EventBus()
        EventLogBridge(bus)

        bus.emit(FelixEvent(
            event_type=EventType.TASK_COMPLETED,
            source="agent:research-001",
            data={"confidence": 0.75},
        ))

        assert len(handler.records) > 0  # type: ignore[attr-defined]
        parsed = json.loads(handler.records[0])  # type: ignore[attr-defined]
        assert parsed["event_type"] == "task.completed"
        assert parsed["event_source"] == "agent:research-001"

    def test_detach(self):
        handler = _capture_handler()
        configure_logging(FelixLogConfig(level="DEBUG", output_handler=handler))

        bus = EventBus()
        bridge = EventLogBridge(bus)
        bridge.detach()

        bus.emit(FelixEvent(event_type="test", source="test"))
        # Should have no event log records after detach
        event_records = [r for r in handler.records if "test" in r and "[test]" in r]  # type: ignore[attr-defined]
        assert len(event_records) == 0

    def test_custom_level_map(self):
        handler = _capture_handler()
        configure_logging(FelixLogConfig(level="DEBUG", output_handler=handler))

        bus = EventBus()
        EventLogBridge(bus, level_map={"custom.event": "WARNING"})

        bus.emit(FelixEvent(event_type="custom.event", source="test"))

        assert any("WARNING" in r for r in handler.records)  # type: ignore[attr-defined]

    def test_default_level_is_debug(self):
        handler = _capture_handler()
        configure_logging(FelixLogConfig(level="DEBUG", output_handler=handler))

        bus = EventBus()
        EventLogBridge(bus)

        bus.emit(FelixEvent(event_type="unknown.event.type", source="test"))

        assert any("DEBUG" in r for r in handler.records)  # type: ignore[attr-defined]

    def test_workflow_events_log_at_info(self):
        handler = _capture_handler()
        configure_logging(FelixLogConfig(level="DEBUG", output_handler=handler))

        bus = EventBus()
        EventLogBridge(bus)

        bus.emit(FelixEvent(event_type=EventType.WORKFLOW_STARTED, source="wf"))
        bus.emit(FelixEvent(event_type=EventType.WORKFLOW_COMPLETED, source="wf"))

        info_records = [r for r in handler.records if "INFO" in r]  # type: ignore[attr-defined]
        assert len(info_records) >= 2
