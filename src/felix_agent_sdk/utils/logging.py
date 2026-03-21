"""Structured logging configuration for the Felix SDK.

Provides a one-call ``configure_logging()`` helper that sets up sensible
defaults for all ``felix_agent_sdk.*`` loggers, plus an ``EventLogBridge``
that subscribes to an :class:`EventBus` and logs every event automatically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

from felix_agent_sdk.events.bus import EventBus
from felix_agent_sdk.events.types import FelixEvent

_ROOT_LOGGER_NAME = "felix_agent_sdk"

# Default log level mapping for events (event_type prefix → level name)
_DEFAULT_EVENT_LEVELS: Dict[str, str] = {
    "workflow.started": "INFO",
    "workflow.completed": "INFO",
    "workflow.converged": "INFO",
    "workflow.round": "DEBUG",
    "workflow.synthesis": "DEBUG",
    "task.": "DEBUG",
    "agent.": "DEBUG",
    "message.": "DEBUG",
    "stream.": "DEBUG",
    "spawn.": "INFO",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FelixLogConfig:
    """Configuration for :func:`configure_logging`.

    Attributes:
        level: Root log level for all ``felix_agent_sdk.*`` loggers.
        format: ``"text"`` for human-readable, ``"json"`` for structured output.
        subsystem_levels: Override levels for specific loggers, e.g.
            ``{"felix_agent_sdk.providers": "DEBUG"}``.
        include_timestamps: Prefix lines with ISO-8601 timestamps (text mode).
        output_handler: Pre-built handler to use. When ``None`` a
            :class:`logging.StreamHandler` writing to stderr is created.
    """

    level: str = "INFO"
    format: Literal["text", "json"] = "text"
    subsystem_levels: Dict[str, str] = field(default_factory=dict)
    include_timestamps: bool = True
    output_handler: Optional[logging.Handler] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON object per line.

    Fields: ``timestamp``, ``level``, ``logger``, ``message``, and any
    ``extra`` keys set on the record.
    """

    def format(self, record: logging.LogRecord) -> str:
        obj: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Capture extra fields attached by EventLogBridge or user code
        for key in ("event_type", "event_source", "event_data"):
            val = getattr(record, key, None)
            if val is not None:
                obj[key] = val
        if record.exc_info and record.exc_info[1] is not None:
            obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(obj, default=str)


def _build_text_formatter(include_timestamps: bool) -> logging.Formatter:
    parts = []
    if include_timestamps:
        parts.append("%(asctime)s")
    parts.extend(["%(levelname)-8s", "%(name)s", "%(message)s"])
    return logging.Formatter(" ".join(parts))


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


def configure_logging(config: Optional[FelixLogConfig] = None) -> None:
    """Set up logging for all ``felix_agent_sdk.*`` loggers.

    Idempotent — safe to call multiple times. Subsequent calls replace
    the handler and formatter but do not duplicate handlers.

    Args:
        config: Logging configuration. Uses sensible defaults when ``None``.
    """
    cfg = config or FelixLogConfig()

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))

    # Remove any handlers previously attached by this function
    for h in list(root.handlers):
        if getattr(h, "_felix_managed", False):
            root.removeHandler(h)

    handler = cfg.output_handler or logging.StreamHandler()
    handler._felix_managed = True  # type: ignore[attr-defined]

    if cfg.format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(_build_text_formatter(cfg.include_timestamps))

    root.addHandler(handler)

    # Per-subsystem overrides
    for logger_name, level_str in cfg.subsystem_levels.items():
        sub = logging.getLogger(logger_name)
        sub.setLevel(getattr(logging, level_str.upper(), logging.INFO))


# ---------------------------------------------------------------------------
# EventLogBridge
# ---------------------------------------------------------------------------


class EventLogBridge:
    """Subscribe to an :class:`EventBus` and log every event.

    Each event is logged to the ``felix_agent_sdk.events`` logger at a
    level determined by the event type (see ``_DEFAULT_EVENT_LEVELS``).
    Custom level mappings can be provided via *level_map*.

    Examples::

        bus = EventBus()
        configure_logging()
        bridge = EventLogBridge(bus)
        # Now every event emitted on *bus* appears in the log.

        # Cleanup
        bridge.detach()
    """

    def __init__(
        self,
        bus: EventBus,
        level_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self._bus = bus
        self._level_map = level_map or dict(_DEFAULT_EVENT_LEVELS)
        self._logger = logging.getLogger(f"{_ROOT_LOGGER_NAME}.events")
        bus.subscribe_all(self._on_event)

    # ------------------------------------------------------------------

    def _resolve_level(self, event_type: str) -> int:
        """Return the numeric log level for *event_type*."""
        # Exact match first
        level_str = self._level_map.get(event_type)
        if level_str:
            return getattr(logging, level_str.upper(), logging.DEBUG)
        # Prefix match
        for prefix, lvl in self._level_map.items():
            if event_type.startswith(prefix):
                return getattr(logging, lvl.upper(), logging.DEBUG)
        return logging.DEBUG

    def _on_event(self, event: FelixEvent) -> None:
        level = self._resolve_level(event.event_type)
        extra = {
            "event_type": event.event_type,
            "event_source": event.source,
            "event_data": event.data,
        }
        self._logger.log(
            level,
            "[%s] %s %s",
            event.event_type,
            event.source,
            _summarise_data(event.data),
            extra=extra,
        )

    # ------------------------------------------------------------------

    def detach(self) -> None:
        """Unsubscribe from the event bus."""
        self._bus.unsubscribe_all(self._on_event)


def _summarise_data(data: Dict[str, Any], max_len: int = 120) -> str:
    """One-line summary of event data for text logs."""
    if not data:
        return ""
    parts = [f"{k}={v}" for k, v in data.items()]
    text = " ".join(parts)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text
