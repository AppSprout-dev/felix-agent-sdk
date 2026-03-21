"""Felix shared utilities.

Logging configuration, event-log bridging, and common helpers.
"""

from felix_agent_sdk.utils.logging import (
    EventLogBridge,
    FelixLogConfig,
    JSONFormatter,
    configure_logging,
)

__all__ = [
    "configure_logging",
    "FelixLogConfig",
    "JSONFormatter",
    "EventLogBridge",
]
