"""Felix visualization module — terminal-based helix rendering.

Provides a reusable ASCII visualizer that renders agents progressing down
a 3D helix in real time, showing phase boundaries, confidence levels, and
per-agent status in a sidebar panel.
"""

from __future__ import annotations

from felix_agent_sdk.visualization.helix_visualizer import (
    AgentDisplayState,
    HelixVisualizer,
    VisualizerStreamHandler,
)

__all__ = [
    "AgentDisplayState",
    "HelixVisualizer",
    "VisualizerStreamHandler",
]
