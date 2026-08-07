"""Felix dynamic team adaptation.

Confidence-driven agent spawning that adapts team composition in real time.
"""

from felix_agent_sdk.spawning.confidence_monitor import (
    ConfidenceMonitor,
    ConfidenceStatus,
    SpawnRecommendation,
)
from felix_agent_sdk.spawning.content_analyzer import ContentAnalyzer, CoverageReport
from felix_agent_sdk.spawning.optimizer import TeamSizeConfig, TeamSizeOptimizer
from felix_agent_sdk.spawning.spawner import DynamicSpawner

__all__ = [
    "ConfidenceMonitor",
    "ConfidenceStatus",
    "SpawnRecommendation",
    "ContentAnalyzer",
    "CoverageReport",
    "TeamSizeConfig",
    "TeamSizeOptimizer",
    "DynamicSpawner",
]
