"""Confidence monitoring for dynamic spawning decisions.

Tracks per-agent and team-wide confidence trends, detects stagnation,
and recommends whether the system should spawn, hold, or prune agents.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class SpawnRecommendation(str, Enum):
    """What the monitor recommends the spawner do."""

    SPAWN = "spawn"
    HOLD = "hold"
    PRUNE = "prune"


@dataclass(frozen=True)
class ConfidenceStatus:
    """Snapshot of the team's confidence health."""

    team_average: float
    trend: str  # "rising", "falling", "stable"
    gap_to_threshold: float
    is_stagnating: bool
    recommendation: SpawnRecommendation


# If the team average is more than this below the threshold, spawn
# immediately regardless of trend.
_CRITICAL_GAP = 0.2

# Decimal places used when rounding confidence values in status reports.
_REPORT_PRECISION = 4


class ConfidenceMonitor:
    """Track team confidence and decide whether spawning is warranted.

    The monitor maintains a sliding window of per-agent confidence
    values and computes team-level statistics each time :meth:`update`
    is called.

    Args:
        threshold: Team average confidence below which spawning is
            recommended.
        stagnation_window: Number of consecutive updates with
            less than *stagnation_delta* improvement before the
            team is considered stagnating.
        stagnation_delta: Minimum improvement per update to avoid
            stagnation.
    """

    def __init__(
        self,
        threshold: float = 0.80,
        stagnation_window: int = 3,
        stagnation_delta: float = 0.01,
    ) -> None:
        self._threshold = threshold
        self._stagnation_window = stagnation_window
        self._stagnation_delta = stagnation_delta

        self._agent_history: Dict[str, List[float]] = defaultdict(list)
        self._team_averages: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, agent_id: str, confidence: float) -> None:
        """Record a new confidence observation for *agent_id*."""
        self._agent_history[agent_id].append(confidence)

    def record_round(self, confidences: Dict[str, float]) -> None:
        """Record a full round of confidence observations.

        Also updates the team average history for trend detection.
        """
        for agent_id, confidence in confidences.items():
            self.update(agent_id, confidence)

        if confidences:
            avg = sum(confidences.values()) / len(confidences)
            self._team_averages.append(avg)

    def should_spawn(self) -> bool:
        """Return ``True`` if the team would benefit from more agents."""
        return self.get_status().recommendation == SpawnRecommendation.SPAWN

    def get_status(self) -> ConfidenceStatus:
        """Compute current confidence status and spawn recommendation."""
        avg = self._current_average()
        trend = self._compute_trend()
        gap = self._threshold - avg
        stagnating = self._is_stagnating()
        rec = self._decide(avg, trend, gap, stagnating)

        return ConfidenceStatus(
            team_average=round(avg, _REPORT_PRECISION),
            trend=trend,
            gap_to_threshold=round(max(gap, 0.0), _REPORT_PRECISION),
            is_stagnating=stagnating,
            recommendation=rec,
        )

    def _decide(
        self, avg: float, trend: str, gap: float, stagnating: bool
    ) -> SpawnRecommendation:
        """Pure decision function: given metrics, return a recommendation."""
        if avg >= self._threshold:
            return SpawnRecommendation.HOLD
        if stagnating or trend == "falling" or gap > _CRITICAL_GAP:
            return SpawnRecommendation.SPAWN
        return SpawnRecommendation.HOLD

    def reset(self) -> None:
        """Clear all tracked state."""
        self._agent_history.clear()
        self._team_averages.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _current_average(self) -> float:
        if not self._team_averages:
            return 0.0
        return self._team_averages[-1]

    def _compute_trend(self) -> str:
        avgs = self._team_averages
        if len(avgs) < 2:
            return "stable"
        delta = avgs[-1] - avgs[-2]
        if delta > self._stagnation_delta:
            return "rising"
        elif delta < -self._stagnation_delta:
            return "falling"
        return "stable"

    def _is_stagnating(self) -> bool:
        avgs = self._team_averages
        if len(avgs) < self._stagnation_window:
            return False
        window = avgs[-self._stagnation_window:]
        total_change = abs(window[-1] - window[0])
        return total_change < self._stagnation_delta
