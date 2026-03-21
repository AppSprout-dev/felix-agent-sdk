"""Team size optimisation heuristic.

Recommends optimal team size based on task complexity signals and
current result quality.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Base team size for a simple task
_BASE_TEAM_SIZE = 3

# Each complexity signal adds this many agents
_SIGNAL_INCREMENT = 1

# Hard cap to prevent runaway growth
_MAX_TEAM_SIZE = 15


class TeamSizeOptimizer:
    """Heuristic recommender for team size.

    Considers task description length, topic breadth (keyword count),
    and current confidence spread to suggest an appropriate team size.

    Args:
        min_size: Minimum team size to recommend.
        max_size: Maximum team size to recommend.
    """

    def __init__(self, min_size: int = 3, max_size: int = _MAX_TEAM_SIZE) -> None:
        self._min_size = min_size
        self._max_size = max_size

    def recommend_team_size(
        self,
        task_description: str,
        current_results: List[Dict[str, Any]] | None = None,
    ) -> int:
        """Return a recommended team size.

        Args:
            task_description: The task text (length is a complexity signal).
            current_results: Existing results (optional). Each dict should
                have ``confidence`` (float) and ``content`` (str) keys.

        Returns:
            Recommended team size clamped to [min_size, max_size].
        """
        size = _BASE_TEAM_SIZE

        # Signal 1: long task descriptions suggest complexity
        if len(task_description) > 200:
            size += _SIGNAL_INCREMENT
        if len(task_description) > 500:
            size += _SIGNAL_INCREMENT

        # Signal 2: low average confidence from existing results
        if current_results:
            confidences = [r.get("confidence", 0.5) for r in current_results]
            avg_conf = sum(confidences) / len(confidences)
            if avg_conf < 0.5:
                size += _SIGNAL_INCREMENT * 2
            elif avg_conf < 0.7:
                size += _SIGNAL_INCREMENT

        # Signal 3: wide confidence spread suggests disagreement
        if current_results and len(current_results) >= 2:
            confidences = [r.get("confidence", 0.5) for r in current_results]
            spread = max(confidences) - min(confidences)
            if spread > 0.3:
                size += _SIGNAL_INCREMENT

        return max(self._min_size, min(self._max_size, size))
