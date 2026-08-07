"""Team size optimisation.

Recommends optimal team size based on task complexity signals and
current result quality. Thresholds are config-driven (not buried magic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class TeamSizeConfig:
    """Configurable thresholds for :class:`TeamSizeOptimizer`.

    Defaults preserve historical heuristic behaviour while allowing
    callers to tune without forking the class.
    """

    base_size: int = 3
    signal_increment: int = 1
    max_size: int = 15
    length_medium: int = 200
    length_long: int = 500
    conf_low: float = 0.5
    conf_mid: float = 0.7
    conf_default: float = 0.5
    spread_threshold: float = 0.3
    # When average confidence is already this high, skip length-based
    # growth — extra agents mostly burn tokens without quality lift.
    high_confidence_skip_length: float = 0.85


class TeamSizeOptimizer:
    """Recommender for team size from complexity + quality signals.

    Considers task description length and current confidence spread to
    suggest an appropriate team size. High-confidence rounds avoid
    inflating headcount for long prompts (token efficiency).

    Args:
        min_size: Minimum team size to recommend.
        max_size: Maximum team size to recommend.
        config: Optional threshold config (defaults match prior magic numbers).
    """

    def __init__(
        self,
        min_size: int = 3,
        max_size: int | None = None,
        config: TeamSizeConfig | None = None,
    ) -> None:
        self._config = config or TeamSizeConfig()
        self._min_size = min_size
        self._max_size = max_size if max_size is not None else self._config.max_size

    @property
    def config(self) -> TeamSizeConfig:
        return self._config

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
        cfg = self._config
        size = cfg.base_size

        confidences: list[float] = []
        if current_results:
            confidences = [
                float(r.get("confidence", cfg.conf_default)) for r in current_results
            ]

        avg_conf = (
            sum(confidences) / len(confidences) if confidences else None
        )
        high_quality = (
            avg_conf is not None and avg_conf >= cfg.high_confidence_skip_length
        )

        # Length signals — skipped when quality is already high (efficiency).
        if not high_quality:
            if len(task_description) > cfg.length_medium:
                size += cfg.signal_increment
            if len(task_description) > cfg.length_long:
                size += cfg.signal_increment

        if confidences:
            if avg_conf is not None and avg_conf < cfg.conf_low:
                size += cfg.signal_increment * 2
            elif avg_conf is not None and avg_conf < cfg.conf_mid:
                size += cfg.signal_increment

            if len(confidences) >= 2:
                spread = max(confidences) - min(confidences)
                if spread > cfg.spread_threshold:
                    size += cfg.signal_increment

        return max(self._min_size, min(self._max_size, size))
