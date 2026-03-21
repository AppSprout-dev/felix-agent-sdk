"""Dynamic spawner — orchestrates confidence-driven agent creation.

Combines :class:`ConfidenceMonitor` and :class:`ContentAnalyzer` to
decide whether and what type of agent to spawn each round.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from felix_agent_sdk.agents.factory import AgentFactory
from felix_agent_sdk.agents.llm_agent import LLMAgent, LLMResult
from felix_agent_sdk.communication.spoke import SpokeManager
from felix_agent_sdk.events.bus import EventBus
from felix_agent_sdk.events.mixins import EventEmitterMixin
from felix_agent_sdk.events.types import EventType
from felix_agent_sdk.spawning.confidence_monitor import ConfidenceMonitor
from felix_agent_sdk.spawning.content_analyzer import ContentAnalyzer

logger = logging.getLogger(__name__)


class DynamicSpawner(EventEmitterMixin):
    """Confidence-driven agent spawner.

    Called once per round by the workflow runner. Consults the
    :class:`ConfidenceMonitor` and :class:`ContentAnalyzer` to decide
    whether to create new agents. Newly spawned agents are connected
    to the hub via the spoke manager and returned to the caller so
    they participate in subsequent rounds.

    Args:
        factory: Agent factory for creating new agents.
        spoke_mgr: Spoke manager for hub connectivity.
        monitor: Confidence monitor instance.
        analyzer: Content analyzer instance.
        max_spawned: Maximum number of agents to spawn across the
            entire workflow.
        event_bus: Optional event bus for observability.
    """

    def __init__(
        self,
        factory: AgentFactory,
        spoke_mgr: SpokeManager,
        monitor: Optional[ConfidenceMonitor] = None,
        analyzer: Optional[ContentAnalyzer] = None,
        max_spawned: int = 3,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self._factory = factory
        self._spoke_mgr = spoke_mgr
        self._monitor = monitor or ConfidenceMonitor()
        self._analyzer = analyzer or ContentAnalyzer()
        self._max_spawned = max_spawned
        self._total_spawned = 0
        if event_bus is not None:
            self.set_event_bus(event_bus)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_spawn(
        self,
        agents: List[LLMAgent],
        round_results: List[LLMResult],
        current_time: float,
    ) -> List[LLMAgent]:
        """Evaluate whether to spawn and return any new agents.

        Args:
            agents: Current team of agents.
            round_results: Results from the most recent round.
            current_time: Current helix time parameter.

        Returns:
            List of newly spawned agents (empty if none spawned).
        """
        if self._total_spawned >= self._max_spawned:
            return []

        if not round_results:
            return []

        # Feed confidence data to the monitor
        confidences: Dict[str, float] = {
            r.agent_id: r.confidence for r in round_results
        }
        self._monitor.record_round(confidences)

        # Check if spawning is recommended
        if not self._monitor.should_spawn():
            return []

        # Determine what type of agent to spawn
        result_dicts = self._results_to_dicts(round_results)
        coverage = self._analyzer.analyze_coverage(result_dicts)
        agent_type = coverage.recommended_type

        # Spawn
        status = self._monitor.get_status()
        self.emit_event(
            EventType.SPAWN_TRIGGERED,
            {
                "agent_type": agent_type,
                "reason": status.recommendation.value,
                "team_average": status.team_average,
                "gap": status.gap_to_threshold,
                "trend": status.trend,
            },
        )

        spawn_time = min(current_time, 0.9)
        agent = self._factory.create_agent(
            agent_type=agent_type,
            spawn_time=spawn_time,
        )
        if getattr(self, "_event_bus", None) is not None:
            agent.set_event_bus(self._event_bus)  # type: ignore[arg-type]
        self._spoke_mgr.create_spoke(agent.agent_id, agent=agent)
        self._total_spawned += 1

        logger.info(
            "Dynamic spawn: %s agent %s (total spawned: %d/%d)",
            agent_type,
            agent.agent_id,
            self._total_spawned,
            self._max_spawned,
        )

        self.emit_event(
            EventType.SPAWN_COMPLETED,
            {
                "agent_id": agent.agent_id,
                "agent_type": agent_type,
                "spawn_time": spawn_time,
                "total_spawned": self._total_spawned,
            },
        )

        return [agent]

    @property
    def total_spawned(self) -> int:
        """Number of agents spawned so far."""
        return self._total_spawned

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _results_to_dicts(results: List[LLMResult]) -> List[Dict[str, Any]]:
        """Convert LLMResult list to simple dicts for the analyzer."""
        return [
            {
                "content": r.content,
                "agent_type": r.position_info.get("agent_type", ""),
                "confidence": r.confidence,
            }
            for r in results
        ]
