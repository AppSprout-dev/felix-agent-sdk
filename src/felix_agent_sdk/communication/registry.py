"""Agent registry for tracking helix positions, phases, and collaboration.

Extracted from CalebisGross/felix src/communication/central_post.py (AgentRegistry class).
Refactored: pure in-memory (no SQLite), phase boundaries use SDK constants.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from felix_agent_sdk.core.helix import ANALYSIS_END, EXPLORATION_END

logger = logging.getLogger(__name__)


class AgentRegistry:
    """In-memory registry tracking agent positions, phases, and performance.

    Agents register themselves on spawn and update their position as they
    traverse the helix.  The registry provides phase-aware lookups and
    convergence status queries for the CentralPost coordination hub.

    Phase boundaries (from core.helix constants):
        exploration: t in [0.0, EXPLORATION_END)   — EXPLORATION_END = 0.4
        analysis:    t in [EXPLORATION_END, ANALYSIS_END) — ANALYSIS_END = 0.7
        synthesis:   t in [ANALYSIS_END, 1.0]
    """

    def __init__(self) -> None:
        # agent_id -> metadata dict
        self._agents: Dict[str, Dict[str, Any]] = {}
        # agent_id -> position info dict
        self._positions: Dict[str, Dict[str, Any]] = {}
        # agent_id -> performance metrics dict
        self._performance: Dict[str, Dict[str, Any]] = {}
        # agent_id -> list of (influenced_agent_id, timestamp) tuples
        self._collaborations: Dict[str, List[tuple]] = {}
        # rolling confidence window for trend calculation
        self._confidence_history: List[float] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a new agent with optional metadata.

        Args:
            agent_id: Unique identifier for the agent.
            metadata: Optional dict of agent attributes (agent_type, spawn_time, …).
        """
        self._agents[agent_id] = {
            "agent_id": agent_id,
            "registered_at": time.time(),
            **(metadata or {}),
        }
        self._positions[agent_id] = {
            "depth_ratio": 0.0,
            "phase": "exploration",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }
        self._performance[agent_id] = {
            "confidence": 0.0,
            "confidence_history": [],
            "tasks_completed": 0,
            "errors": 0,
        }
        self._collaborations[agent_id] = []
        logger.debug("Registered agent %s", agent_id)

    def deregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the registry.

        Args:
            agent_id: Identifier of the agent to remove.
        """
        self._agents.pop(agent_id, None)
        self._positions.pop(agent_id, None)
        self._performance.pop(agent_id, None)
        self._collaborations.pop(agent_id, None)
        logger.debug("Deregistered agent %s", agent_id)

    # ------------------------------------------------------------------
    # Position tracking
    # ------------------------------------------------------------------

    def update_agent_position(self, agent_id: str, position_info: Dict[str, Any]) -> None:
        """Update the helix position for a registered agent.

        Args:
            agent_id: Target agent identifier.
            position_info: Dict containing at minimum ``depth_ratio`` (t value).
                           May also include x, y, z coordinates and phase.
        """
        if agent_id not in self._agents:
            logger.warning("update_agent_position: unknown agent %s", agent_id)
            return

        depth_ratio = position_info.get("depth_ratio", 0.0)
        phase = self._get_phase_from_depth(depth_ratio)

        self._positions[agent_id] = {
            "depth_ratio": depth_ratio,
            "phase": phase,
            "x": position_info.get("x", 0.0),
            "y": position_info.get("y", 0.0),
            "z": position_info.get("z", 0.0),
            "updated_at": time.time(),
        }

    # ------------------------------------------------------------------
    # Performance / confidence
    # ------------------------------------------------------------------

    def update_agent_performance(self, agent_id: str, metrics: Dict[str, Any]) -> None:
        """Update performance metrics for an agent, including confidence tracking.

        Args:
            agent_id: Target agent identifier.
            metrics: Dict which may include ``confidence``, ``tasks_completed``, ``errors``.
        """
        if agent_id not in self._agents:
            logger.warning("update_agent_performance: unknown agent %s", agent_id)
            return

        perf = self._performance[agent_id]

        if "confidence" in metrics:
            confidence = float(metrics["confidence"])
            perf["confidence"] = confidence
            history = perf.setdefault("confidence_history", [])
            history.append(confidence)
            if len(history) > 50:
                perf["confidence_history"] = history[-50:]
            # Also update the global rolling window for trend calculation
            self._confidence_history.append(confidence)
            if len(self._confidence_history) > 100:
                self._confidence_history = self._confidence_history[-100:]

        if "tasks_completed" in metrics:
            perf["tasks_completed"] = int(metrics["tasks_completed"])

        if "errors" in metrics:
            perf["errors"] = int(metrics["errors"])

    # ------------------------------------------------------------------
    # Collaboration
    # ------------------------------------------------------------------

    def record_collaboration(self, agent_id: str, influenced_agent_id: str) -> None:
        """Record that agent_id influenced influenced_agent_id.

        Args:
            agent_id: The agent that initiated or contributed to the collaboration.
            influenced_agent_id: The agent that was influenced.
        """
        if agent_id not in self._collaborations:
            self._collaborations[agent_id] = []
        self._collaborations[agent_id].append((influenced_agent_id, time.time()))

    # ------------------------------------------------------------------
    # Phase-based lookups
    # ------------------------------------------------------------------

    def get_agents_in_phase(self, phase: str) -> List[str]:
        """Return agent IDs whose current position is in the given phase.

        Args:
            phase: One of "exploration", "analysis", "synthesis".

        Returns:
            List of agent IDs in that phase.
        """
        return [
            agent_id
            for agent_id, pos in self._positions.items()
            if agent_id in self._agents and pos.get("phase") == phase
        ]

    def get_nearby_agents(self, agent_id: str, radius_threshold: float = 0.1) -> List[str]:
        """Return agents within *radius_threshold* depth_ratio of *agent_id*.

        Args:
            agent_id: Reference agent.
            radius_threshold: Maximum depth_ratio distance to consider "nearby".

        Returns:
            List of nearby agent IDs (excluding the reference agent itself).
        """
        if agent_id not in self._positions:
            return []

        ref_depth = self._positions[agent_id].get("depth_ratio", 0.0)
        nearby = []
        for other_id, pos in self._positions.items():
            if other_id == agent_id or other_id not in self._agents:
                continue
            if abs(pos.get("depth_ratio", 0.0) - ref_depth) <= radius_threshold:
                nearby.append(other_id)
        return nearby

    # ------------------------------------------------------------------
    # Convergence status
    # ------------------------------------------------------------------

    def get_convergence_status(self) -> Dict[str, Any]:
        """Compute team-wide convergence status.

        Returns a dict with:
        - ``confidence_trend``: recent trend direction ("rising", "falling", "stable")
        - ``phase_distribution``: count of agents per phase
        - ``synthesis_ready``: True if majority are in synthesis with high confidence
        - ``active_agent_count``: total registered agents
        - ``collaboration_density``: float measure of collaboration activity
        """
        active_count = len(self._agents)
        phase_distribution = {
            "exploration": len(self.get_agents_in_phase("exploration")),
            "analysis": len(self.get_agents_in_phase("analysis")),
            "synthesis": len(self.get_agents_in_phase("synthesis")),
        }

        confidence_trend = self._get_recent_confidence_trend()
        collaboration_density = self._calculate_collaboration_density()

        # Synthesis ready: majority in synthesis AND average confidence >= 0.7
        synthesis_count = phase_distribution["synthesis"]
        synthesis_ready = False
        if active_count > 0 and synthesis_count > active_count / 2:
            synthesis_confidences = [
                self._performance[aid]["confidence"]
                for aid in self.get_agents_in_phase("synthesis")
                if aid in self._performance
            ]
            if synthesis_confidences:
                avg_confidence = sum(synthesis_confidences) / len(synthesis_confidences)
                synthesis_ready = avg_confidence >= 0.7

        return {
            "confidence_trend": confidence_trend,
            "phase_distribution": phase_distribution,
            "synthesis_ready": synthesis_ready,
            "active_agent_count": active_count,
            "collaboration_density": collaboration_density,
        }

    # ------------------------------------------------------------------
    # Info accessors
    # ------------------------------------------------------------------

    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Return combined info dict for an agent, or None if not registered.

        Merges metadata, position, and performance into a single dict.
        """
        if agent_id not in self._agents:
            return None
        return {
            **self._agents[agent_id],
            "position": self._positions.get(agent_id, {}),
            "performance": self._performance.get(agent_id, {}),
        }

    def get_active_agents(self) -> List[str]:
        """Return the list of all currently registered agent IDs."""
        return list(self._agents.keys())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_phase_from_depth(self, depth_ratio: float) -> str:
        """Map a helix depth_ratio (t value) to a phase name.

        Uses SDK constants:
            [0.0, EXPLORATION_END)            -> "exploration"
            [EXPLORATION_END, ANALYSIS_END)   -> "analysis"
            [ANALYSIS_END, 1.0]               -> "synthesis"
        """
        if depth_ratio < EXPLORATION_END:
            return "exploration"
        if depth_ratio < ANALYSIS_END:
            return "analysis"
        return "synthesis"

    def _get_current_phase(self, agent_id: str) -> str:
        """Return the current phase for a registered agent."""
        if agent_id not in self._positions:
            return "exploration"
        depth_ratio = self._positions[agent_id].get("depth_ratio", 0.0)
        return self._get_phase_from_depth(depth_ratio)

    def _get_recent_confidence_trend(self, window: int = 10) -> str:
        """Calculate the recent confidence trend over the last *window* values.

        Returns:
            "rising" if trend > +0.05, "falling" if < -0.05, else "stable".
        """
        history = self._confidence_history[-window:]
        if len(history) < 2:
            return "stable"

        half = max(1, len(history) // 2)
        earlier_avg = sum(history[:half]) / half
        recent_avg = sum(history[half:]) / (len(history) - half)
        delta = recent_avg - earlier_avg

        if delta > 0.05:
            return "rising"
        if delta < -0.05:
            return "falling"
        return "stable"

    def _calculate_collaboration_density(self) -> float:
        """Compute a normalised collaboration density score.

        Defined as total collaboration events divided by max possible
        agent pairs (N * (N-1)), clamped to [0.0, 1.0].
        """
        total_events = sum(len(events) for events in self._collaborations.values())
        agent_count = len(self._agents)
        if agent_count < 2:
            return 0.0
        max_pairs = agent_count * (agent_count - 1)
        return min(1.0, total_events / max_pairs)
