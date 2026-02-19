"""Agent lifecycle management for Felix Agent SDK.

Ported from CalebisGross/felix src/agents/agent.py.
Core agent that traverses the helical path with non-linear progression,
adaptive velocity, and confidence tracking.

Mathematical Foundation:
    Agent spawn distribution: T_i ~ U(0,1) (uniform random timing).
    Position progression along helix with velocity and acceleration modifiers.
    Confidence-driven velocity adaptation enables emergent self-regulation.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from felix_agent_sdk.core.helix import HelixGeometry, HelixPosition


class AgentState(Enum):
    """Agent lifecycle states."""

    WAITING = "waiting"
    SPAWNING = "spawning"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class Agent:
    """Autonomous agent that traverses the helical path.

    Agents spawn at random times and progress along the helix while
    processing assigned tasks.  Position and state are tracked throughout
    the lifecycle.
    """

    def __init__(
        self,
        agent_id: str,
        helix: HelixGeometry,
        spawn_time: float = 0.0,
        velocity: Optional[float] = None,
    ) -> None:
        """Initialize agent with lifecycle parameters.

        Args:
            agent_id: Unique identifier for the agent.
            helix: Helix geometry for path calculation.
            spawn_time: Time when agent becomes active (0.0 to 1.0).
            velocity: Fixed velocity multiplier. ``None`` picks a random
                value in [0.7, 1.3].

        Raises:
            ValueError: If *agent_id* is empty or *spawn_time* outside [0, 1].
        """
        self._validate_initialization(agent_id, spawn_time)

        self.agent_id = agent_id
        self.spawn_time = spawn_time
        self._helix = helix
        self._state = AgentState.WAITING
        self.current_task: Optional[Any] = None

        # Overridden by subclasses (e.g. LLMAgent sets "research", "analysis", …)
        self.agent_type: str = "generic"
        self.confidence: float = 0.0

        # Non-linear progression mechanics
        self._progress: float = 0.0
        self._spawn_timestamp: Optional[float] = None
        self._velocity: float = velocity if velocity is not None else random.uniform(0.7, 1.3)
        self._acceleration: float = 1.0
        self._pause_until: Optional[float] = None
        self._confidence_history: List[float] = []

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_initialization(agent_id: str, spawn_time: float) -> None:
        if not agent_id or agent_id.strip() == "":
            raise ValueError("agent_id cannot be empty")
        if not (0.0 <= spawn_time <= 1.0):
            raise ValueError("spawn_time must be between 0 and 1")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> AgentState:
        """Current lifecycle state."""
        return self._state

    @property
    def progress(self) -> float:
        """Current progress along the helix (0.0 to 1.0)."""
        return self._progress

    @property
    def position(self) -> HelixPosition:
        """Phase-aware position wrapper at the agent's current progress."""
        return HelixPosition(self._helix, self._progress)

    @property
    def velocity(self) -> float:
        """Effective velocity (base velocity * acceleration)."""
        return self._velocity * self._acceleration

    @property
    def is_paused(self) -> bool:
        return self._pause_until is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def can_spawn(self, current_time: float) -> bool:
        """Return ``True`` if the agent may spawn at *current_time*."""
        return current_time >= self.spawn_time

    def spawn(self, current_time: float, task: Any = None) -> None:
        """Spawn the agent — transition from WAITING to ACTIVE.

        Raises:
            ValueError: If spawn conditions are not met.
        """
        if not self.can_spawn(current_time):
            raise ValueError("Cannot spawn agent before spawn_time")
        if self._state != AgentState.WAITING:
            raise ValueError("Agent already spawned")

        self._progress = 0.0
        self._state = AgentState.ACTIVE
        self.current_task = task
        self._spawn_timestamp = current_time

    # ------------------------------------------------------------------
    # Position progression
    # ------------------------------------------------------------------

    def update_position(self, current_time: float) -> None:
        """Advance the agent along the helix based on elapsed time.

        Non-linear progression applies velocity and acceleration modifiers.

        Raises:
            ValueError: If the agent has not been spawned.
        """
        if self._state == AgentState.WAITING:
            raise ValueError("Cannot update position of unspawned agent")
        if self._state in (AgentState.COMPLETED, AgentState.FAILED):
            return

        if self._spawn_timestamp is None:
            raise ValueError("Cannot update position: agent has not spawned")

        # Respect pause
        if self._pause_until is not None and current_time < self._pause_until:
            return

        base_progression = current_time - self._spawn_timestamp
        effective_velocity = self._velocity * self._acceleration
        self._progress = max(0.0, min(base_progression * effective_velocity, 1.0))

        self._adapt_velocity_from_confidence()

        if self._progress >= 1.0:
            self._state = AgentState.COMPLETED

    def get_position(self, current_time: float) -> Optional[Tuple[float, float, float]]:
        """Return (x, y, z) on the helix, or ``None`` if not yet spawned."""
        if self._state == AgentState.WAITING:
            return None
        self.update_position(current_time)
        return self._helix.get_position(self._progress)

    def get_position_info(self) -> Dict[str, Any]:
        """Detailed position dictionary for diagnostics and prompt context."""
        pos = self.position
        return {
            "x": pos.x,
            "y": pos.y,
            "z": pos.z,
            "radius": pos.radius,
            "depth_ratio": self._progress,
            "progress": self._progress,
            "phase": pos.phase,
        }

    # ------------------------------------------------------------------
    # Confidence tracking
    # ------------------------------------------------------------------

    def record_confidence(self, confidence: float) -> None:
        """Record a confidence score and adapt velocity accordingly."""
        self.confidence = confidence
        self._confidence_history.append(confidence)
        if len(self._confidence_history) > 10:
            self._confidence_history = self._confidence_history[-10:]

    def _adapt_velocity_from_confidence(self) -> None:
        """Adjust acceleration based on confidence trend."""
        if len(self._confidence_history) < 2:
            return

        recent_avg = sum(self._confidence_history[-2:]) / 2
        earlier = self._confidence_history[:-2]
        earlier_avg = sum(earlier) / max(1, len(earlier))
        trend = recent_avg - earlier_avg

        if trend > 0.1:
            self._acceleration = min(1.5, self._acceleration * 1.1)
        elif trend < -0.1:
            self._acceleration = max(0.5, self._acceleration * 0.9)

    # ------------------------------------------------------------------
    # Velocity control
    # ------------------------------------------------------------------

    def pause_for_duration(self, duration: float, current_time: float) -> None:
        """Pause progression for *duration* time units."""
        self._pause_until = current_time + duration

    def set_velocity_multiplier(self, velocity: float) -> None:
        """Set the base velocity multiplier (clamped to [0.1, 3.0])."""
        self._velocity = max(0.1, min(3.0, velocity))

    # ------------------------------------------------------------------
    # Debugging helpers
    # ------------------------------------------------------------------

    def get_progression_info(self) -> Dict[str, Any]:
        """Detailed progression snapshot for debugging."""
        return {
            "velocity": self._velocity,
            "acceleration": self._acceleration,
            "effective_velocity": self.velocity,
            "is_paused": self.is_paused,
            "pause_until": self._pause_until,
            "confidence_history": self._confidence_history.copy(),
            "confidence_trend": (
                self._confidence_history[-1] - self._confidence_history[0]
                if len(self._confidence_history) > 1
                else 0.0
            ),
        }

    def __repr__(self) -> str:
        return (
            f"Agent(id={self.agent_id!r}, spawn_time={self.spawn_time}, "
            f"state={self._state.value!r}, progress={self._progress:.3f})"
        )


# ---------------------------------------------------------------------------
# Spawn-time generation (used by AgentFactory)
# ---------------------------------------------------------------------------


def generate_spawn_times(count: int, seed: Optional[int] = None) -> List[float]:
    """Generate random spawn times in [0, 1].

    Replicates the OpenSCAD function ``rands(0, 1, N, seed)``.

    Args:
        count: Number of spawn times to generate.
        seed: Optional random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)
    return [random.random() for _ in range(count)]
