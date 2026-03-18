"""CentralPost — Felix hub-spoke O(N) coordination hub.

Ported from CalebisGross/felix src/communication/central_post.py.
Refactored: no SQLite, no memory system integration, no GUI, no web search.
Provider slot reserved for future use (not yet wired to message handling).
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from felix_agent_sdk.communication.messages import Message, MessageType
from felix_agent_sdk.communication.registry import AgentRegistry

if TYPE_CHECKING:
    from felix_agent_sdk.agents.base import Agent
    from felix_agent_sdk.providers.base import BaseProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent lifecycle event enum
# ---------------------------------------------------------------------------


class AgentLifecycleEvent(Enum):
    """Events emitted during an agent's lifecycle."""

    SPAWNED = "spawned"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# CentralPost
# ---------------------------------------------------------------------------


class CentralPost:
    """Hub-spoke coordination hub for Felix agent teams.

    All inter-agent messages route through this single hub, ensuring O(N)
    communication complexity instead of O(N²) direct mesh connections.

    Architecture:
        - Sync ``Queue`` for single-threaded / test usage.
        - Async ``asyncio.Queue`` (lazy-initialised) for async runtimes.
        - ``AgentRegistry`` tracks helix positions, phases, and confidence.
        - Lifecycle callbacks notify callers when agents spawn/complete/fail.

    Args:
        max_agents: Maximum number of agents that may be registered simultaneously.
        enable_metrics: Reserved flag for future metrics collection.
        provider: Optional provider reference (slot — not used by message handling yet).
    """

    def __init__(
        self,
        max_agents: int = 25,
        enable_metrics: bool = False,
        provider: Optional[BaseProvider] = None,
    ) -> None:
        self._max_agents = max_agents
        self._enable_metrics = enable_metrics
        self._provider = provider

        # Agent tracking
        self._registered_agents: Dict[str, Any] = {}
        self._connection_times: Dict[str, float] = {}

        # Registry
        self.agent_registry = AgentRegistry()

        # Sync message queue
        self._message_queue: Queue[Message] = Queue()
        self._processed_messages: List[Message] = []
        self._total_messages_processed: int = 0

        # Async message queue (lazy — created on first async use)
        self._async_queue: Optional[asyncio.Queue] = None  # type: ignore[type-arg]

        # Lifecycle callbacks: event -> list of callables
        self._lifecycle_callbacks: Dict[AgentLifecycleEvent, List[Callable]] = {
            event: [] for event in AgentLifecycleEvent
        }

        self._is_active: bool = True
        logger.debug("CentralPost initialised (max_agents=%d)", max_agents)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_connections(self) -> int:
        """Number of currently registered agents."""
        return len(self._registered_agents)

    @property
    def message_queue_size(self) -> int:
        """Number of messages waiting in the sync queue."""
        return self._message_queue.qsize()

    @property
    def is_active(self) -> bool:
        """True until ``shutdown()`` is called."""
        return self._is_active

    @property
    def total_messages_processed(self) -> int:
        """Cumulative count of messages processed (sync + async)."""
        return self._total_messages_processed

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(
        self, agent: Agent, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Register an agent object and extract metadata from its attributes.

        Args:
            agent: Agent instance with at minimum an ``agent_id`` attribute.
            metadata: Extra metadata to merge (takes precedence over auto-extracted).

        Returns:
            The ``agent_id`` string on success, or ``None`` if the hub is at capacity.
        """
        agent_id: str = getattr(agent, "agent_id", None) or str(id(agent))

        if (
            len(self._registered_agents) >= self._max_agents
            and agent_id not in self._registered_agents
        ):
            logger.warning(
                "CentralPost at capacity (%d/%d) — cannot register %s",
                len(self._registered_agents),
                self._max_agents,
                agent_id,
            )
            return None

        auto_meta: Dict[str, Any] = {}
        for attr in ("agent_type", "spawn_time", "confidence", "state"):
            val = getattr(agent, attr, None)
            if val is not None:
                auto_meta[attr] = val.value if hasattr(val, "value") else val

        combined_meta = {**auto_meta, **(metadata or {})}
        self._registered_agents[agent_id] = combined_meta
        self._connection_times[agent_id] = time.time()
        self.agent_registry.register_agent(agent_id, combined_meta)

        logger.debug("Registered agent object %s", agent_id)
        return agent_id

    def register_agent_id(self, agent_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Lightweight registration by ID only — no agent object required.

        Args:
            agent_id: Unique identifier string.
            metadata: Optional metadata dict.

        Returns:
            The ``agent_id`` string.

        Raises:
            RuntimeError: If the hub is at capacity.
        """
        if (
            len(self._registered_agents) >= self._max_agents
            and agent_id not in self._registered_agents
        ):
            raise RuntimeError(
                f"CentralPost at capacity ({self._max_agents} agents) — cannot register {agent_id!r}"
            )

        self._registered_agents[agent_id] = metadata or {}
        self._connection_times[agent_id] = time.time()
        self.agent_registry.register_agent(agent_id, metadata)
        logger.debug("Registered agent ID %s", agent_id)
        return agent_id

    def deregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the hub.

        Args:
            agent_id: Identifier of the agent to remove.

        Returns:
            True if the agent was registered and has been removed, False otherwise.
        """
        if agent_id not in self._registered_agents:
            return False

        del self._registered_agents[agent_id]
        self._connection_times.pop(agent_id, None)
        self.agent_registry.deregister_agent(agent_id)
        logger.debug("Deregistered agent %s", agent_id)
        return True

    def is_agent_registered(self, agent_id: str) -> bool:
        """Return True if *agent_id* is currently registered."""
        return agent_id in self._registered_agents

    # ------------------------------------------------------------------
    # Sync message queue
    # ------------------------------------------------------------------

    def queue_message(self, message: Message) -> None:
        """Enqueue a message for synchronous processing.

        Args:
            message: The Message to enqueue.
        """
        self._message_queue.put(message)
        logger.debug(
            "Queued message %s [%s] from %s",
            message.message_id,
            message.message_type.value,
            message.sender_id,
        )

    def has_pending_messages(self) -> bool:
        """Return True if there are messages waiting in the sync queue."""
        return not self._message_queue.empty()

    def process_next_message(self) -> Optional[Message]:
        """Dequeue and handle the next message synchronously.

        Returns:
            The processed Message, or None if the queue was empty.
        """
        try:
            message = self._message_queue.get_nowait()
        except Empty:
            return None

        self._handle_message(message)
        self._processed_messages.append(message)
        self._total_messages_processed += 1
        return message

    # ------------------------------------------------------------------
    # Async message queue
    # ------------------------------------------------------------------

    def _ensure_async_queue(self) -> asyncio.Queue:  # type: ignore[type-arg]
        """Lazily create the async queue on first use."""
        if self._async_queue is None:
            self._async_queue = asyncio.Queue()
        return self._async_queue

    async def queue_message_async(self, message: Message) -> None:
        """Enqueue a message for asynchronous processing.

        Args:
            message: The Message to enqueue.
        """
        await self._ensure_async_queue().put(message)
        logger.debug(
            "Async queued message %s [%s] from %s",
            message.message_id,
            message.message_type.value,
            message.sender_id,
        )

    async def process_next_message_async(self) -> Optional[Message]:
        """Dequeue and handle the next message asynchronously.

        Returns:
            The processed Message, or None if the queue was empty.
        """
        q = self._ensure_async_queue()
        try:
            message = q.get_nowait()
        except asyncio.QueueEmpty:
            return None

        self._handle_message(message)
        self._processed_messages.append(message)
        self._total_messages_processed += 1
        return message

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    def _handle_message(self, message: Message) -> None:
        """Dispatch a message to the appropriate type-specific handler."""
        dispatch: Dict[MessageType, Callable[[Message], None]] = {
            MessageType.TASK_REQUEST: self._handle_task_request,
            MessageType.TASK_ASSIGNMENT: self._handle_task_assignment,
            MessageType.STATUS_UPDATE: self._handle_status_update,
            MessageType.TASK_COMPLETE: self._handle_task_complete,
            MessageType.ERROR_REPORT: self._handle_error_report,
            MessageType.PHASE_ANNOUNCE: self._handle_phase_announce,
            MessageType.CONVERGENCE_SIGNAL: self._handle_convergence_signal,
            MessageType.COLLABORATION_REQUEST: self._handle_collaboration_request,
            MessageType.SYNTHESIS_READY: self._handle_synthesis_ready,
            MessageType.AGENT_QUERY: self._handle_agent_query,
            MessageType.AGENT_DISCOVERY: self._handle_agent_discovery,
            MessageType.SYNTHESIS_FEEDBACK: self._handle_synthesis_feedback,
            MessageType.CONTRIBUTION_EVALUATION: self._handle_contribution_evaluation,
            MessageType.IMPROVEMENT_REQUEST: self._handle_improvement_request,
        }
        handler = dispatch.get(message.message_type)
        if handler:
            handler(message)
        else:
            logger.warning("No handler for message type %s", message.message_type)

    # ------------------------------------------------------------------
    # Individual message handlers
    # ------------------------------------------------------------------

    def _handle_task_request(self, message: Message) -> None:
        """A sender is requesting a task be handled."""
        logger.debug(
            "TASK_REQUEST from %s: %s",
            message.sender_id,
            message.content.get("task_description", ""),
        )

    def _handle_task_assignment(self, message: Message) -> None:
        """A task has been assigned to an agent."""
        assigned_to = message.content.get("assigned_to", message.receiver_id)
        logger.debug(
            "TASK_ASSIGNMENT from %s to %s",
            message.sender_id,
            assigned_to,
        )

    def _handle_status_update(self, message: Message) -> None:
        """An agent reports a progress or confidence update."""
        sender_id = message.sender_id
        content = message.content

        # Update position if provided
        position_info: Optional[Dict[str, Any]] = content.get("position_info")
        if position_info:
            self.agent_registry.update_agent_position(sender_id, position_info)

        # Update performance metrics if provided
        metrics: Dict[str, Any] = {}
        if "confidence" in content:
            metrics["confidence"] = content["confidence"]
        if "tasks_completed" in content:
            metrics["tasks_completed"] = content["tasks_completed"]
        if metrics:
            self.agent_registry.update_agent_performance(sender_id, metrics)

    def _handle_task_complete(self, message: Message) -> None:
        """An agent reports task completion."""
        logger.debug("TASK_COMPLETE from %s", message.sender_id)
        if "confidence" in message.content:
            self.agent_registry.update_agent_performance(
                message.sender_id, {"confidence": message.content["confidence"]}
            )
        self.emit_lifecycle_event(AgentLifecycleEvent.COMPLETED, message.sender_id)

    def _handle_error_report(self, message: Message) -> None:
        """An agent reports an error condition."""
        logger.warning(
            "ERROR_REPORT from %s: %s",
            message.sender_id,
            message.content.get("error", "unknown error"),
        )
        self.agent_registry.update_agent_performance(message.sender_id, {"errors": 1})
        self.emit_lifecycle_event(AgentLifecycleEvent.FAILED, message.sender_id)

    def _handle_phase_announce(self, message: Message) -> None:
        """An agent announces a phase transition."""
        phase = message.content.get("phase", "unknown")
        logger.debug("PHASE_ANNOUNCE from %s: phase=%s", message.sender_id, phase)
        depth_ratio = message.content.get("depth_ratio")
        if depth_ratio is not None:
            self.agent_registry.update_agent_position(
                message.sender_id, {"depth_ratio": depth_ratio, "phase": phase}
            )

    def _handle_convergence_signal(self, message: Message) -> None:
        """An agent signals team-wide convergence."""
        logger.debug("CONVERGENCE_SIGNAL from %s", message.sender_id)
        confidence = message.content.get("confidence")
        if confidence is not None:
            self.agent_registry.update_agent_performance(
                message.sender_id, {"confidence": confidence}
            )

    def _handle_collaboration_request(self, message: Message) -> None:
        """An agent requests collaboration with another."""
        target = message.content.get("target_agent_id", message.receiver_id)
        if target:
            self.agent_registry.record_collaboration(message.sender_id, target)
        logger.debug(
            "COLLABORATION_REQUEST from %s targeting %s",
            message.sender_id,
            target,
        )

    def _handle_synthesis_ready(self, message: Message) -> None:
        """An agent signals it is ready for synthesis."""
        logger.debug("SYNTHESIS_READY from %s", message.sender_id)
        self.agent_registry.update_agent_position(
            message.sender_id, {"depth_ratio": message.content.get("depth_ratio", 1.0)}
        )

    def _handle_agent_query(self, message: Message) -> None:
        """An agent is querying hub state."""
        query_type = message.content.get("query_type", "team_composition")
        logger.debug("AGENT_QUERY from %s: %s", message.sender_id, query_type)

    def _handle_agent_discovery(self, message: Message) -> None:
        """An agent announces itself for discovery by peers."""
        metadata = message.content.get("metadata", {})
        if metadata:
            self.agent_registry.update_agent_performance(message.sender_id, metadata)
        logger.debug("AGENT_DISCOVERY from %s", message.sender_id)

    def _handle_synthesis_feedback(self, message: Message) -> None:
        """Feedback on a synthesis output."""
        logger.debug("SYNTHESIS_FEEDBACK from %s", message.sender_id)

    def _handle_contribution_evaluation(self, message: Message) -> None:
        """Evaluation of an agent's contribution quality."""
        score = message.content.get("score")
        if score is not None:
            self.agent_registry.update_agent_performance(
                message.sender_id, {"confidence": float(score)}
            )
        logger.debug("CONTRIBUTION_EVALUATION for %s", message.sender_id)

    def _handle_improvement_request(self, message: Message) -> None:
        """A request for improvement on previous output."""
        logger.debug("IMPROVEMENT_REQUEST from %s", message.sender_id)

    # ------------------------------------------------------------------
    # Message history
    # ------------------------------------------------------------------

    def get_recent_messages(
        self,
        count: int = 20,
        message_type: Optional[MessageType] = None,
    ) -> List[Message]:
        """Return the most recent processed messages, optionally filtered by type.

        Args:
            count: Maximum number of messages to return.
            message_type: If given, only return messages of this type.

        Returns:
            List of Message objects, most recent last.
        """
        messages = self._processed_messages
        if message_type is not None:
            messages = [m for m in messages if m.message_type == message_type]
        return messages[-count:]

    # ------------------------------------------------------------------
    # Team awareness
    # ------------------------------------------------------------------

    def query_team_awareness(self, query_type: str) -> Dict[str, Any]:
        """Query the hub for team-wide awareness data.

        Args:
            query_type: One of "team_composition", "phase_distribution", "confidence".

        Returns:
            Dict with relevant team data for the requested query type.
        """
        if query_type == "team_composition":
            return {
                "active_agents": self.agent_registry.get_active_agents(),
                "active_count": self.active_connections,
                "max_agents": self._max_agents,
            }

        if query_type == "phase_distribution":
            return {
                "exploration": self.agent_registry.get_agents_in_phase("exploration"),
                "analysis": self.agent_registry.get_agents_in_phase("analysis"),
                "synthesis": self.agent_registry.get_agents_in_phase("synthesis"),
            }

        if query_type == "confidence":
            convergence = self.agent_registry.get_convergence_status()
            return {
                "confidence_trend": convergence["confidence_trend"],
                "synthesis_ready": convergence["synthesis_ready"],
                "collaboration_density": convergence["collaboration_density"],
            }

        # Unknown query type — return convergence status as default
        return self.agent_registry.get_convergence_status()

    # ------------------------------------------------------------------
    # Lifecycle events
    # ------------------------------------------------------------------

    def add_lifecycle_callback(
        self,
        event: AgentLifecycleEvent,
        callback: Callable[[str], None],
    ) -> None:
        """Register a callback to be invoked when *event* fires.

        Args:
            event: The lifecycle event to listen for.
            callback: Callable receiving the agent_id as its sole argument.
        """
        self._lifecycle_callbacks[event].append(callback)

    def remove_lifecycle_callback(
        self,
        event: AgentLifecycleEvent,
        callback: Callable[[str], None],
    ) -> None:
        """Unregister a previously added callback.

        Args:
            event: The lifecycle event the callback was registered for.
            callback: The callback to remove.
        """
        callbacks = self._lifecycle_callbacks.get(event, [])
        if callback in callbacks:
            callbacks.remove(callback)

    def emit_lifecycle_event(self, event: AgentLifecycleEvent, agent_id: str) -> None:
        """Invoke all callbacks registered for *event*.

        Args:
            event: The lifecycle event that occurred.
            agent_id: The agent ID to pass to each callback.
        """
        for callback in self._lifecycle_callbacks.get(event, []):
            try:
                callback(agent_id)
            except Exception:
                logger.exception(
                    "Lifecycle callback error for event %s, agent %s",
                    event.value,
                    agent_id,
                )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Shutdown the hub synchronously, draining and clearing all state."""
        self._is_active = False

        # Drain sync queue
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except Empty:
                break

        self._registered_agents.clear()
        self._connection_times.clear()
        self._processed_messages.clear()
        logger.debug("CentralPost shut down")

    async def shutdown_async(self) -> None:
        """Shutdown the hub asynchronously, draining the async queue."""
        self._is_active = False

        if self._async_queue is not None:
            while not self._async_queue.empty():
                try:
                    self._async_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        self._registered_agents.clear()
        self._connection_times.clear()
        self._processed_messages.clear()
        logger.debug("CentralPost async shutdown complete")
