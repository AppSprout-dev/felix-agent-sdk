"""Spoke — agent-side connection to the CentralPost hub.

Ported from CalebisGross/felix src/communication/spoke.py.
Changes from original:
- Import paths updated to SDK package layout.
- Agent parameter typed via TYPE_CHECKING to avoid circular import.
- print() calls replaced with logger.warning().
- SpokeManager.process_all_messages() simplified: routes through spoke.receive_message().
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from felix_agent_sdk.communication.messages import Message, MessageType

if TYPE_CHECKING:
    from felix_agent_sdk.agents.base import Agent
    from felix_agent_sdk.communication.central_post import CentralPost

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DeliveryConfirmation
# ---------------------------------------------------------------------------


@dataclass
class DeliveryConfirmation:
    """Acknowledgement returned when a message is delivered through a Spoke."""

    message_id: str
    delivered: bool
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# SpokeConnection
# ---------------------------------------------------------------------------


@dataclass
class SpokeConnection:
    """Lightweight record of a registered spoke connection."""

    agent_id: str
    spoke_id: str
    connected_at: float = field(default_factory=time.time)
    is_active: bool = True
    messages_sent: int = 0
    messages_received: int = 0


# ---------------------------------------------------------------------------
# Spoke
# ---------------------------------------------------------------------------


class Spoke:
    """Agent-side connection endpoint for hub-spoke communication.

    Each agent holds one Spoke instance that manages its connection to
    the CentralPost hub.  All messages sent by the agent flow outward
    through ``send_message()``, and inbound messages from the hub arrive
    via ``receive_message()``.

    Args:
        agent_id: Identifier of the owning agent.
        hub: The CentralPost hub this spoke connects to.
        agent: Optional agent object reference (used for metadata extraction).
    """

    def __init__(
        self,
        agent_id: str,
        hub: CentralPost,
        agent: Optional[Agent] = None,
    ) -> None:
        self._agent_id = agent_id
        self._hub = hub
        self._agent = agent
        self._spoke_id: str = str(uuid.uuid4())
        self._is_connected: bool = False
        self._connection: Optional[SpokeConnection] = None
        self._inbound_queue: List[Message] = []
        self._message_handlers: Dict[MessageType, Callable[[Message], None]] = {}
        self._sent_count: int = 0
        self._received_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def agent_id(self) -> str:
        """Identifier of the agent that owns this spoke."""
        return self._agent_id

    @property
    def spoke_id(self) -> str:
        """Unique identifier for this spoke endpoint."""
        return self._spoke_id

    @property
    def is_connected(self) -> bool:
        """True while this spoke has an active hub connection."""
        return self._is_connected

    @property
    def messages_sent(self) -> int:
        """Total messages dispatched to the hub."""
        return self._sent_count

    @property
    def messages_received(self) -> int:
        """Total messages received from the hub."""
        return self._received_count

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Establish this spoke's connection to the hub.

        Registers the agent with CentralPost and marks the spoke active.

        Args:
            metadata: Optional metadata dict forwarded to hub registration.

        Returns:
            True if connection was established successfully, False otherwise.
        """
        if self._is_connected:
            return True

        if self._agent is not None:
            result = self._hub.register_agent(self._agent, metadata)
        else:
            try:
                result = self._hub.register_agent_id(self._agent_id, metadata)
            except RuntimeError:
                logger.warning(
                    "Spoke.connect: hub at capacity, could not register %s", self._agent_id
                )
                return False

        if result is None:
            return False

        self._is_connected = True
        self._connection = SpokeConnection(
            agent_id=self._agent_id,
            spoke_id=self._spoke_id,
        )
        logger.debug("Spoke %s connected for agent %s", self._spoke_id, self._agent_id)
        return True

    def disconnect(self) -> None:
        """Disconnect from the hub and deregister the agent."""
        if not self._is_connected:
            return

        self._hub.deregister_agent(self._agent_id)
        self._is_connected = False
        if self._connection:
            self._connection.is_active = False
        logger.debug("Spoke %s disconnected for agent %s", self._spoke_id, self._agent_id)

    def reconnect(self, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Disconnect and reconnect to the hub.

        Args:
            metadata: Optional metadata to use for re-registration.

        Returns:
            True if reconnection succeeded.
        """
        self.disconnect()
        return self.connect(metadata)

    # ------------------------------------------------------------------
    # Sending messages
    # ------------------------------------------------------------------

    def send_message(
        self,
        message_type: MessageType,
        content: Dict[str, Any],
        receiver_id: Optional[str] = None,
    ) -> DeliveryConfirmation:
        """Build and send a message to the hub.

        Args:
            message_type: The type of message to send.
            content: Message payload dictionary.
            receiver_id: Optional target agent ID for directed messages.

        Returns:
            DeliveryConfirmation indicating success or failure.
        """
        if not self._is_connected:
            return DeliveryConfirmation(
                message_id="",
                delivered=False,
                error="Spoke is not connected",
            )

        message = Message(
            sender_id=self._agent_id,
            message_type=message_type,
            content=content,
            receiver_id=receiver_id,
        )

        try:
            self._hub.queue_message(message)
            self._sent_count += 1
            if self._connection:
                self._connection.messages_sent += 1
            return DeliveryConfirmation(message_id=message.message_id, delivered=True)
        except Exception as exc:
            return DeliveryConfirmation(
                message_id=message.message_id,
                delivered=False,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Receiving messages
    # ------------------------------------------------------------------

    def receive_message(self, message: Message) -> None:
        """Receive an inbound message from the hub.

        Dispatches to a registered handler if one exists for the message
        type, otherwise stores in the inbound queue.

        Args:
            message: The Message delivered by the hub.
        """
        self._received_count += 1
        if self._connection:
            self._connection.messages_received += 1

        handler = self._message_handlers.get(message.message_type)
        if handler:
            try:
                handler(message)
            except Exception:
                logger.exception(
                    "Handler error in spoke %s for message type %s",
                    self._spoke_id,
                    message.message_type.value,
                )
        else:
            self._inbound_queue.append(message)

    def register_handler(
        self, message_type: MessageType, handler: Callable[[Message], None]
    ) -> None:
        """Register a handler for a specific message type.

        Args:
            message_type: Message type this handler should process.
            handler: Callable accepting a single Message argument.
        """
        self._message_handlers[message_type] = handler

    def unregister_handler(self, message_type: MessageType) -> None:
        """Remove a previously registered message handler.

        Args:
            message_type: The message type whose handler should be removed.
        """
        self._message_handlers.pop(message_type, None)

    def get_pending_messages(self) -> List[Message]:
        """Return and clear all messages waiting in the inbound queue.

        Returns:
            List of unhandled inbound messages.
        """
        messages = list(self._inbound_queue)
        self._inbound_queue.clear()
        return messages

    def has_pending_messages(self) -> bool:
        """Return True if there are unhandled inbound messages."""
        return bool(self._inbound_queue)

    # ------------------------------------------------------------------
    # Connection info
    # ------------------------------------------------------------------

    def get_connection_info(self) -> Dict[str, Any]:
        """Return a snapshot of connection metadata."""
        return {
            "agent_id": self._agent_id,
            "spoke_id": self._spoke_id,
            "is_connected": self._is_connected,
            "messages_sent": self._sent_count,
            "messages_received": self._received_count,
            "pending_inbound": len(self._inbound_queue),
        }

    def __repr__(self) -> str:
        return (
            f"Spoke(agent_id={self._agent_id!r}, "
            f"spoke_id={self._spoke_id!r}, "
            f"connected={self._is_connected})"
        )


# ---------------------------------------------------------------------------
# SpokeManager
# ---------------------------------------------------------------------------


class SpokeManager:
    """Manages the collection of Spoke connections in an agent team.

    Typically held by the CentralPost side or a workflow runner that needs
    to create spokes, route inbound messages, and broadcast to all agents.

    Args:
        hub: The CentralPost hub that spokes should connect to.
    """

    def __init__(self, hub: CentralPost) -> None:
        self._hub = hub
        self._spokes: Dict[str, Spoke] = {}

    # ------------------------------------------------------------------
    # Spoke lifecycle
    # ------------------------------------------------------------------

    def create_spoke(
        self,
        agent_id: str,
        agent: Optional[Agent] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_connect: bool = True,
    ) -> Spoke:
        """Create a new Spoke for *agent_id* and optionally connect it.

        Args:
            agent_id: Unique agent identifier.
            agent: Optional Agent object (enables metadata extraction).
            metadata: Additional registration metadata.
            auto_connect: If True, call ``spoke.connect()`` immediately.

        Returns:
            The newly created (and optionally connected) Spoke.
        """
        spoke = Spoke(agent_id=agent_id, hub=self._hub, agent=agent)
        if auto_connect:
            spoke.connect(metadata)
        self._spokes[agent_id] = spoke
        return spoke

    def get_spoke(self, agent_id: str) -> Optional[Spoke]:
        """Return the Spoke for *agent_id*, or None if not found."""
        return self._spokes.get(agent_id)

    def remove_spoke(self, agent_id: str) -> bool:
        """Disconnect and remove the Spoke for *agent_id*.

        Returns:
            True if the spoke existed and was removed, False otherwise.
        """
        spoke = self._spokes.pop(agent_id, None)
        if spoke is None:
            return False
        if spoke.is_connected:
            spoke.disconnect()
        return True

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------

    def broadcast_message(
        self,
        message_type: MessageType,
        content: Dict[str, Any],
        sender_id: str = "hub",
        exclude_ids: Optional[List[str]] = None,
    ) -> List[DeliveryConfirmation]:
        """Deliver a message to all connected spokes except those in *exclude_ids*.

        The message is delivered directly to each spoke's ``receive_message()``
        method (inbound path), not enqueued on the hub.

        Args:
            message_type: The type of message to broadcast.
            content: Message payload.
            sender_id: Identifier of the broadcast originator (default "hub").
            exclude_ids: Agent IDs to skip.

        Returns:
            List of DeliveryConfirmation, one per recipient spoke.
        """
        exclude = set(exclude_ids or [])
        confirmations: List[DeliveryConfirmation] = []

        for agent_id, spoke in list(self._spokes.items()):
            if agent_id in exclude:
                continue
            if not spoke.is_connected:
                logger.warning(
                    "broadcast_message: skipping disconnected spoke for agent %s",
                    agent_id,
                )
                continue

            message = Message(
                sender_id=sender_id,
                message_type=message_type,
                content=content,
            )
            try:
                spoke.receive_message(message)
                confirmations.append(
                    DeliveryConfirmation(message_id=message.message_id, delivered=True)
                )
            except Exception as exc:
                confirmations.append(
                    DeliveryConfirmation(
                        message_id=message.message_id,
                        delivered=False,
                        error=str(exc),
                    )
                )

        return confirmations

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    def process_all_messages(self) -> int:
        """Route all queued hub messages through their respective spokes.

        Drains the hub's sync queue and delivers each message to the
        target spoke (via ``receiver_id``) or broadcasts to all spokes
        when no receiver is specified.

        Returns:
            Number of messages processed.
        """
        count = 0
        while self._hub.has_pending_messages():
            message = self._hub.process_next_message()
            if message is None:
                break
            count += 1

            # Directed delivery
            if message.receiver_id and message.receiver_id in self._spokes:
                spoke = self._spokes[message.receiver_id]
                spoke.receive_message(message)
            # Broadcast to all spokes except the sender
            else:
                for agent_id, spoke in list(self._spokes.items()):
                    if agent_id != message.sender_id and spoke.is_connected:
                        spoke.receive_message(message)

        return count

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown_all(self) -> None:
        """Disconnect all managed spokes."""
        for agent_id in list(self._spokes.keys()):
            self.remove_spoke(agent_id)
        logger.debug("SpokeManager: all spokes disconnected")

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def active_spoke_count(self) -> int:
        """Number of currently connected spokes."""
        return sum(1 for s in self._spokes.values() if s.is_connected)

    def get_all_agent_ids(self) -> List[str]:
        """Return the list of agent IDs with managed spokes."""
        return list(self._spokes.keys())

    def __repr__(self) -> str:
        return f"SpokeManager(spokes={len(self._spokes)}, active={self.active_spoke_count})"
