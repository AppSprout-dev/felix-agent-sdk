"""Tests for felix_agent_sdk.communication.spoke — Spoke and SpokeManager."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from felix_agent_sdk.communication.central_post import CentralPost
from felix_agent_sdk.communication.messages import Message, MessageType
from felix_agent_sdk.communication.spoke import (
    DeliveryConfirmation,
    Spoke,
    SpokeConnection,
    SpokeManager,
)


# -------------------------------------------------------------------------
# Spoke creation and connection
# -------------------------------------------------------------------------


class TestSpokeCreation:
    def test_spoke_initial_state(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        assert spoke.agent_id == "agent-1"
        assert spoke.is_connected is False
        assert spoke.messages_sent == 0
        assert spoke.messages_received == 0

    def test_spoke_has_unique_spoke_id(self, central_post):
        spoke1 = Spoke(agent_id="agent-1", hub=central_post)
        spoke2 = Spoke(agent_id="agent-2", hub=central_post)
        assert spoke1.spoke_id != spoke2.spoke_id

    def test_connect_by_agent_id(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        result = spoke.connect()
        assert result is True
        assert spoke.is_connected is True
        assert central_post.is_agent_registered("agent-1")

    def test_connect_with_agent_object(self, central_post):
        agent = MagicMock()
        agent.agent_id = "agent-mock"
        agent.agent_type = "research"
        agent.spawn_time = 0.1
        agent.confidence = 0.5
        agent.state = MagicMock(value="active")
        spoke = Spoke(agent_id="agent-mock", hub=central_post, agent=agent)
        result = spoke.connect()
        assert result is True
        assert spoke.is_connected is True

    def test_connect_when_already_connected_returns_true(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        spoke.connect()
        assert spoke.connect() is True  # idempotent

    def test_connect_at_hub_capacity_returns_false(self):
        hub = CentralPost(max_agents=1)
        try:
            hub.register_agent_id("existing")
            spoke = Spoke(agent_id="overflow", hub=hub)
            result = spoke.connect()
            assert result is False
            assert spoke.is_connected is False
        finally:
            hub.shutdown()

    def test_connect_with_metadata(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        spoke.connect(metadata={"role": "critic"})
        assert central_post.is_agent_registered("agent-1")


# -------------------------------------------------------------------------
# Disconnect and reconnect
# -------------------------------------------------------------------------


class TestSpokeDisconnect:
    def test_disconnect(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        spoke.connect()
        spoke.disconnect()
        assert spoke.is_connected is False
        assert not central_post.is_agent_registered("agent-1")

    def test_disconnect_when_not_connected_is_noop(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        spoke.disconnect()  # Should not raise
        assert spoke.is_connected is False

    def test_reconnect(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        spoke.connect()
        result = spoke.reconnect()
        assert result is True
        assert spoke.is_connected is True
        assert central_post.is_agent_registered("agent-1")

    def test_reconnect_with_metadata(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        spoke.connect()
        result = spoke.reconnect(metadata={"updated": True})
        assert result is True


# -------------------------------------------------------------------------
# Send messages
# -------------------------------------------------------------------------


class TestSpokeSendMessage:
    def test_send_message_when_connected(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        spoke.connect()
        confirmation = spoke.send_message(
            MessageType.STATUS_UPDATE,
            {"status": "working"},
        )
        assert confirmation.delivered is True
        assert confirmation.message_id != ""
        assert spoke.messages_sent == 1

    def test_send_message_when_not_connected(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        confirmation = spoke.send_message(
            MessageType.STATUS_UPDATE,
            {"status": "working"},
        )
        assert confirmation.delivered is False
        assert confirmation.error == "Spoke is not connected"
        assert spoke.messages_sent == 0

    def test_send_message_with_receiver(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        spoke.connect()
        confirmation = spoke.send_message(
            MessageType.TASK_ASSIGNMENT,
            {"task": "analyze"},
            receiver_id="agent-2",
        )
        assert confirmation.delivered is True
        # Message should be in the hub's queue
        assert central_post.has_pending_messages()

    def test_multiple_sends_increment_counter(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        spoke.connect()
        for _ in range(5):
            spoke.send_message(MessageType.STATUS_UPDATE, {})
        assert spoke.messages_sent == 5


# -------------------------------------------------------------------------
# Receive messages
# -------------------------------------------------------------------------


class TestSpokeReceiveMessage:
    def test_receive_increments_counter(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        msg = Message(
            sender_id="hub",
            message_type=MessageType.TASK_ASSIGNMENT,
            content={"task": "do it"},
        )
        spoke.receive_message(msg)
        assert spoke.messages_received == 1

    def test_receive_without_handler_queues_message(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        msg = Message(
            sender_id="hub",
            message_type=MessageType.TASK_ASSIGNMENT,
            content={"task": "do it"},
        )
        spoke.receive_message(msg)
        assert spoke.has_pending_messages()
        pending = spoke.get_pending_messages()
        assert len(pending) == 1
        assert pending[0].message_id == msg.message_id

    def test_get_pending_clears_queue(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        msg = Message(
            sender_id="hub",
            message_type=MessageType.TASK_ASSIGNMENT,
            content={},
        )
        spoke.receive_message(msg)
        spoke.get_pending_messages()
        assert not spoke.has_pending_messages()

    def test_receive_with_handler_dispatches(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        received_messages = []
        spoke.register_handler(MessageType.TASK_ASSIGNMENT, lambda m: received_messages.append(m))
        msg = Message(
            sender_id="hub",
            message_type=MessageType.TASK_ASSIGNMENT,
            content={"task": "analyze"},
        )
        spoke.receive_message(msg)
        assert len(received_messages) == 1
        # Message should NOT be in the pending queue when handled
        assert not spoke.has_pending_messages()

    def test_unregister_handler(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        received = []
        spoke.register_handler(MessageType.TASK_ASSIGNMENT, lambda m: received.append(m))
        spoke.unregister_handler(MessageType.TASK_ASSIGNMENT)
        msg = Message(
            sender_id="hub",
            message_type=MessageType.TASK_ASSIGNMENT,
            content={},
        )
        spoke.receive_message(msg)
        assert len(received) == 0
        # Should go to pending queue instead
        assert spoke.has_pending_messages()

    def test_handler_exception_does_not_propagate(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)

        def bad_handler(msg):
            raise RuntimeError("handler error")

        spoke.register_handler(MessageType.TASK_ASSIGNMENT, bad_handler)
        msg = Message(
            sender_id="hub",
            message_type=MessageType.TASK_ASSIGNMENT,
            content={},
        )
        # Should not raise
        spoke.receive_message(msg)
        assert spoke.messages_received == 1


# -------------------------------------------------------------------------
# Delivery tracking and connection info
# -------------------------------------------------------------------------


class TestDeliveryTracking:
    def test_delivery_confirmation_fields(self):
        dc = DeliveryConfirmation(message_id="msg-1", delivered=True)
        assert dc.message_id == "msg-1"
        assert dc.delivered is True
        assert dc.error is None
        assert dc.timestamp > 0

    def test_delivery_confirmation_with_error(self):
        dc = DeliveryConfirmation(
            message_id="msg-1", delivered=False, error="connection lost"
        )
        assert dc.delivered is False
        assert dc.error == "connection lost"

    def test_spoke_connection_fields(self):
        sc = SpokeConnection(agent_id="agent-1", spoke_id="spoke-abc")
        assert sc.agent_id == "agent-1"
        assert sc.spoke_id == "spoke-abc"
        assert sc.is_active is True
        assert sc.messages_sent == 0
        assert sc.messages_received == 0

    def test_get_connection_info(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        spoke.connect()
        spoke.send_message(MessageType.STATUS_UPDATE, {})
        info = spoke.get_connection_info()
        assert info["agent_id"] == "agent-1"
        assert info["spoke_id"] == spoke.spoke_id
        assert info["is_connected"] is True
        assert info["messages_sent"] == 1
        assert info["messages_received"] == 0
        assert info["pending_inbound"] == 0


# -------------------------------------------------------------------------
# Performance metrics (connection-level counters)
# -------------------------------------------------------------------------


class TestSpokePerformanceMetrics:
    def test_sent_counter_matches_connection(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        spoke.connect()
        for _ in range(3):
            spoke.send_message(MessageType.STATUS_UPDATE, {})
        assert spoke.messages_sent == 3
        # The connection record should also track this
        assert spoke._connection.messages_sent == 3

    def test_received_counter_matches_connection(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        spoke.connect()
        for _ in range(4):
            msg = Message(
                sender_id="hub", message_type=MessageType.STATUS_UPDATE, content={}
            )
            spoke.receive_message(msg)
        assert spoke.messages_received == 4
        assert spoke._connection.messages_received == 4

    def test_repr(self, central_post):
        spoke = Spoke(agent_id="agent-1", hub=central_post)
        r = repr(spoke)
        assert "agent-1" in r
        assert "connected=False" in r


# -------------------------------------------------------------------------
# SpokeManager — creation and removal
# -------------------------------------------------------------------------


class TestSpokeManagerCreation:
    def test_create_spoke_auto_connect(self, central_post, spoke_manager):
        spoke = spoke_manager.create_spoke("agent-1")
        assert spoke.is_connected is True
        assert spoke_manager.active_spoke_count == 1

    def test_create_spoke_no_auto_connect(self, central_post, spoke_manager):
        spoke = spoke_manager.create_spoke("agent-1", auto_connect=False)
        assert spoke.is_connected is False

    def test_get_spoke(self, central_post, spoke_manager):
        spoke_manager.create_spoke("agent-1")
        spoke = spoke_manager.get_spoke("agent-1")
        assert spoke is not None
        assert spoke.agent_id == "agent-1"

    def test_get_spoke_nonexistent(self, spoke_manager):
        assert spoke_manager.get_spoke("nonexistent") is None

    def test_remove_spoke(self, central_post, spoke_manager):
        spoke_manager.create_spoke("agent-1")
        result = spoke_manager.remove_spoke("agent-1")
        assert result is True
        assert spoke_manager.get_spoke("agent-1") is None
        assert not central_post.is_agent_registered("agent-1")

    def test_remove_spoke_nonexistent(self, spoke_manager):
        assert spoke_manager.remove_spoke("nonexistent") is False

    def test_get_all_agent_ids(self, central_post, spoke_manager):
        spoke_manager.create_spoke("agent-1")
        spoke_manager.create_spoke("agent-2")
        ids = spoke_manager.get_all_agent_ids()
        assert set(ids) == {"agent-1", "agent-2"}


# -------------------------------------------------------------------------
# SpokeManager — broadcast
# -------------------------------------------------------------------------


class TestSpokeManagerBroadcast:
    def test_broadcast_to_all_connected(self, central_post, spoke_manager):
        spoke_manager.create_spoke("agent-1")
        spoke_manager.create_spoke("agent-2")
        confirmations = spoke_manager.broadcast_message(
            MessageType.PHASE_ANNOUNCE,
            {"phase": "synthesis"},
        )
        assert len(confirmations) == 2
        assert all(c.delivered for c in confirmations)

    def test_broadcast_excludes_specified_ids(self, central_post, spoke_manager):
        spoke_manager.create_spoke("agent-1")
        spoke_manager.create_spoke("agent-2")
        spoke_manager.create_spoke("agent-3")
        confirmations = spoke_manager.broadcast_message(
            MessageType.PHASE_ANNOUNCE,
            {"phase": "analysis"},
            exclude_ids=["agent-2"],
        )
        assert len(confirmations) == 2

    def test_broadcast_skips_disconnected_spokes(self, central_post, spoke_manager):
        spoke1 = spoke_manager.create_spoke("agent-1")
        spoke2 = spoke_manager.create_spoke("agent-2")
        spoke2.disconnect()
        confirmations = spoke_manager.broadcast_message(
            MessageType.STATUS_UPDATE,
            {"status": "working"},
        )
        # Only agent-1 should receive (agent-2 disconnected)
        assert len(confirmations) == 1
        assert confirmations[0].delivered is True

    def test_broadcast_messages_received_by_spokes(self, central_post, spoke_manager):
        spoke1 = spoke_manager.create_spoke("agent-1")
        spoke2 = spoke_manager.create_spoke("agent-2")
        spoke_manager.broadcast_message(
            MessageType.CONVERGENCE_SIGNAL,
            {"confidence": 0.9},
            sender_id="hub",
        )
        # Both spokes should have a pending inbound message
        assert spoke1.has_pending_messages()
        assert spoke2.has_pending_messages()


# -------------------------------------------------------------------------
# SpokeManager — process_all_messages
# -------------------------------------------------------------------------


class TestSpokeManagerProcessAll:
    def test_process_all_directed_messages(self, central_post, spoke_manager):
        spoke1 = spoke_manager.create_spoke("agent-1")
        spoke2 = spoke_manager.create_spoke("agent-2")

        # agent-1 sends to agent-2
        spoke1.send_message(
            MessageType.TASK_ASSIGNMENT,
            {"task": "analyze"},
            receiver_id="agent-2",
        )
        count = spoke_manager.process_all_messages()
        assert count == 1
        # agent-2 should have received the message
        assert spoke2.has_pending_messages()
        # agent-1 should NOT have received it (directed)
        assert not spoke1.has_pending_messages()

    def test_process_all_broadcast_messages(self, central_post, spoke_manager):
        spoke1 = spoke_manager.create_spoke("agent-1")
        spoke2 = spoke_manager.create_spoke("agent-2")

        # agent-1 sends a broadcast (no receiver_id)
        spoke1.send_message(
            MessageType.STATUS_UPDATE,
            {"status": "done"},
        )
        count = spoke_manager.process_all_messages()
        assert count == 1
        # agent-2 should receive it, agent-1 should not (sender exclusion)
        assert spoke2.has_pending_messages()
        assert not spoke1.has_pending_messages()

    def test_process_all_returns_zero_when_empty(self, spoke_manager):
        assert spoke_manager.process_all_messages() == 0


# -------------------------------------------------------------------------
# SpokeManager — shutdown_all
# -------------------------------------------------------------------------


class TestSpokeManagerShutdown:
    def test_shutdown_all_disconnects_spokes(self, central_post):
        manager = SpokeManager(hub=central_post)
        manager.create_spoke("agent-1")
        manager.create_spoke("agent-2")
        manager.shutdown_all()
        assert manager.active_spoke_count == 0
        assert manager.get_all_agent_ids() == []

    def test_shutdown_all_deregisters_from_hub(self, central_post):
        manager = SpokeManager(hub=central_post)
        manager.create_spoke("agent-1")
        manager.shutdown_all()
        assert not central_post.is_agent_registered("agent-1")

    def test_repr(self, central_post, spoke_manager):
        spoke_manager.create_spoke("agent-1")
        r = repr(spoke_manager)
        assert "spokes=1" in r
        assert "active=1" in r
