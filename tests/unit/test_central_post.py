"""Tests for felix_agent_sdk.communication.central_post — CentralPost hub."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from felix_agent_sdk.communication.central_post import (
    AgentLifecycleEvent,
    CentralPost,
)
from felix_agent_sdk.communication.messages import Message, MessageType
from felix_agent_sdk.communication.registry import AgentRegistry


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _make_mock_agent(
    agent_id: str = "agent-001",
    agent_type: str = "research",
    spawn_time: float = 0.2,
    confidence: float = 0.5,
) -> MagicMock:
    """Create a MagicMock agent with the attributes CentralPost expects."""
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.agent_type = agent_type
    agent.spawn_time = spawn_time
    agent.confidence = confidence
    agent.state = MagicMock(value="active")
    return agent


def _make_message(
    sender_id: str = "agent-001",
    message_type: MessageType = MessageType.STATUS_UPDATE,
    content: dict | None = None,
    receiver_id: str | None = None,
) -> Message:
    return Message(
        sender_id=sender_id,
        message_type=message_type,
        content=content or {},
        receiver_id=receiver_id,
    )


# -------------------------------------------------------------------------
# Construction
# -------------------------------------------------------------------------


class TestCentralPostConstruction:
    def test_default_max_agents(self):
        hub = CentralPost()
        assert hub._max_agents == 25
        hub.shutdown()

    def test_custom_max_agents(self, central_post):
        assert central_post._max_agents == 10

    def test_is_active_on_creation(self, central_post):
        assert central_post.is_active is True

    def test_no_agents_on_creation(self, central_post):
        assert central_post.active_connections == 0

    def test_empty_message_queue(self, central_post):
        assert central_post.message_queue_size == 0

    def test_total_messages_processed_is_zero(self, central_post):
        assert central_post.total_messages_processed == 0

    def test_has_agent_registry(self, central_post):
        assert isinstance(central_post.agent_registry, AgentRegistry)


# -------------------------------------------------------------------------
# Agent registration (object)
# -------------------------------------------------------------------------


class TestAgentRegistrationObject:
    def test_register_agent_object(self, central_post):
        agent = _make_mock_agent("agent-001")
        result = central_post.register_agent(agent)
        assert result == "agent-001"
        assert central_post.active_connections == 1

    def test_register_extracts_metadata(self, central_post):
        agent = _make_mock_agent("agent-001", agent_type="analysis")
        central_post.register_agent(agent)
        assert central_post.is_agent_registered("agent-001")

    def test_register_with_extra_metadata(self, central_post):
        agent = _make_mock_agent("agent-001")
        central_post.register_agent(agent, metadata={"priority": "high"})
        assert central_post.is_agent_registered("agent-001")

    def test_register_at_capacity_returns_none(self, central_post):
        # Fill to capacity (max_agents=10)
        for i in range(10):
            agent = _make_mock_agent(f"agent-{i:03d}")
            central_post.register_agent(agent)
        # 11th registration should fail
        overflow_agent = _make_mock_agent("overflow")
        result = central_post.register_agent(overflow_agent)
        assert result is None

    def test_re_register_same_agent_succeeds(self, central_post):
        """Re-registering an already registered agent updates, doesn't fail."""
        agent = _make_mock_agent("agent-001")
        central_post.register_agent(agent)
        result = central_post.register_agent(agent)
        assert result == "agent-001"

    def test_agent_without_agent_id_uses_id(self, central_post):
        agent = MagicMock()
        agent.agent_id = None
        agent.agent_type = None
        agent.spawn_time = None
        agent.confidence = None
        agent.state = None
        result = central_post.register_agent(agent)
        assert result is not None


# -------------------------------------------------------------------------
# Agent registration (ID only)
# -------------------------------------------------------------------------


class TestAgentRegistrationId:
    def test_register_agent_id(self, central_post):
        result = central_post.register_agent_id("agent-simple")
        assert result == "agent-simple"
        assert central_post.active_connections == 1

    def test_register_agent_id_with_metadata(self, central_post):
        central_post.register_agent_id("agent-1", metadata={"role": "critic"})
        assert central_post.is_agent_registered("agent-1")

    def test_register_agent_id_at_capacity_raises(self, central_post):
        for i in range(10):
            central_post.register_agent_id(f"agent-{i}")
        with pytest.raises(RuntimeError, match="at capacity"):
            central_post.register_agent_id("overflow")

    def test_re_register_same_id_succeeds_at_capacity(self, central_post):
        """Re-registering an existing ID even at capacity should work."""
        for i in range(10):
            central_post.register_agent_id(f"agent-{i}")
        # Re-register agent-0 (already registered) — should not raise
        result = central_post.register_agent_id("agent-0")
        assert result == "agent-0"


# -------------------------------------------------------------------------
# Deregistration
# -------------------------------------------------------------------------


class TestDeregistration:
    def test_deregister_registered_agent(self, central_post):
        central_post.register_agent_id("agent-1")
        assert central_post.deregister_agent("agent-1") is True
        assert not central_post.is_agent_registered("agent-1")
        assert central_post.active_connections == 0

    def test_deregister_unknown_agent_returns_false(self, central_post):
        assert central_post.deregister_agent("nonexistent") is False

    def test_deregister_frees_capacity(self, central_post):
        for i in range(10):
            central_post.register_agent_id(f"agent-{i}")
        central_post.deregister_agent("agent-0")
        # Should be able to register a new agent now
        result = central_post.register_agent_id("agent-new")
        assert result == "agent-new"


# -------------------------------------------------------------------------
# Message queuing (FIFO)
# -------------------------------------------------------------------------


class TestMessageQueuing:
    def test_queue_message_increases_size(self, central_post):
        msg = _make_message()
        central_post.queue_message(msg)
        assert central_post.message_queue_size == 1

    def test_has_pending_messages(self, central_post):
        assert central_post.has_pending_messages() is False
        central_post.queue_message(_make_message())
        assert central_post.has_pending_messages() is True

    def test_fifo_ordering(self, central_post):
        msg1 = _make_message(sender_id="first")
        msg2 = _make_message(sender_id="second")
        msg3 = _make_message(sender_id="third")
        central_post.queue_message(msg1)
        central_post.queue_message(msg2)
        central_post.queue_message(msg3)

        processed1 = central_post.process_next_message()
        processed2 = central_post.process_next_message()
        processed3 = central_post.process_next_message()

        assert processed1.sender_id == "first"
        assert processed2.sender_id == "second"
        assert processed3.sender_id == "third"

    def test_process_empty_queue_returns_none(self, central_post):
        assert central_post.process_next_message() is None


# -------------------------------------------------------------------------
# Message processing and handler dispatch
# -------------------------------------------------------------------------


class TestMessageProcessing:
    def test_process_increments_total(self, central_post):
        central_post.queue_message(_make_message())
        central_post.process_next_message()
        assert central_post.total_messages_processed == 1

    def test_processed_messages_stored(self, central_post):
        msg = _make_message()
        central_post.queue_message(msg)
        central_post.process_next_message()
        recent = central_post.get_recent_messages(count=10)
        assert len(recent) == 1
        assert recent[0].message_id == msg.message_id

    def test_get_recent_messages_filtered_by_type(self, central_post):
        msg1 = _make_message(message_type=MessageType.STATUS_UPDATE)
        msg2 = _make_message(message_type=MessageType.TASK_COMPLETE)
        central_post.queue_message(msg1)
        central_post.queue_message(msg2)
        central_post.process_next_message()
        central_post.process_next_message()

        status_msgs = central_post.get_recent_messages(
            count=10, message_type=MessageType.STATUS_UPDATE
        )
        assert len(status_msgs) == 1
        assert status_msgs[0].message_type == MessageType.STATUS_UPDATE

    def test_status_update_handler_updates_registry(self, central_post):
        central_post.register_agent_id("agent-1")
        msg = _make_message(
            sender_id="agent-1",
            message_type=MessageType.STATUS_UPDATE,
            content={"confidence": 0.9},
        )
        central_post.queue_message(msg)
        central_post.process_next_message()
        info = central_post.agent_registry.get_agent_info("agent-1")
        assert info["performance"]["confidence"] == 0.9

    def test_status_update_with_position_info(self, central_post):
        central_post.register_agent_id("agent-1")
        msg = _make_message(
            sender_id="agent-1",
            message_type=MessageType.STATUS_UPDATE,
            content={"position_info": {"depth_ratio": 0.6}},
        )
        central_post.queue_message(msg)
        central_post.process_next_message()
        info = central_post.agent_registry.get_agent_info("agent-1")
        assert info["position"]["depth_ratio"] == 0.6

    def test_task_complete_emits_completed_event(self, central_post):
        central_post.register_agent_id("agent-1")
        callback_log = []
        central_post.add_lifecycle_callback(
            AgentLifecycleEvent.COMPLETED, lambda aid: callback_log.append(aid)
        )
        msg = _make_message(
            sender_id="agent-1",
            message_type=MessageType.TASK_COMPLETE,
            content={"confidence": 0.95},
        )
        central_post.queue_message(msg)
        central_post.process_next_message()
        assert "agent-1" in callback_log

    def test_error_report_emits_failed_event(self, central_post):
        central_post.register_agent_id("agent-1")
        callback_log = []
        central_post.add_lifecycle_callback(
            AgentLifecycleEvent.FAILED, lambda aid: callback_log.append(aid)
        )
        msg = _make_message(
            sender_id="agent-1",
            message_type=MessageType.ERROR_REPORT,
            content={"error": "something broke"},
        )
        central_post.queue_message(msg)
        central_post.process_next_message()
        assert "agent-1" in callback_log

    def test_collaboration_request_records_collaboration(self, central_post):
        central_post.register_agent_id("agent-1")
        central_post.register_agent_id("agent-2")
        msg = _make_message(
            sender_id="agent-1",
            message_type=MessageType.COLLABORATION_REQUEST,
            content={"target_agent_id": "agent-2"},
        )
        central_post.queue_message(msg)
        central_post.process_next_message()
        # Collaboration should be recorded in the registry
        status = central_post.agent_registry.get_convergence_status()
        assert status["collaboration_density"] > 0.0

    def test_all_14_message_types_dispatched(self, central_post):
        """Every MessageType has a handler and doesn't raise."""
        central_post.register_agent_id("agent-1")
        for mt in MessageType:
            msg = _make_message(
                sender_id="agent-1",
                message_type=mt,
                content={"depth_ratio": 0.8, "phase": "synthesis"},
            )
            central_post.queue_message(msg)
            result = central_post.process_next_message()
            assert result is not None


# -------------------------------------------------------------------------
# Team awareness queries
# -------------------------------------------------------------------------


class TestTeamAwareness:
    def test_query_team_composition(self, central_post):
        central_post.register_agent_id("agent-1")
        central_post.register_agent_id("agent-2")
        result = central_post.query_team_awareness("team_composition")
        assert result["active_count"] == 2
        assert result["max_agents"] == 10
        assert set(result["active_agents"]) == {"agent-1", "agent-2"}

    def test_query_phase_distribution(self, central_post):
        central_post.register_agent_id("agent-1")
        central_post.register_agent_id("agent-2")
        central_post.agent_registry.update_agent_position("agent-1", {"depth_ratio": 0.1})
        central_post.agent_registry.update_agent_position("agent-2", {"depth_ratio": 0.8})
        result = central_post.query_team_awareness("phase_distribution")
        assert "agent-1" in result["exploration"]
        assert "agent-2" in result["synthesis"]

    def test_query_confidence(self, central_post):
        result = central_post.query_team_awareness("confidence")
        assert "confidence_trend" in result
        assert "synthesis_ready" in result
        assert "collaboration_density" in result

    def test_query_unknown_type_returns_convergence(self, central_post):
        result = central_post.query_team_awareness("something_unknown")
        assert "confidence_trend" in result
        assert "active_agent_count" in result


# -------------------------------------------------------------------------
# Lifecycle callbacks
# -------------------------------------------------------------------------


class TestLifecycleCallbacks:
    def test_add_and_emit_callback(self, central_post):
        log = []
        central_post.add_lifecycle_callback(
            AgentLifecycleEvent.SPAWNED, lambda aid: log.append(aid)
        )
        central_post.emit_lifecycle_event(AgentLifecycleEvent.SPAWNED, "agent-1")
        assert log == ["agent-1"]

    def test_multiple_callbacks_for_same_event(self, central_post):
        log1 = []
        log2 = []
        central_post.add_lifecycle_callback(
            AgentLifecycleEvent.COMPLETED, lambda aid: log1.append(aid)
        )
        central_post.add_lifecycle_callback(
            AgentLifecycleEvent.COMPLETED, lambda aid: log2.append(aid)
        )
        central_post.emit_lifecycle_event(AgentLifecycleEvent.COMPLETED, "agent-1")
        assert log1 == ["agent-1"]
        assert log2 == ["agent-1"]

    def test_remove_lifecycle_callback(self, central_post):
        log = []
        def cb(aid):
            return log.append(aid)
        central_post.add_lifecycle_callback(AgentLifecycleEvent.FAILED, cb)
        central_post.remove_lifecycle_callback(AgentLifecycleEvent.FAILED, cb)
        central_post.emit_lifecycle_event(AgentLifecycleEvent.FAILED, "agent-1")
        assert log == []

    def test_callback_exception_does_not_propagate(self, central_post):
        def bad_callback(aid):
            raise RuntimeError("callback error")

        central_post.add_lifecycle_callback(AgentLifecycleEvent.SPAWNED, bad_callback)
        # Should not raise
        central_post.emit_lifecycle_event(AgentLifecycleEvent.SPAWNED, "agent-1")

    def test_lifecycle_event_enum_values(self):
        assert AgentLifecycleEvent.SPAWNED.value == "spawned"
        assert AgentLifecycleEvent.COMPLETED.value == "completed"
        assert AgentLifecycleEvent.FAILED.value == "failed"


# -------------------------------------------------------------------------
# Shutdown
# -------------------------------------------------------------------------


class TestCentralPostShutdown:
    def test_shutdown_sets_inactive(self):
        hub = CentralPost(max_agents=5)
        hub.shutdown()
        assert hub.is_active is False

    def test_shutdown_drains_queue(self):
        hub = CentralPost(max_agents=5)
        hub.queue_message(_make_message())
        hub.queue_message(_make_message())
        hub.shutdown()
        assert hub.message_queue_size == 0

    def test_shutdown_clears_agents(self):
        hub = CentralPost(max_agents=5)
        hub.register_agent_id("agent-1")
        hub.shutdown()
        assert hub.active_connections == 0

    def test_shutdown_clears_processed_messages(self):
        hub = CentralPost(max_agents=5)
        hub.queue_message(_make_message())
        hub.process_next_message()
        hub.shutdown()
        assert hub.get_recent_messages() == []
