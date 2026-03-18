"""Tests for felix_agent_sdk.communication.messages — MessageType and Message."""

from __future__ import annotations

import time
import uuid

import pytest

from felix_agent_sdk.communication.messages import Message, MessageType


# -------------------------------------------------------------------------
# MessageType enum
# -------------------------------------------------------------------------


class TestMessageType:
    """Verify all 14 message types exist with correct values."""

    def test_enum_has_14_members(self):
        assert len(MessageType) == 14

    def test_core_task_lifecycle_types(self):
        assert MessageType.TASK_REQUEST.value == "task_request"
        assert MessageType.TASK_ASSIGNMENT.value == "task_assignment"
        assert MessageType.STATUS_UPDATE.value == "status_update"
        assert MessageType.TASK_COMPLETE.value == "task_complete"
        assert MessageType.ERROR_REPORT.value == "error_report"

    def test_phase_aware_coordination_types(self):
        assert MessageType.PHASE_ANNOUNCE.value == "phase_announce"
        assert MessageType.CONVERGENCE_SIGNAL.value == "convergence_signal"
        assert MessageType.COLLABORATION_REQUEST.value == "collaboration_request"
        assert MessageType.SYNTHESIS_READY.value == "synthesis_ready"
        assert MessageType.AGENT_QUERY.value == "agent_query"
        assert MessageType.AGENT_DISCOVERY.value == "agent_discovery"

    def test_feedback_integration_types(self):
        assert MessageType.SYNTHESIS_FEEDBACK.value == "synthesis_feedback"
        assert MessageType.CONTRIBUTION_EVALUATION.value == "contribution_evaluation"
        assert MessageType.IMPROVEMENT_REQUEST.value == "improvement_request"

    def test_no_system_action_type(self):
        names = {mt.name for mt in MessageType}
        assert "SYSTEM_ACTION" not in names

    def test_all_values_are_strings(self):
        for mt in MessageType:
            assert isinstance(mt.value, str)

    def test_all_values_are_unique(self):
        values = [mt.value for mt in MessageType]
        assert len(values) == len(set(values))


# -------------------------------------------------------------------------
# Message dataclass construction
# -------------------------------------------------------------------------


class TestMessageConstruction:
    """Verify Message dataclass fields and defaults."""

    def test_required_fields(self):
        msg = Message(
            sender_id="agent-1",
            message_type=MessageType.TASK_REQUEST,
            content={"task": "do something"},
        )
        assert msg.sender_id == "agent-1"
        assert msg.message_type == MessageType.TASK_REQUEST
        assert msg.content == {"task": "do something"}

    def test_auto_generated_message_id(self):
        msg = Message(
            sender_id="a",
            message_type=MessageType.STATUS_UPDATE,
            content={},
        )
        # Should be a valid UUID string
        parsed = uuid.UUID(msg.message_id)
        assert str(parsed) == msg.message_id

    def test_unique_ids_across_messages(self):
        msg1 = Message(sender_id="a", message_type=MessageType.STATUS_UPDATE, content={})
        msg2 = Message(sender_id="a", message_type=MessageType.STATUS_UPDATE, content={})
        assert msg1.message_id != msg2.message_id

    def test_auto_timestamp(self):
        before = time.time()
        msg = Message(
            sender_id="a",
            message_type=MessageType.STATUS_UPDATE,
            content={},
        )
        after = time.time()
        assert before <= msg.timestamp <= after

    def test_explicit_timestamp(self):
        msg = Message(
            sender_id="a",
            message_type=MessageType.STATUS_UPDATE,
            content={},
            timestamp=1234567890.0,
        )
        assert msg.timestamp == 1234567890.0

    def test_receiver_id_default_is_none(self):
        msg = Message(
            sender_id="a",
            message_type=MessageType.STATUS_UPDATE,
            content={},
        )
        assert msg.receiver_id is None

    def test_explicit_receiver_id(self):
        msg = Message(
            sender_id="a",
            message_type=MessageType.STATUS_UPDATE,
            content={},
            receiver_id="agent-b",
        )
        assert msg.receiver_id == "agent-b"

    def test_explicit_message_id(self):
        msg = Message(
            sender_id="a",
            message_type=MessageType.STATUS_UPDATE,
            content={},
            message_id="custom-id-123",
        )
        assert msg.message_id == "custom-id-123"

    def test_content_dict_preserved(self):
        content = {"nested": {"key": [1, 2, 3]}, "flag": True}
        msg = Message(
            sender_id="a",
            message_type=MessageType.TASK_COMPLETE,
            content=content,
        )
        assert msg.content == content
        assert msg.content["nested"]["key"] == [1, 2, 3]


# -------------------------------------------------------------------------
# Message with sample_message fixture
# -------------------------------------------------------------------------


class TestSampleMessageFixture:
    """Tests using the sample_message fixture from conftest."""

    def test_fixture_sender_id(self, sample_message):
        assert sample_message.sender_id == "agent-test"

    def test_fixture_message_type(self, sample_message):
        assert sample_message.message_type == MessageType.STATUS_UPDATE

    def test_fixture_has_confidence(self, sample_message):
        assert "confidence" in sample_message.content
        assert sample_message.content["confidence"] == 0.75
