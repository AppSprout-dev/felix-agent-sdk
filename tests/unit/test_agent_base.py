"""Tests for felix_agent_sdk.agents.base — Agent and AgentState."""

from __future__ import annotations

import pytest

from felix_agent_sdk.agents.base import Agent, AgentState, generate_spawn_times
from felix_agent_sdk.core.helix import HelixGeometry


@pytest.fixture
def helix():
    return HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)


@pytest.fixture
def agent(helix):
    return Agent(agent_id="agent-001", helix=helix, spawn_time=0.2, velocity=1.0)


# -------------------------------------------------------------------------
# AgentState
# -------------------------------------------------------------------------


class TestAgentState:
    def test_all_states_exist(self):
        states = {s.value for s in AgentState}
        assert states == {"waiting", "spawning", "active", "completed", "failed"}

    def test_values(self):
        assert AgentState.WAITING.value == "waiting"
        assert AgentState.ACTIVE.value == "active"
        assert AgentState.COMPLETED.value == "completed"


# -------------------------------------------------------------------------
# Agent construction
# -------------------------------------------------------------------------


class TestAgentConstruction:
    def test_default_state_is_waiting(self, agent):
        assert agent.state == AgentState.WAITING

    def test_initial_progress_is_zero(self, agent):
        assert agent.progress == 0.0

    def test_agent_id_stored(self, agent):
        assert agent.agent_id == "agent-001"

    def test_spawn_time_stored(self, agent):
        assert agent.spawn_time == 0.2

    def test_custom_velocity(self, agent):
        assert agent._velocity == 1.0

    def test_random_velocity_when_none(self, helix):
        a = Agent("a", helix, velocity=None)
        assert 0.7 <= a._velocity <= 1.3

    def test_empty_id_raises(self, helix):
        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            Agent("", helix)

    def test_spawn_time_out_of_range_raises(self, helix):
        with pytest.raises(ValueError, match="spawn_time must be between 0 and 1"):
            Agent("a", helix, spawn_time=1.5)


# -------------------------------------------------------------------------
# Lifecycle
# -------------------------------------------------------------------------


class TestAgentLifecycle:
    def test_can_spawn_before_spawn_time(self, agent):
        assert agent.can_spawn(0.1) is False

    def test_can_spawn_at_spawn_time(self, agent):
        assert agent.can_spawn(0.2) is True

    def test_can_spawn_after_spawn_time(self, agent):
        assert agent.can_spawn(0.5) is True

    def test_spawn_transitions_to_active(self, agent):
        agent.spawn(0.3, task="some_task")
        assert agent.state == AgentState.ACTIVE

    def test_spawn_sets_task(self, agent):
        agent.spawn(0.3, task="my_task")
        assert agent.current_task == "my_task"

    def test_spawn_before_time_raises(self, agent):
        with pytest.raises(ValueError, match="Cannot spawn agent before"):
            agent.spawn(0.1)

    def test_double_spawn_raises(self, agent):
        agent.spawn(0.3)
        with pytest.raises(ValueError, match="Agent already spawned"):
            agent.spawn(0.4)


# -------------------------------------------------------------------------
# Position
# -------------------------------------------------------------------------


class TestAgentPosition:
    def test_position_returns_helix_position(self, agent):
        pos = agent.position
        assert pos.phase == "exploration"  # t=0.0

    def test_position_phase_at_zero_is_exploration(self, agent):
        assert agent.position.is_exploration

    def test_update_position_advances_t(self, agent):
        agent.spawn(0.2)
        agent.update_position(0.5)
        assert agent.progress > 0.0

    def test_update_position_clamps_at_1(self, agent):
        agent.spawn(0.3)
        agent.update_position(100.0)
        assert agent.progress == 1.0

    def test_update_position_respects_velocity(self, helix):
        slow = Agent("slow", helix, spawn_time=0.0, velocity=0.5)
        fast = Agent("fast", helix, spawn_time=0.0, velocity=2.0)
        slow.spawn(0.0)
        fast.spawn(0.0)
        slow.update_position(0.3)
        fast.update_position(0.3)
        assert fast.progress > slow.progress

    def test_completion_on_reaching_end(self, agent):
        agent.spawn(0.3)
        agent.update_position(100.0)
        assert agent.state == AgentState.COMPLETED

    def test_get_position_info_has_required_keys(self, agent):
        info = agent.get_position_info()
        required = {"x", "y", "z", "radius", "depth_ratio", "progress", "phase"}
        assert required.issubset(info.keys())

    def test_get_position_returns_none_when_waiting(self, agent):
        assert agent.get_position(0.1) is None

    def test_get_position_returns_tuple_when_active(self, agent):
        agent.spawn(0.3)
        pos = agent.get_position(0.5)
        assert pos is not None
        assert len(pos) == 3

    def test_update_waiting_agent_raises(self, agent):
        with pytest.raises(ValueError, match="Cannot update position of unspawned"):
            agent.update_position(0.5)


# -------------------------------------------------------------------------
# Confidence
# -------------------------------------------------------------------------


class TestAgentConfidence:
    def test_record_confidence_appends(self, agent):
        agent.record_confidence(0.5)
        assert len(agent._confidence_history) == 1

    def test_record_confidence_caps_at_10(self, agent):
        for i in range(15):
            agent.record_confidence(float(i) / 15)
        assert len(agent._confidence_history) == 10

    def test_record_confidence_updates_current(self, agent):
        agent.record_confidence(0.75)
        assert agent.confidence == 0.75

    def test_high_confidence_increases_acceleration(self, agent):
        agent.spawn(0.3)
        # Establish high-confidence trend
        for _ in range(4):
            agent.record_confidence(0.9)
        old_acc = agent._acceleration
        agent.update_position(0.5)
        # With consistently high confidence, acceleration should increase or stay
        assert agent._acceleration >= old_acc

    def test_low_confidence_decreases_acceleration(self, agent):
        agent.spawn(0.3)
        # Start with good, then drop
        agent.record_confidence(0.8)
        agent.record_confidence(0.8)
        agent.record_confidence(0.3)
        agent.record_confidence(0.2)
        old_acc = agent._acceleration
        agent.update_position(0.4)
        assert agent._acceleration <= old_acc


# -------------------------------------------------------------------------
# Velocity control
# -------------------------------------------------------------------------


class TestVelocityControl:
    def test_velocity_property(self, agent):
        assert agent.velocity == agent._velocity * agent._acceleration

    def test_set_velocity_multiplier_clamps_high(self, agent):
        agent.set_velocity_multiplier(10.0)
        assert agent._velocity == 3.0

    def test_set_velocity_multiplier_clamps_low(self, agent):
        agent.set_velocity_multiplier(0.0)
        assert agent._velocity == 0.1

    def test_pause(self, agent):
        agent.pause_for_duration(1.0, 0.5)
        assert agent.is_paused

    def test_paused_agent_does_not_advance(self, agent):
        agent.spawn(0.2)
        agent.pause_for_duration(10.0, 0.2)
        agent.update_position(0.5)
        assert agent.progress == 0.0


# -------------------------------------------------------------------------
# generate_spawn_times
# -------------------------------------------------------------------------


class TestGenerateSpawnTimes:
    def test_count(self):
        times = generate_spawn_times(5)
        assert len(times) == 5

    def test_range(self):
        times = generate_spawn_times(100)
        assert all(0.0 <= t <= 1.0 for t in times)

    def test_deterministic_with_seed(self):
        a = generate_spawn_times(10, seed=42)
        b = generate_spawn_times(10, seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        a = generate_spawn_times(10, seed=1)
        b = generate_spawn_times(10, seed=2)
        assert a != b


# -------------------------------------------------------------------------
# Repr
# -------------------------------------------------------------------------


class TestAgentRepr:
    def test_repr_contains_id(self, agent):
        assert "agent-001" in repr(agent)

    def test_repr_contains_state(self, agent):
        assert "waiting" in repr(agent)
