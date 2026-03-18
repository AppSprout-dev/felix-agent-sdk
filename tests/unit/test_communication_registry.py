"""Tests for felix_agent_sdk.communication.registry — AgentRegistry."""

from __future__ import annotations

import pytest

from felix_agent_sdk.communication.registry import AgentRegistry
from felix_agent_sdk.core.helix import ANALYSIS_END, EXPLORATION_END


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def registry():
    """A fresh AgentRegistry for each test."""
    return AgentRegistry()


# -------------------------------------------------------------------------
# Registration / Deregistration
# -------------------------------------------------------------------------


class TestRegistration:
    def test_register_agent_basic(self, registry):
        registry.register_agent("agent-1")
        assert "agent-1" in registry.get_active_agents()

    def test_register_agent_with_metadata(self, registry):
        registry.register_agent("agent-1", metadata={"agent_type": "research"})
        info = registry.get_agent_info("agent-1")
        assert info is not None
        assert info["agent_type"] == "research"

    def test_register_multiple_agents(self, registry):
        for i in range(5):
            registry.register_agent(f"agent-{i}")
        assert len(registry.get_active_agents()) == 5

    def test_deregister_agent(self, registry):
        registry.register_agent("agent-1")
        registry.deregister_agent("agent-1")
        assert "agent-1" not in registry.get_active_agents()

    def test_deregister_unknown_agent_no_error(self, registry):
        # Should not raise
        registry.deregister_agent("nonexistent")

    def test_get_agent_info_returns_none_for_unknown(self, registry):
        assert registry.get_agent_info("nonexistent") is None

    def test_get_agent_info_has_position_and_performance(self, registry):
        registry.register_agent("agent-1")
        info = registry.get_agent_info("agent-1")
        assert "position" in info
        assert "performance" in info

    def test_initial_position_is_exploration(self, registry):
        registry.register_agent("agent-1")
        info = registry.get_agent_info("agent-1")
        assert info["position"]["phase"] == "exploration"
        assert info["position"]["depth_ratio"] == 0.0

    def test_initial_performance_is_zero(self, registry):
        registry.register_agent("agent-1")
        info = registry.get_agent_info("agent-1")
        perf = info["performance"]
        assert perf["confidence"] == 0.0
        assert perf["tasks_completed"] == 0
        assert perf["errors"] == 0


# -------------------------------------------------------------------------
# Position tracking
# -------------------------------------------------------------------------


class TestPositionTracking:
    def test_update_agent_position(self, registry):
        registry.register_agent("agent-1")
        registry.update_agent_position("agent-1", {"depth_ratio": 0.5})
        info = registry.get_agent_info("agent-1")
        assert info["position"]["depth_ratio"] == 0.5

    def test_update_position_unknown_agent_ignored(self, registry):
        # Should not raise, just log a warning
        registry.update_agent_position("nonexistent", {"depth_ratio": 0.3})

    def test_position_includes_coordinates(self, registry):
        registry.register_agent("agent-1")
        registry.update_agent_position(
            "agent-1", {"depth_ratio": 0.2, "x": 1.0, "y": 2.0, "z": 3.0}
        )
        info = registry.get_agent_info("agent-1")
        pos = info["position"]
        assert pos["x"] == 1.0
        assert pos["y"] == 2.0
        assert pos["z"] == 3.0


# -------------------------------------------------------------------------
# Phase boundaries: EXPLORATION_END=0.4, ANALYSIS_END=0.7
# -------------------------------------------------------------------------


class TestPhaseBoundaries:
    """Verify that phase boundaries use 0.4 / 0.7, NOT 0.3 / 0.7."""

    def test_constants_are_correct(self):
        assert EXPLORATION_END == 0.4
        assert ANALYSIS_END == 0.7

    def test_exploration_phase_at_zero(self, registry):
        registry.register_agent("agent-1")
        registry.update_agent_position("agent-1", {"depth_ratio": 0.0})
        info = registry.get_agent_info("agent-1")
        assert info["position"]["phase"] == "exploration"

    def test_exploration_phase_just_below_boundary(self, registry):
        registry.register_agent("agent-1")
        registry.update_agent_position("agent-1", {"depth_ratio": 0.39})
        info = registry.get_agent_info("agent-1")
        assert info["position"]["phase"] == "exploration"

    def test_analysis_phase_at_boundary(self, registry):
        registry.register_agent("agent-1")
        registry.update_agent_position("agent-1", {"depth_ratio": 0.4})
        info = registry.get_agent_info("agent-1")
        assert info["position"]["phase"] == "analysis"

    def test_analysis_phase_just_below_synthesis(self, registry):
        registry.register_agent("agent-1")
        registry.update_agent_position("agent-1", {"depth_ratio": 0.69})
        info = registry.get_agent_info("agent-1")
        assert info["position"]["phase"] == "analysis"

    def test_synthesis_phase_at_boundary(self, registry):
        registry.register_agent("agent-1")
        registry.update_agent_position("agent-1", {"depth_ratio": 0.7})
        info = registry.get_agent_info("agent-1")
        assert info["position"]["phase"] == "synthesis"

    def test_synthesis_phase_at_one(self, registry):
        registry.register_agent("agent-1")
        registry.update_agent_position("agent-1", {"depth_ratio": 1.0})
        info = registry.get_agent_info("agent-1")
        assert info["position"]["phase"] == "synthesis"

    def test_get_agents_in_exploration_phase(self, registry):
        registry.register_agent("agent-1")
        registry.register_agent("agent-2")
        registry.update_agent_position("agent-1", {"depth_ratio": 0.1})
        registry.update_agent_position("agent-2", {"depth_ratio": 0.5})
        in_exploration = registry.get_agents_in_phase("exploration")
        assert "agent-1" in in_exploration
        assert "agent-2" not in in_exploration

    def test_get_agents_in_analysis_phase(self, registry):
        registry.register_agent("agent-1")
        registry.register_agent("agent-2")
        registry.update_agent_position("agent-1", {"depth_ratio": 0.5})
        registry.update_agent_position("agent-2", {"depth_ratio": 0.8})
        in_analysis = registry.get_agents_in_phase("analysis")
        assert "agent-1" in in_analysis
        assert "agent-2" not in in_analysis

    def test_get_agents_in_synthesis_phase(self, registry):
        registry.register_agent("agent-1")
        registry.register_agent("agent-2")
        registry.update_agent_position("agent-1", {"depth_ratio": 0.8})
        registry.update_agent_position("agent-2", {"depth_ratio": 0.3})
        in_synthesis = registry.get_agents_in_phase("synthesis")
        assert "agent-1" in in_synthesis
        assert "agent-2" not in in_synthesis


# -------------------------------------------------------------------------
# Confidence tracking
# -------------------------------------------------------------------------


class TestConfidenceTracking:
    def test_update_agent_performance_confidence(self, registry):
        registry.register_agent("agent-1")
        registry.update_agent_performance("agent-1", {"confidence": 0.85})
        info = registry.get_agent_info("agent-1")
        assert info["performance"]["confidence"] == 0.85

    def test_confidence_history_tracked(self, registry):
        registry.register_agent("agent-1")
        for val in [0.5, 0.6, 0.7]:
            registry.update_agent_performance("agent-1", {"confidence": val})
        info = registry.get_agent_info("agent-1")
        assert info["performance"]["confidence_history"] == [0.5, 0.6, 0.7]

    def test_confidence_history_capped_at_50(self, registry):
        registry.register_agent("agent-1")
        for i in range(60):
            registry.update_agent_performance("agent-1", {"confidence": i / 60.0})
        info = registry.get_agent_info("agent-1")
        assert len(info["performance"]["confidence_history"]) == 50

    def test_global_confidence_history_capped_at_100(self, registry):
        registry.register_agent("agent-1")
        for i in range(120):
            registry.update_agent_performance("agent-1", {"confidence": i / 120.0})
        assert len(registry._confidence_history) == 100

    def test_update_tasks_completed(self, registry):
        registry.register_agent("agent-1")
        registry.update_agent_performance("agent-1", {"tasks_completed": 5})
        info = registry.get_agent_info("agent-1")
        assert info["performance"]["tasks_completed"] == 5

    def test_update_errors(self, registry):
        registry.register_agent("agent-1")
        registry.update_agent_performance("agent-1", {"errors": 3})
        info = registry.get_agent_info("agent-1")
        assert info["performance"]["errors"] == 3

    def test_update_unknown_agent_ignored(self, registry):
        # Should not raise
        registry.update_agent_performance("nonexistent", {"confidence": 0.9})


# -------------------------------------------------------------------------
# Convergence status
# -------------------------------------------------------------------------


class TestConvergenceStatus:
    def test_empty_registry_status(self, registry):
        status = registry.get_convergence_status()
        assert status["active_agent_count"] == 0
        assert status["confidence_trend"] == "stable"
        assert status["synthesis_ready"] is False
        assert status["collaboration_density"] == 0.0

    def test_phase_distribution(self, registry):
        registry.register_agent("a1")
        registry.register_agent("a2")
        registry.register_agent("a3")
        registry.update_agent_position("a1", {"depth_ratio": 0.1})  # exploration
        registry.update_agent_position("a2", {"depth_ratio": 0.5})  # analysis
        registry.update_agent_position("a3", {"depth_ratio": 0.8})  # synthesis
        status = registry.get_convergence_status()
        assert status["phase_distribution"]["exploration"] == 1
        assert status["phase_distribution"]["analysis"] == 1
        assert status["phase_distribution"]["synthesis"] == 1

    def test_synthesis_ready_when_majority_in_synthesis_high_confidence(self, registry):
        # Need majority in synthesis with avg confidence >= 0.7
        for i in range(3):
            aid = f"agent-{i}"
            registry.register_agent(aid)
            registry.update_agent_position(aid, {"depth_ratio": 0.9})
            registry.update_agent_performance(aid, {"confidence": 0.85})
        status = registry.get_convergence_status()
        assert status["synthesis_ready"] is True

    def test_synthesis_not_ready_when_low_confidence(self, registry):
        for i in range(3):
            aid = f"agent-{i}"
            registry.register_agent(aid)
            registry.update_agent_position(aid, {"depth_ratio": 0.9})
            registry.update_agent_performance(aid, {"confidence": 0.3})
        status = registry.get_convergence_status()
        assert status["synthesis_ready"] is False

    def test_synthesis_not_ready_when_minority_in_synthesis(self, registry):
        # 1 in synthesis, 2 in exploration
        registry.register_agent("a1")
        registry.register_agent("a2")
        registry.register_agent("a3")
        registry.update_agent_position("a1", {"depth_ratio": 0.9})
        registry.update_agent_performance("a1", {"confidence": 0.9})
        registry.update_agent_position("a2", {"depth_ratio": 0.1})
        registry.update_agent_position("a3", {"depth_ratio": 0.1})
        status = registry.get_convergence_status()
        assert status["synthesis_ready"] is False

    def test_confidence_trend_rising(self, registry):
        registry.register_agent("a1")
        # Feed rising confidence values
        for i in range(10):
            registry.update_agent_performance("a1", {"confidence": 0.3 + (i * 0.05)})
        status = registry.get_convergence_status()
        assert status["confidence_trend"] == "rising"

    def test_confidence_trend_falling(self, registry):
        registry.register_agent("a1")
        # Feed falling confidence values
        for i in range(10):
            registry.update_agent_performance("a1", {"confidence": 0.9 - (i * 0.05)})
        status = registry.get_convergence_status()
        assert status["confidence_trend"] == "falling"

    def test_confidence_trend_stable(self, registry):
        registry.register_agent("a1")
        # Feed constant values
        for _ in range(10):
            registry.update_agent_performance("a1", {"confidence": 0.5})
        status = registry.get_convergence_status()
        assert status["confidence_trend"] == "stable"


# -------------------------------------------------------------------------
# Nearby agents
# -------------------------------------------------------------------------


class TestNearbyAgents:
    def test_nearby_agents_within_threshold(self, registry):
        registry.register_agent("a1")
        registry.register_agent("a2")
        registry.update_agent_position("a1", {"depth_ratio": 0.5})
        registry.update_agent_position("a2", {"depth_ratio": 0.55})
        nearby = registry.get_nearby_agents("a1", radius_threshold=0.1)
        assert "a2" in nearby

    def test_nearby_agents_excludes_self(self, registry):
        registry.register_agent("a1")
        nearby = registry.get_nearby_agents("a1", radius_threshold=1.0)
        assert "a1" not in nearby

    def test_nearby_agents_excludes_far_agents(self, registry):
        registry.register_agent("a1")
        registry.register_agent("a2")
        registry.update_agent_position("a1", {"depth_ratio": 0.1})
        registry.update_agent_position("a2", {"depth_ratio": 0.9})
        nearby = registry.get_nearby_agents("a1", radius_threshold=0.1)
        assert "a2" not in nearby

    def test_nearby_agents_unknown_agent_returns_empty(self, registry):
        nearby = registry.get_nearby_agents("nonexistent")
        assert nearby == []

    def test_nearby_agents_default_threshold(self, registry):
        registry.register_agent("a1")
        registry.register_agent("a2")
        registry.update_agent_position("a1", {"depth_ratio": 0.5})
        registry.update_agent_position("a2", {"depth_ratio": 0.59})
        # Default threshold is 0.1
        nearby = registry.get_nearby_agents("a1")
        assert "a2" in nearby


# -------------------------------------------------------------------------
# Collaboration density
# -------------------------------------------------------------------------


class TestCollaborationDensity:
    def test_no_collaboration_density_zero(self, registry):
        registry.register_agent("a1")
        registry.register_agent("a2")
        status = registry.get_convergence_status()
        assert status["collaboration_density"] == 0.0

    def test_collaboration_density_increases(self, registry):
        registry.register_agent("a1")
        registry.register_agent("a2")
        registry.record_collaboration("a1", "a2")
        status = registry.get_convergence_status()
        assert status["collaboration_density"] > 0.0

    def test_collaboration_density_capped_at_one(self, registry):
        registry.register_agent("a1")
        registry.register_agent("a2")
        # Record many collaborations to exceed max_pairs
        for _ in range(50):
            registry.record_collaboration("a1", "a2")
        status = registry.get_convergence_status()
        assert status["collaboration_density"] <= 1.0

    def test_single_agent_density_zero(self, registry):
        registry.register_agent("a1")
        registry.record_collaboration("a1", "a2")
        status = registry.get_convergence_status()
        assert status["collaboration_density"] == 0.0
