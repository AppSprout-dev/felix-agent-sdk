"""Tests for the helix visualizer module."""

from __future__ import annotations

import pytest

from felix_agent_sdk.core.helix import HelixConfig
from felix_agent_sdk.visualization import AgentDisplayState, HelixVisualizer


@pytest.fixture
def helix():
    return HelixConfig.default().to_geometry()


@pytest.fixture
def viz(helix):
    return HelixVisualizer(helix)


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------


class TestRegistration:
    def test_register_agent(self, viz):
        viz.register_agent("rm", label="RM", color="green")
        assert "rm" in viz._agents
        assert viz._agents["rm"].label == "RM"

    def test_register_multiple(self, viz):
        viz.register_agent("a1", "A1")
        viz.register_agent("a2", "A2")
        assert len(viz._agents) == 2

    def test_register_default_color(self, viz):
        viz.register_agent("x", "XX")
        # Default colour is cyan
        assert viz._agents["x"].color == "\033[96m"


# ------------------------------------------------------------------
# Update
# ------------------------------------------------------------------


class TestUpdate:
    def test_update_agent(self, viz):
        viz.register_agent("rm", "RM")
        viz.update("rm", progress=0.5, confidence=0.8, phase="analysis")
        assert viz._agents["rm"].progress == 0.5
        assert viz._agents["rm"].confidence == 0.8
        assert viz._agents["rm"].phase == "analysis"

    def test_update_unknown_raises(self, viz):
        with pytest.raises(KeyError):
            viz.update("nonexistent", progress=0.5, confidence=0.5)

    def test_update_auto_phase_exploration(self, viz):
        viz.register_agent("rm", "RM")
        viz.update("rm", progress=0.1, confidence=0.5)
        assert viz._agents["rm"].phase == "exploration"

    def test_update_auto_phase_analysis(self, viz):
        viz.register_agent("rm", "RM")
        viz.update("rm", progress=0.5, confidence=0.5)
        assert viz._agents["rm"].phase == "analysis"

    def test_update_auto_phase_synthesis(self, viz):
        viz.register_agent("rm", "RM")
        viz.update("rm", progress=0.9, confidence=0.5)
        assert viz._agents["rm"].phase == "synthesis"

    def test_update_status(self, viz):
        viz.register_agent("rm", "RM")
        viz.update("rm", progress=0.3, confidence=0.7, status="Searching...")
        assert viz._agents["rm"].status == "Searching..."


# ------------------------------------------------------------------
# Render
# ------------------------------------------------------------------


class TestRender:
    def test_render_to_string_returns_str(self, viz):
        viz.register_agent("rm", "RM", "green")
        viz.update("rm", 0.3, 0.7, "exploration")
        output = viz.render_to_string(tick=1, day=1)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_render_contains_agent_labels(self, viz):
        viz.register_agent("rm", "RM", "green")
        viz.update("rm", 0.3, 0.7)
        output = viz.render_to_string()
        assert "RM" in output

    def test_render_contains_phase_names(self, viz):
        viz.register_agent("rm", "RM")
        viz.update("rm", 0.5, 0.5)
        output = viz.render_to_string()
        assert "EXPLORATION" in output or "ANALYSIS" in output or "SYNTHESIS" in output

    def test_render_header_info(self, viz):
        viz.register_agent("rm", "RM")
        viz.update("rm", 0.5, 0.5)
        output = viz.render_to_string(tick=47, day=12, extra_info={"score": 0.763})
        assert "47" in output
        assert "12" in output

    def test_render_extra_info(self, viz):
        viz.register_agent("rm", "RM")
        viz.update("rm", 0.5, 0.5)
        output = viz.render_to_string(extra_info={"score": 0.763})
        assert "score" in output

    def test_render_team_confidence(self, viz):
        viz.register_agent("a1", "A1")
        viz.register_agent("a2", "A2")
        viz.update("a1", 0.3, 0.8)
        viz.update("a2", 0.5, 0.6)
        output = viz.render_to_string()
        assert "Confidence" in output or "confidence" in output.lower()

    def test_render_empty_agents(self, viz):
        """Render with no agents should not raise."""
        output = viz.render_to_string()
        assert isinstance(output, str)


# ------------------------------------------------------------------
# NO_COLOR
# ------------------------------------------------------------------


class TestNoColor:
    def test_no_color_env(self, helix, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        from felix_agent_sdk.visualization import terminal

        assert not terminal.supports_color()

    def test_force_color_env(self, helix, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("FORCE_COLOR", "1")
        from felix_agent_sdk.visualization import terminal

        assert terminal.supports_color()


# ------------------------------------------------------------------
# Live context manager
# ------------------------------------------------------------------


class TestLiveContext:
    def test_live_context_manager(self, viz):
        with viz.live() as v:
            assert v is viz


# ------------------------------------------------------------------
# Dimensions
# ------------------------------------------------------------------


class TestDimensions:
    def test_default_dimensions(self, helix):
        viz = HelixVisualizer(helix)
        assert viz._width == 50
        assert viz._height == 28
        assert viz._sidebar_width == 40

    def test_custom_dimensions(self, helix):
        viz = HelixVisualizer(helix, width=60, height=20, sidebar_width=30)
        assert viz._width == 60
        assert viz._height == 20
        assert viz._sidebar_width == 30

    def test_custom_title(self, helix):
        viz = HelixVisualizer(helix, title="My Helix")
        assert viz._title == "My Helix"


# ------------------------------------------------------------------
# AgentDisplayState dataclass
# ------------------------------------------------------------------


class TestAgentDisplayState:
    def test_defaults(self):
        state = AgentDisplayState(agent_id="test", label="TT", color="\033[96m")
        assert state.progress == 0.0
        assert state.confidence == 0.0
        assert state.phase == "exploration"
        assert state.status == ""
