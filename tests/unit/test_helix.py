"""Unit tests for felix_agent_sdk.core.helix module."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from felix_agent_sdk.core.helix import (
    ANALYSIS_END,
    EXPLORATION_END,
    HelixConfig,
    HelixGeometry,
    HelixPosition,
)


# ---------------------------------------------------------------------------
# HelixGeometry
# ---------------------------------------------------------------------------


class TestHelixGeometry:
    """Tests for the ported HelixGeometry class."""

    def test_construction_valid(self) -> None:
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        assert h.top_radius == 3.0
        assert h.bottom_radius == 0.5
        assert h.height == 8.0
        assert h.turns == 2

    def test_construction_invalid_top_not_greater_than_bottom(self) -> None:
        with pytest.raises(ValueError, match="top_radius must be greater"):
            HelixGeometry(top_radius=0.5, bottom_radius=0.5, height=8.0, turns=2)

    def test_construction_invalid_height_zero(self) -> None:
        with pytest.raises(ValueError, match="height must be positive"):
            HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=0, turns=2)

    def test_construction_invalid_turns_zero(self) -> None:
        with pytest.raises(ValueError, match="turns must be positive"):
            HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=0)

    def test_get_position_invalid_t_below(self) -> None:
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        with pytest.raises(ValueError, match="t must be between 0 and 1"):
            h.get_position(-0.1)

    def test_get_position_invalid_t_above(self) -> None:
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        with pytest.raises(ValueError, match="t must be between 0 and 1"):
            h.get_position(1.1)

    def test_get_position_at_t0(self) -> None:
        """At t=0: z=height, angle=0, R=top_radius -> x=top_radius, y=0."""
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        x, y, z = h.get_position(0.0)
        assert math.isclose(z, 8.0, rel_tol=1e-9)
        assert math.isclose(x, 3.0, rel_tol=1e-9)
        assert math.isclose(y, 0.0, abs_tol=1e-9)

    def test_get_position_at_t1(self) -> None:
        """At t=1: z=0, angle=2*pi*turns, R=bottom_radius."""
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        x, y, z = h.get_position(1.0)
        assert math.isclose(z, 0.0, abs_tol=1e-9)
        # angle = 2*pi*2 = 4*pi -> cos(4*pi)=1, sin(4*pi)=0
        assert math.isclose(x, 0.5, rel_tol=1e-9)
        assert math.isclose(y, 0.0, abs_tol=1e-9)

    def test_get_position_midpoint(self) -> None:
        """At t=0.5: z=4.0, verify position is geometrically consistent."""
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        x, y, z = h.get_position(0.5)
        assert math.isclose(z, 4.0, rel_tol=1e-9)
        # Radius at z=4 should be between bottom and top
        r = math.sqrt(x * x + y * y)
        assert 0.5 < r < 3.0

    def test_get_radius_at_top(self) -> None:
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        assert math.isclose(h.get_radius(8.0), 3.0, rel_tol=1e-9)

    def test_get_radius_at_bottom(self) -> None:
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        assert math.isclose(h.get_radius(0.0), 0.5, rel_tol=1e-9)

    def test_get_radius_clamps_below_zero(self) -> None:
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        assert math.isclose(h.get_radius(-5.0), 0.5, rel_tol=1e-9)

    def test_get_radius_clamps_above_height(self) -> None:
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        assert math.isclose(h.get_radius(100.0), 3.0, rel_tol=1e-9)

    def test_get_angle_at_t0(self) -> None:
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        assert math.isclose(h.get_angle_at_t(0.0), 0.0, abs_tol=1e-9)

    def test_get_angle_at_t1(self) -> None:
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        expected = 2.0 * math.pi * 2  # 4*pi
        assert math.isclose(h.get_angle_at_t(1.0), expected, rel_tol=1e-9)

    def test_get_tangent_vector_is_unit_length(self) -> None:
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            tx, ty, tz = h.get_tangent_vector(t)
            length = math.sqrt(tx * tx + ty * ty + tz * tz)
            assert math.isclose(length, 1.0, abs_tol=1e-6)

    def test_get_tangent_vector_boundary_t0(self) -> None:
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        tx, ty, tz = h.get_tangent_vector(0.0)
        length = math.sqrt(tx * tx + ty * ty + tz * tz)
        assert math.isclose(length, 1.0, abs_tol=1e-6)

    def test_get_tangent_vector_boundary_t1(self) -> None:
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        tx, ty, tz = h.get_tangent_vector(1.0)
        length = math.sqrt(tx * tx + ty * ty + tz * tz)
        assert math.isclose(length, 1.0, abs_tol=1e-6)

    def test_approximate_arc_length_positive(self) -> None:
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        arc = h.approximate_arc_length()
        assert arc > 0.0

    def test_approximate_arc_length_zero_range(self) -> None:
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        arc = h.approximate_arc_length(t_start=0.5, t_end=0.5)
        assert math.isclose(arc, 0.0, abs_tol=1e-9)

    def test_approximate_arc_length_invalid_range(self) -> None:
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        with pytest.raises(ValueError, match="Invalid t_start or t_end"):
            h.approximate_arc_length(t_start=0.8, t_end=0.2)

    def test_repr(self) -> None:
        h = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        r = repr(h)
        assert "HelixGeometry" in r
        assert "3.0" in r
        assert "0.5" in r


# ---------------------------------------------------------------------------
# HelixConfig
# ---------------------------------------------------------------------------


class TestHelixConfig:
    """Tests for the HelixConfig dataclass and presets."""

    def test_default_preset_values(self) -> None:
        c = HelixConfig.default()
        assert c.top_radius == 3.0
        assert c.bottom_radius == 0.5
        assert c.height == 8.0
        assert c.turns == 2.0

    def test_research_heavy_preset_values(self) -> None:
        c = HelixConfig.research_heavy()
        assert c.top_radius == 5.0
        assert c.bottom_radius == 0.5
        assert c.height == 10.0
        assert c.turns == 3.0

    def test_fast_convergence_preset_values(self) -> None:
        c = HelixConfig.fast_convergence()
        assert c.top_radius == 2.0
        assert c.bottom_radius == 0.5
        assert c.height == 5.0
        assert c.turns == 1.0

    def test_to_geometry_returns_helix_geometry(self) -> None:
        c = HelixConfig.default()
        g = c.to_geometry()
        assert isinstance(g, HelixGeometry)
        assert g.top_radius == 3.0
        assert g.bottom_radius == 0.5
        assert g.height == 8.0
        assert g.turns == 2

    def test_to_geometry_default_matches_constants(self) -> None:
        """Default preset matches the canonical constants from CLAUDE.md."""
        g = HelixConfig.default().to_geometry()
        assert g.top_radius == 3.0
        assert g.bottom_radius == 0.5
        assert g.height == 8.0
        assert g.turns == 2

    def test_preset_immutability(self) -> None:
        c = HelixConfig.default()
        with pytest.raises(FrozenInstanceError):
            c.top_radius = 99.0  # type: ignore[misc]

    def test_repr(self) -> None:
        c = HelixConfig.default()
        r = repr(c)
        assert "HelixConfig" in r
        assert "3.0" in r


# ---------------------------------------------------------------------------
# HelixPosition
# ---------------------------------------------------------------------------


class TestHelixPosition:
    """Tests for the HelixPosition phase-aware wrapper."""

    def _default_geometry(self) -> HelixGeometry:
        return HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)

    def test_construction_valid(self) -> None:
        g = self._default_geometry()
        p = HelixPosition(g, t=0.5)
        assert p.t == 0.5
        assert p.geometry is g

    def test_construction_invalid_t(self) -> None:
        g = self._default_geometry()
        with pytest.raises(ValueError, match="t must be between 0 and 1"):
            HelixPosition(g, t=1.5)

    def test_coordinates_match_geometry(self) -> None:
        g = self._default_geometry()
        p = HelixPosition(g, t=0.3)
        expected = g.get_position(0.3)
        assert p.coordinates == expected

    def test_coordinates_are_cached(self) -> None:
        g = self._default_geometry()
        p = HelixPosition(g, t=0.5)
        first = p.coordinates
        second = p.coordinates
        assert first is second

    def test_xyz_properties_match_coords(self) -> None:
        g = self._default_geometry()
        p = HelixPosition(g, t=0.5)
        cx, cy, cz = p.coordinates
        assert p.x == cx
        assert p.y == cy
        assert p.z == cz

    def test_radius_property(self) -> None:
        g = self._default_geometry()
        p = HelixPosition(g, t=0.0)
        assert math.isclose(p.radius, 3.0, rel_tol=1e-9)

    def test_angle_property(self) -> None:
        g = self._default_geometry()
        p = HelixPosition(g, t=0.0)
        assert math.isclose(p.angle, 0.0, abs_tol=1e-9)

    def test_phase_at_t0_is_exploration(self) -> None:
        g = self._default_geometry()
        p = HelixPosition(g, t=0.0)
        assert p.phase == "exploration"

    def test_phase_at_exploration_boundary(self) -> None:
        """t=EXPLORATION_END exactly falls into analysis, not exploration."""
        g = self._default_geometry()
        p = HelixPosition(g, t=EXPLORATION_END)
        assert p.phase == "analysis"

    def test_phase_in_analysis_band(self) -> None:
        g = self._default_geometry()
        p = HelixPosition(g, t=0.55)
        assert p.phase == "analysis"

    def test_phase_at_analysis_boundary(self) -> None:
        """t=ANALYSIS_END exactly falls into synthesis."""
        g = self._default_geometry()
        p = HelixPosition(g, t=ANALYSIS_END)
        assert p.phase == "synthesis"

    def test_phase_at_t1_is_synthesis(self) -> None:
        g = self._default_geometry()
        p = HelixPosition(g, t=1.0)
        assert p.phase == "synthesis"

    def test_is_exploration_true_in_band(self) -> None:
        g = self._default_geometry()
        p = HelixPosition(g, t=0.2)
        assert p.is_exploration is True
        assert p.is_analysis is False
        assert p.is_synthesis is False

    def test_is_analysis_true_in_band(self) -> None:
        g = self._default_geometry()
        p = HelixPosition(g, t=0.5)
        assert p.is_analysis is True
        assert p.is_exploration is False
        assert p.is_synthesis is False

    def test_is_synthesis_true_in_band(self) -> None:
        g = self._default_geometry()
        p = HelixPosition(g, t=0.9)
        assert p.is_synthesis is True
        assert p.is_exploration is False
        assert p.is_analysis is False

    def test_temperature_hint_at_t0(self) -> None:
        g = self._default_geometry()
        p = HelixPosition(g, t=0.0)
        assert math.isclose(p.temperature_hint, 1.0, rel_tol=1e-9)

    def test_temperature_hint_at_t1(self) -> None:
        g = self._default_geometry()
        p = HelixPosition(g, t=1.0)
        assert math.isclose(p.temperature_hint, 0.0, abs_tol=1e-9)

    def test_temperature_hint_midpoint(self) -> None:
        g = self._default_geometry()
        p = HelixPosition(g, t=0.5)
        assert math.isclose(p.temperature_hint, 0.5, rel_tol=1e-9)

    def test_repr(self) -> None:
        g = self._default_geometry()
        p = HelixPosition(g, t=0.5)
        r = repr(p)
        assert "HelixPosition" in r
        assert "phase=" in r
        assert "coords=" in r
