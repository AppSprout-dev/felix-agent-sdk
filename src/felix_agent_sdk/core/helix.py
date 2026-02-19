"""Felix core helix geometry — parametric 3D spiral for agent positioning.

Ported from CalebisGross/felix src/core/helix_geometry.py.

Mathematical Foundation:
    Parametric helix with exponential radius tapering (wider at top, narrower at bottom).
    Position vector r(t) = (R(t) cos(theta(t)), R(t) sin(theta(t)), z(t))
    Parameter t in [0,1] where t=0 is top (wide), t=1 is bottom (narrow).
    Radius R(z) = R_bottom * (R_top / R_bottom)^(z / height)  -- exponential tapering
    Angular function theta(t) = 2*pi * turns * t
    Height z(t) = height * (1 - t)  -- descends from top to bottom
"""

from __future__ import annotations

import math
from typing import Tuple


class HelixGeometry:
    """Core helix mathematical model for agent positioning."""

    def __init__(
        self,
        top_radius: float,
        bottom_radius: float,
        height: float,
        turns: int,
    ) -> None:
        """Initialize helix with geometric parameters.

        Args:
            top_radius: Radius at the top of the helix (t=0, z=height).
            bottom_radius: Radius at the bottom of the helix (t=1, z=0).
            height: Total vertical height of the helix.
            turns: Number of complete rotations from top to bottom.

        Raises:
            ValueError: If parameters are invalid.
        """
        self._validate_parameters(top_radius, bottom_radius, height, turns)

        self.top_radius = top_radius
        self.bottom_radius = bottom_radius
        self.height = height
        self.turns = turns

    def _validate_parameters(
        self,
        top_radius: float,
        bottom_radius: float,
        height: float,
        turns: int,
    ) -> None:
        """Validate helix parameters for mathematical consistency."""
        if top_radius <= bottom_radius:
            raise ValueError("top_radius must be greater than bottom_radius for tapering")

        if height <= 0:
            raise ValueError("height must be positive")

        if turns <= 0:
            raise ValueError("turns must be positive")

    def get_position(self, t: float) -> Tuple[float, float, float]:
        """Calculate 3D position along helix path.

        Parametric equations:
            z(t) = height * (1 - t)
            R(z) = bottom_radius * (top_radius / bottom_radius)^(z / height)
            theta(t) = 2*pi * turns * t
            x(t) = R(z) * cos(theta(t))
            y(t) = R(z) * sin(theta(t))

        Args:
            t: Parameter value between 0 (top/wide) and 1 (bottom/narrow).

        Returns:
            Tuple of (x, y, z) coordinates.

        Raises:
            ValueError: If t is outside [0,1] range.
        """
        if not (0.0 <= t <= 1.0):
            raise ValueError("t must be between 0 and 1")

        # Height position (top at t=0, bottom at t=1)
        z = self.height * (1.0 - t)

        # Radius at current height (exponential tapering, wider at top)
        radius = self.get_radius(z)

        # Angular position
        angle_radians = t * self.turns * 2.0 * math.pi

        x = radius * math.cos(angle_radians)
        y = radius * math.sin(angle_radians)

        return (x, y, z)

    def get_radius(self, z: float) -> float:
        """Calculate radius at given height using exponential tapering.

        R(z) = bottom_radius * (top_radius / bottom_radius)^(z / height)

        At z = height: R = top_radius (wide).
        At z = 0:      R = bottom_radius (narrow).

        Args:
            z: Height value (0 = bottom, height = top).

        Returns:
            Radius at the specified height.
        """
        z = max(0.0, min(z, self.height))

        radius_ratio = self.top_radius / self.bottom_radius
        height_fraction = z / self.height
        radius: float = self.bottom_radius * math.pow(radius_ratio, height_fraction)

        return radius

    def get_angle_at_t(self, t: float) -> float:
        """Calculate rotation angle (in radians) at parameter t.

        Args:
            t: Parameter value between 0 and 1.

        Returns:
            Angle in radians.
        """
        if not (0.0 <= t <= 1.0):
            raise ValueError("t must be between 0 and 1")

        return t * self.turns * 2.0 * math.pi

    def get_tangent_vector(self, t: float) -> Tuple[float, float, float]:
        """Calculate approximate normalized tangent vector at parameter t.

        Useful for agent orientation and movement direction.
        """
        if not (0.0 <= t <= 1.0):
            raise ValueError("t must be between 0 and 1")

        eps = 1e-8
        t1 = max(0.0, t - eps)
        t2 = min(1.0, t + eps)

        x1, y1, z1 = self.get_position(t1)
        x2, y2, z2 = self.get_position(t2)

        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1

        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length > 0:
            return (dx / length, dy / length, dz / length)
        return (0.0, 0.0, 0.0)

    def approximate_arc_length(
        self,
        t_start: float = 0.0,
        t_end: float = 1.0,
        segments: int = 1000,
    ) -> float:
        """Approximate arc length of helix segment using linear interpolation."""
        if not (0.0 <= t_start <= t_end <= 1.0):
            raise ValueError("Invalid t_start or t_end values")

        if segments < 1:
            raise ValueError("segments must be positive")

        total_length = 0.0
        dt = (t_end - t_start) / segments

        prev_x, prev_y, prev_z = self.get_position(t_start)

        for i in range(1, segments + 1):
            t = t_start + i * dt
            x, y, z = self.get_position(t)

            distance = math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2 + (z - prev_z) ** 2)
            total_length += distance

            prev_x, prev_y, prev_z = x, y, z

        return total_length

    def __repr__(self) -> str:
        return (
            f"HelixGeometry(top_radius={self.top_radius}, "
            f"bottom_radius={self.bottom_radius}, "
            f"height={self.height}, turns={self.turns})"
        )
