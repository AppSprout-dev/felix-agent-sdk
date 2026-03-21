"""Reusable terminal helix visualizer — renders agents spiralling from exploration to synthesis.

Draws a side-view cross-section of the helix in the terminal, showing each
agent's position, phase, confidence, and status in real time.  Adapted from
the demo visualizer in ``examples/05_deep_research_live/helix_visualizer.py``
and generalised for library-wide reuse.

No external dependencies beyond the Felix SDK and the Python stdlib.
"""

from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from felix_agent_sdk.core.helix import ANALYSIS_END, EXPLORATION_END, HelixGeometry
from felix_agent_sdk.visualization.terminal import (
    BOLD,
    COLOR_MAP,
    DIM,
    PHASE_COLORS,
    RESET,
    clear_screen,
    colorize,
    hide_cursor,
    progress_bar,
    show_cursor,
)

# Phase boundary glyphs
_PHASE_ICONS: Dict[str, str] = {
    "exploration": "~",
    "analysis": "=",
    "synthesis": "#",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class AgentDisplayState:
    """Display state for a registered agent.

    Attributes:
        agent_id: Unique agent identifier.
        label: Two-character short label rendered on the helix (e.g. ``"RM"``).
        color: ANSI escape code for this agent's colour.
        progress: Parametric position ``t`` in ``[0, 1]``.
        confidence: Agent confidence in ``[0, 1]``.
        phase: Current workflow phase name.
        status: Short free-text status string.
    """

    agent_id: str
    label: str
    color: str
    progress: float = 0.0
    confidence: float = 0.0
    phase: str = "exploration"
    status: str = ""


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------


class HelixVisualizer:
    """Renders an ASCII helix with live agent positions in the terminal.

    Usage::

        from felix_agent_sdk.core.helix import HelixConfig
        from felix_agent_sdk.visualization import HelixVisualizer

        helix = HelixConfig.default().to_geometry()
        viz = HelixVisualizer(helix)
        viz.register_agent("rm", label="RM", color="green")
        viz.update("rm", progress=0.3, confidence=0.7, phase="exploration")
        with viz.live() as v:
            v.render(tick=1, day=1)

    Args:
        helix: A ``HelixGeometry`` that defines the spiral shape.
        title: Header title shown above the helix.
        width: Character width of the helix canvas.
        height: Character height of the helix canvas.
        sidebar_width: Character width of the sidebar agent panel.
    """

    def __init__(
        self,
        helix: HelixGeometry,
        *,
        title: str = "F E L I X",
        width: int = 50,
        height: int = 28,
        sidebar_width: int = 40,
    ) -> None:
        self._helix = helix
        self._title = title
        self._width = width
        self._height = height
        self._sidebar_width = sidebar_width
        self._agents: Dict[str, AgentDisplayState] = {}
        self._frame: int = 0

    # ------------------------------------------------------------------
    # Registration & update
    # ------------------------------------------------------------------

    def register_agent(
        self,
        agent_id: str,
        label: str,
        color: str = "cyan",
    ) -> None:
        """Register an agent for display.

        Args:
            agent_id: Unique identifier for this agent.
            label: Two-character label rendered on the helix (e.g. ``"RM"``).
            color: Colour name — one of ``'cyan'``, ``'yellow'``, ``'green'``,
                ``'red'``, ``'magenta'``, ``'white'``.
        """
        ansi = COLOR_MAP.get(color, COLOR_MAP["cyan"])
        self._agents[agent_id] = AgentDisplayState(
            agent_id=agent_id,
            label=label,
            color=ansi,
        )

    def update(
        self,
        agent_id: str,
        progress: float,
        confidence: float,
        phase: str = "",
        status: str = "",
    ) -> None:
        """Update an agent's display state.

        Args:
            agent_id: Must match a previously registered agent.
            progress: Parametric position ``t`` in ``[0, 1]``.
            confidence: Agent confidence in ``[0, 1]``.
            phase: Phase name (``"exploration"``, ``"analysis"``,
                ``"synthesis"``). Auto-derived from *progress* if empty.
            status: Optional short status string for the sidebar.

        Raises:
            KeyError: If *agent_id* was not registered.
        """
        if agent_id not in self._agents:
            raise KeyError(f"Agent {agent_id!r} is not registered")
        agent = self._agents[agent_id]
        agent.progress = progress
        agent.confidence = confidence
        if phase:
            agent.phase = phase
        else:
            # Auto-derive phase from progress
            if progress < EXPLORATION_END:
                agent.phase = "exploration"
            elif progress < ANALYSIS_END:
                agent.phase = "analysis"
            else:
                agent.phase = "synthesis"
        if status:
            agent.status = status

    # ------------------------------------------------------------------
    # Rendering — public
    # ------------------------------------------------------------------

    def render(
        self,
        *,
        tick: int = 0,
        day: int = 0,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Clear the terminal and print the current frame.

        Args:
            tick: Current simulation tick for the header.
            day: Current simulation day for the header.
            extra_info: Arbitrary key-value pairs shown in the header.
        """
        sys.stdout.write(clear_screen())
        sys.stdout.write(self.render_to_string(tick=tick, day=day, extra_info=extra_info))
        sys.stdout.flush()

    def render_to_string(
        self,
        *,
        tick: int = 0,
        day: int = 0,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return the current frame as a string without writing to stdout.

        Args:
            tick: Current simulation tick for the header.
            day: Current simulation day for the header.
            extra_info: Arbitrary key-value pairs shown in the header.

        Returns:
            The rendered frame as a multi-line string.
        """
        self._frame += 1
        canvas = self._blank_canvas()
        self._draw_helix_backbone(canvas)
        self._draw_phase_boundaries(canvas)

        agents = list(self._agents.values())
        for agent in sorted(agents, key=lambda a: a.progress):
            self._place_agent(canvas, agent)

        sidebar = self._build_sidebar(agents)
        lines = self._merge(canvas, sidebar)
        header = self._build_header(tick, day, extra_info)
        footer = self._build_footer(agents)
        return "\n".join([header, *lines, footer, ""])

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def live(self) -> Iterator[HelixVisualizer]:
        """Context manager that hides the cursor on entry and restores it on exit.

        Yields:
            This ``HelixVisualizer`` instance.
        """
        try:
            sys.stdout.write(hide_cursor())
            sys.stdout.flush()
            yield self
        finally:
            sys.stdout.write(show_cursor())
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # Canvas internals
    # ------------------------------------------------------------------

    def _blank_canvas(self) -> List[List[str]]:
        """Create an empty canvas (list of character rows)."""
        return [[" "] * self._width for _ in range(self._height)]

    def _draw_helix_backbone(self, canvas: List[List[str]]) -> None:
        """Draw the spiral backbone as phase-coloured glyphs on the canvas."""
        centre = self._width // 2
        max_radius_chars = (self._width // 2) - 2

        for row in range(self._height):
            t = row / max(self._height - 1, 1)

            # Radius fraction normalised to [0, 1]
            z = self._helix.height * (1.0 - t)
            radius = self._helix.get_radius(z)
            max_r = self._helix.get_radius(self._helix.height)  # top radius
            r_frac = radius / max_r if max_r > 0 else 0

            # Angle with slow animation offset
            angle = self._helix.get_angle_at_t(t) + self._frame * 0.05
            x_offset = int(r_frac * max_radius_chars * math.cos(angle))
            col = centre + x_offset

            # Phase-based glyph and colour
            if t < EXPLORATION_END:
                glyph = _PHASE_ICONS["exploration"]
                colour = PHASE_COLORS["exploration"]
            elif t < ANALYSIS_END:
                glyph = _PHASE_ICONS["analysis"]
                colour = PHASE_COLORS["analysis"]
            else:
                glyph = _PHASE_ICONS["synthesis"]
                colour = PHASE_COLORS["synthesis"]

            if 0 <= col < self._width:
                canvas[row][col] = colorize(DIM + colour, glyph)

    def _draw_phase_boundaries(self, canvas: List[List[str]]) -> None:
        """Draw dashed horizontal lines at phase boundaries with labels."""
        explore_row = int(EXPLORATION_END * (self._height - 1))
        analysis_row = int(ANALYSIS_END * (self._height - 1))

        for col in range(self._width):
            if canvas[explore_row][col].strip() == "":
                canvas[explore_row][col] = colorize(DIM + PHASE_COLORS["exploration"], "-")
            if canvas[analysis_row][col].strip() == "":
                canvas[analysis_row][col] = colorize(DIM + PHASE_COLORS["analysis"], "-")

        # Phase labels centred on the boundary rows
        explore_label = " EXPLORATION "
        analysis_label = " ANALYSIS "
        e_start = max(0, (self._width - len(explore_label)) // 2)
        a_start = max(0, (self._width - len(analysis_label)) // 2)

        for i, ch in enumerate(explore_label):
            c = e_start + i
            if 0 <= c < self._width:
                canvas[explore_row][c] = colorize(BOLD + PHASE_COLORS["exploration"], ch)

        for i, ch in enumerate(analysis_label):
            c = a_start + i
            if 0 <= c < self._width:
                canvas[analysis_row][c] = colorize(BOLD + PHASE_COLORS["analysis"], ch)

    def _place_agent(self, canvas: List[List[str]], agent: AgentDisplayState) -> None:
        """Place an agent label on the canvas at its helix position."""
        t = agent.progress
        row = int(t * (self._height - 1))
        row = max(0, min(self._height - 1, row))

        centre = self._width // 2
        max_radius_chars = (self._width // 2) - 2

        z = self._helix.height * (1.0 - t)
        radius = self._helix.get_radius(z)
        max_r = self._helix.get_radius(self._helix.height)
        r_frac = radius / max_r if max_r > 0 else 0

        angle = self._helix.get_angle_at_t(t)
        x_offset = int(r_frac * max_radius_chars * math.cos(angle))
        col = centre + x_offset
        col = max(1, min(self._width - 2, col))

        # Confidence-based intensity
        intensity = BOLD if agent.confidence > 0.6 else ""
        label_str = f"[{agent.label}]"
        canvas[row][col] = colorize(intensity + agent.color, label_str)

        # Trim overlap from adjacent cells consumed by the multi-char label
        if col + 1 < self._width:
            canvas[row][col + 1] = ""
        if col + 2 < self._width:
            canvas[row][col + 2] = ""

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    def _build_sidebar(self, agents: List[AgentDisplayState]) -> List[str]:
        """Build the right-hand agent detail panel."""
        lines: List[str] = []
        lines.append(colorize(BOLD, " Agents"))
        lines.append(" " + "-" * (self._sidebar_width - 2))

        for agent in agents:
            phase_colour = PHASE_COLORS.get(agent.phase, RESET)

            # Agent header
            name = f"[{agent.label}] {agent.agent_id}"
            lines.append(f" {colorize(BOLD + agent.color, name)}")

            # Progress bar
            bar_width = self._sidebar_width - 14
            bar = progress_bar(agent.progress, bar_width)
            pct = f"{agent.progress * 100:5.1f}%"
            lines.append(f"   {colorize(agent.color, bar)} {pct}")

            # Confidence + phase
            conf_str = f"conf:{agent.confidence:.2f}"
            phase_str = agent.phase[:5].upper()
            lines.append(f"   {conf_str}  {colorize(phase_colour, phase_str)}")

            # Optional status line
            if agent.status:
                preview = agent.status[: self._sidebar_width - 6]
                if len(agent.status) > self._sidebar_width - 6:
                    preview = preview[: self._sidebar_width - 9] + "..."
                lines.append(f"   {colorize(DIM, preview)}")

            lines.append("")

        # Pad to canvas height
        while len(lines) < self._height:
            lines.append("")

        return lines[: self._height]

    # ------------------------------------------------------------------
    # Header / footer
    # ------------------------------------------------------------------

    def _build_header(
        self,
        tick: int,
        day: int,
        extra_info: Optional[Dict[str, Any]],
    ) -> str:
        """Build the header block above the helix canvas."""
        parts = [f"  {colorize(BOLD, self._title)}"]
        if tick or day:
            parts.append(f"Tick {tick}")
            parts.append(f"Day {day}")
        header_text = "  |  ".join(parts)

        # Extra info key-value pairs
        extra_line = ""
        if extra_info:
            kv = "  ".join(f"{k}={v}" for k, v in extra_info.items())
            extra_line = f"\n  {colorize(DIM, kv)}"

        separator = "  " + "\u2550" * (self._width + self._sidebar_width + 3)

        phase_legend = (
            f"  Phases: {colorize(PHASE_COLORS['exploration'], '~ EXPLORATION')}  "
            f"{colorize(PHASE_COLORS['analysis'], '= ANALYSIS')}  "
            f"{colorize(PHASE_COLORS['synthesis'], '# SYNTHESIS')}"
        )
        return f"\n{header_text}{extra_line}\n{separator}\n{phase_legend}\n"

    def _build_footer(self, agents: List[AgentDisplayState]) -> str:
        """Build the footer with team confidence bar."""
        separator = "  " + "\u2550" * (self._width + self._sidebar_width + 3)

        avg_conf = sum(a.confidence for a in agents) / len(agents) if agents else 0.0
        bar = progress_bar(avg_conf, 30)

        if avg_conf >= 0.75:
            conf_colour = PHASE_COLORS["synthesis"]  # green
        elif avg_conf >= 0.5:
            conf_colour = PHASE_COLORS["analysis"]  # yellow
        else:
            conf_colour = "\033[91m"  # red

        conf_line = (
            f"  Team Confidence: {colorize(conf_colour, bar)} "
            f"{colorize(BOLD + conf_colour, f'{avg_conf:.1%}')}"
        )
        return f"{separator}\n{conf_line}"

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _merge(self, canvas: List[List[str]], sidebar: List[str]) -> List[str]:
        """Merge the helix canvas rows with the sidebar rows."""
        lines: List[str] = []
        for row_idx in range(self._height):
            canvas_line = "".join(canvas[row_idx])
            side = sidebar[row_idx] if row_idx < len(sidebar) else ""
            lines.append(f"  {canvas_line} \u2502{side}")
        return lines
