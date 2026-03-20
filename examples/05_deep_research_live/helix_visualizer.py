"""Terminal helix visualizer — renders agents spiralling from exploration to synthesis.

Draws a side-view cross-section of the helix in the terminal, showing each
agent's position, phase, confidence, and temperature in real time.  Designed
to be eye-catching in demo recordings and Twitter clips.

No external dependencies beyond the Felix SDK and the Python stdlib.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from felix_agent_sdk.core.helix import ANALYSIS_END, EXPLORATION_END, HelixGeometry

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

# Phase colours (256-colour mode)
_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_MAGENTA = "\033[95m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

_CLEAR_SCREEN = "\033[2J\033[H"

_PHASE_COLOUR = {
    "exploration": _CYAN,
    "analysis": _YELLOW,
    "synthesis": _GREEN,
}

_AGENT_GLYPHS = {
    "research": "R",
    "analysis": "A",
    "critic": "C",
    "general": "G",
}

_PHASE_ICONS = {
    "exploration": "~",
    "analysis": "=",
    "synthesis": "#",
}


def _supports_colour() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_USE_COLOUR = _supports_colour()


def _c(code: str, text: str) -> str:
    return f"{code}{text}{_RESET}" if _USE_COLOUR else text


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class AgentSnapshot:
    """Lightweight snapshot of an agent's state for rendering."""

    agent_id: str
    agent_type: str
    progress: float  # t in [0, 1]
    confidence: float
    temperature: float
    phase: str
    content_preview: str = ""


# ---------------------------------------------------------------------------
# Helix renderer
# ---------------------------------------------------------------------------

# Canvas dimensions
HELIX_WIDTH = 50  # horizontal chars for the helix cross-section
HELIX_HEIGHT = 28  # vertical chars (top = t=0, bottom = t=1)
SIDEBAR_WIDTH = 38  # agent detail panel width


class HelixVisualizer:
    """Renders an ASCII helix with live agent positions."""

    def __init__(self, helix: HelixGeometry) -> None:
        self._helix = helix
        self._frame = 0
        self._start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def render_frame(
        self,
        agents: list[AgentSnapshot],
        round_num: int,
        max_rounds: int,
        status_line: str = "",
    ) -> str:
        """Return a complete terminal frame as a string."""
        self._frame += 1
        canvas = self._blank_canvas()

        # Draw helix backbone
        self._draw_helix_backbone(canvas)

        # Draw phase boundaries
        self._draw_phase_boundaries(canvas)

        # Place agents
        for agent in agents:
            self._place_agent(canvas, agent)

        # Compose sidebar
        sidebar = self._build_sidebar(agents)

        # Merge canvas + sidebar
        lines = self._merge(canvas, sidebar)

        # Header + footer
        elapsed = time.monotonic() - self._start_time
        header = self._build_header(round_num, max_rounds, elapsed)
        footer = self._build_footer(agents, status_line)

        return "\n".join([header, *lines, footer, ""])

    def print_frame(
        self,
        agents: list[AgentSnapshot],
        round_num: int,
        max_rounds: int,
        status_line: str = "",
    ) -> None:
        """Clear terminal and print the current frame."""
        frame = self.render_frame(agents, round_num, max_rounds, status_line)
        # Move cursor to top-left and clear screen
        sys.stdout.write(_CLEAR_SCREEN)
        sys.stdout.write(frame)
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Canvas
    # ------------------------------------------------------------------

    def _blank_canvas(self) -> list[list[str]]:
        return [[" "] * HELIX_WIDTH for _ in range(HELIX_HEIGHT)]

    def _draw_helix_backbone(self, canvas: list[list[str]]) -> None:
        """Draw the spiral backbone as dots on the canvas."""
        centre = HELIX_WIDTH // 2
        max_radius_chars = (HELIX_WIDTH // 2) - 2

        for row in range(HELIX_HEIGHT):
            t = row / max(HELIX_HEIGHT - 1, 1)
            # Get radius fraction (normalised to [0, 1])
            z = self._helix.height * (1.0 - t)
            radius = self._helix.get_radius(z)
            max_r = self._helix.get_radius(self._helix.height)  # top radius
            r_frac = radius / max_r if max_r > 0 else 0

            # Angle for this t value (plus slow animation offset)
            angle = self._helix.get_angle_at_t(t) + self._frame * 0.05
            x_offset = int(r_frac * max_radius_chars * math.cos(angle))
            col = centre + x_offset

            # Phase-based glyph
            if t < EXPLORATION_END:
                glyph = _PHASE_ICONS["exploration"]
                colour = _PHASE_COLOUR["exploration"]
            elif t < ANALYSIS_END:
                glyph = _PHASE_ICONS["analysis"]
                colour = _PHASE_COLOUR["analysis"]
            else:
                glyph = _PHASE_ICONS["synthesis"]
                colour = _PHASE_COLOUR["synthesis"]

            if 0 <= col < HELIX_WIDTH:
                canvas[row][col] = _c(_DIM + colour, glyph)

    def _draw_phase_boundaries(self, canvas: list[list[str]]) -> None:
        """Draw horizontal phase boundary lines."""
        explore_row = int(EXPLORATION_END * (HELIX_HEIGHT - 1))
        analysis_row = int(ANALYSIS_END * (HELIX_HEIGHT - 1))

        for col in range(HELIX_WIDTH):
            if canvas[explore_row][col].strip() == "" or canvas[explore_row][col] == " ":
                canvas[explore_row][col] = _c(_DIM + _CYAN, "-")
            if canvas[analysis_row][col].strip() == "" or canvas[analysis_row][col] == " ":
                canvas[analysis_row][col] = _c(_DIM + _YELLOW, "-")

    def _place_agent(self, canvas: list[list[str]], agent: AgentSnapshot) -> None:
        """Place an agent glyph on the canvas at its helix position."""
        t = agent.progress
        row = int(t * (HELIX_HEIGHT - 1))
        row = max(0, min(HELIX_HEIGHT - 1, row))

        centre = HELIX_WIDTH // 2
        max_radius_chars = (HELIX_WIDTH // 2) - 2

        z = self._helix.height * (1.0 - t)
        radius = self._helix.get_radius(z)
        max_r = self._helix.get_radius(self._helix.height)
        r_frac = radius / max_r if max_r > 0 else 0

        angle = self._helix.get_angle_at_t(t)
        x_offset = int(r_frac * max_radius_chars * math.cos(angle))
        col = centre + x_offset
        col = max(1, min(HELIX_WIDTH - 2, col))

        glyph = _AGENT_GLYPHS.get(agent.agent_type, "?")
        colour = _PHASE_COLOUR.get(agent.phase, _RESET)

        # Confidence-based intensity: high = bold, low = dim
        intensity = _BOLD if agent.confidence > 0.6 else ""

        canvas[row][col] = _c(intensity + colour, f"[{glyph}]")

        # Trim overlap from adjacent cells
        if col + 1 < HELIX_WIDTH:
            canvas[row][col + 1] = ""
        if col - 1 >= 0 and canvas[row][col - 1] == " ":
            canvas[row][col - 1] = " "

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    def _build_sidebar(
        self,
        agents: list[AgentSnapshot],
    ) -> list[str]:
        """Build the right-hand agent detail panel."""
        lines: list[str] = []
        lines.append(_c(_BOLD, " Agents"))
        lines.append(" " + "-" * (SIDEBAR_WIDTH - 2))

        for agent in agents:
            colour = _PHASE_COLOUR.get(agent.phase, _RESET)
            glyph = _AGENT_GLYPHS.get(agent.agent_type, "?")

            # Agent header line
            name = f"[{glyph}] {agent.agent_id}"
            lines.append(f" {_c(_BOLD + colour, name)}")

            # Progress bar
            bar_width = SIDEBAR_WIDTH - 14
            filled = int(agent.progress * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            pct = f"{agent.progress * 100:5.1f}%"
            lines.append(f"   {_c(colour, bar)} {pct}")

            # Stats line
            conf_str = f"conf:{agent.confidence:.2f}"
            temp_str = f"temp:{agent.temperature:.2f}"
            phase_str = agent.phase[:5].upper()
            lines.append(f"   {conf_str}  {temp_str}  {_c(colour, phase_str)}")

            # Content preview (truncated)
            if agent.content_preview:
                preview = agent.content_preview[: SIDEBAR_WIDTH - 6]
                if len(agent.content_preview) > SIDEBAR_WIDTH - 6:
                    preview = preview[: SIDEBAR_WIDTH - 9] + "..."
                lines.append(f"   {_c(_DIM, preview)}")

            lines.append("")

        # Pad to canvas height
        while len(lines) < HELIX_HEIGHT:
            lines.append("")

        return lines[:HELIX_HEIGHT]

    # ------------------------------------------------------------------
    # Merge & framing
    # ------------------------------------------------------------------

    def _merge(self, canvas: list[list[str]], sidebar: list[str]) -> list[str]:
        """Merge the helix canvas with the sidebar."""
        lines: list[str] = []
        for row_idx in range(HELIX_HEIGHT):
            canvas_line = "".join(canvas[row_idx])
            side = sidebar[row_idx] if row_idx < len(sidebar) else ""
            lines.append(f"  {canvas_line} │{side}")
        return lines

    def _build_header(self, round_num: int, max_rounds: int, elapsed: float) -> str:
        header_text = (
            f"  {_c(_BOLD, 'F E L I X')}  "
            f"Deep Research Demo  │  "
            f"Round {round_num}/{max_rounds}  │  "
            f"{elapsed:.1f}s elapsed"
        )
        separator = "  " + "═" * (HELIX_WIDTH + SIDEBAR_WIDTH + 3)
        phase_legend = (
            f"  Phases: {_c(_CYAN, '~ EXPLORE')}  "
            f"{_c(_YELLOW, '= ANALYSE')}  "
            f"{_c(_GREEN, '# SYNTHESISE')}"
        )
        return f"\n{header_text}\n{separator}\n{phase_legend}\n"

    def _build_footer(self, agents: list[AgentSnapshot], status_line: str) -> str:
        separator = "  " + "═" * (HELIX_WIDTH + SIDEBAR_WIDTH + 3)
        avg_conf = sum(a.confidence for a in agents) / len(agents) if agents else 0
        avg_bar_w = 30
        filled = int(avg_conf * avg_bar_w)
        avg_bar = "█" * filled + "░" * (avg_bar_w - filled)

        if avg_conf >= 0.75:
            conf_colour = _GREEN
        elif avg_conf >= 0.5:
            conf_colour = _YELLOW
        else:
            conf_colour = _RED
        conf_line = (
            f"  Team Confidence: {_c(conf_colour, avg_bar)} "
            f"{_c(_BOLD + conf_colour, f'{avg_conf:.1%}')}"
        )
        status = f"  {_c(_DIM, status_line)}" if status_line else ""
        return f"{separator}\n{conf_line}\n{status}"


# ---------------------------------------------------------------------------
# Transition animations
# ---------------------------------------------------------------------------


def print_intro() -> None:
    """Print an animated intro banner."""
    banner = r"""
       ╔═══════════════════════════════════════════════════════╗
       ║                                                       ║
       ║    ███████╗███████╗██╗     ██╗██╗  ██╗                ║
       ║    ██╔════╝██╔════╝██║     ██║╚██╗██╔╝                ║
       ║    █████╗  █████╗  ██║     ██║ ╚███╔╝                 ║
       ║    ██╔══╝  ██╔══╝  ██║     ██║ ██╔██╗                 ║
       ║    ██║     ███████╗███████╗██║██╔╝ ██╗                ║
       ║    ╚═╝     ╚══════╝╚══════╝╚═╝╚═╝  ╚═╝                ║
       ║                                                       ║
       ║    Deep Research Demo — Helical Agent Orchestration    ║
       ║    Agents spiral from exploration → synthesis          ║
       ║                                                       ║
       ╚═══════════════════════════════════════════════════════╝
    """
    sys.stdout.write(_CLEAR_SCREEN)
    for line in banner.strip().split("\n"):
        sys.stdout.write(_c(_CYAN, line) + "\n")
        sys.stdout.flush()
        time.sleep(0.05)
    time.sleep(1.5)


def print_phase_transition(phase_name: str) -> None:
    """Flash a phase transition banner."""
    colour = _PHASE_COLOUR.get(phase_name, _RESET)
    label = phase_name.upper()
    bar = "━" * 40
    msg = f"\n  {_c(colour, bar)}\n  {_c(_BOLD + colour, f'  ▶ ENTERING {label} PHASE')}\n  {_c(colour, bar)}\n"
    sys.stdout.write(msg)
    sys.stdout.flush()
    time.sleep(0.8)


def print_synthesis_result(synthesis: str, confidence: float) -> None:
    """Print the final synthesis with styling."""
    sys.stdout.write(_CLEAR_SCREEN)
    separator = "═" * 70
    conf_colour = _GREEN if confidence >= 0.75 else _YELLOW

    header = f"""
  {_c(_BOLD + _GREEN, separator)}
  {_c(_BOLD + _GREEN, "  F E L I X  —  SYNTHESIS COMPLETE")}
  {_c(_BOLD + _GREEN, separator)}

  {_c(_BOLD, "Final Confidence:")} {_c(conf_colour, f"{confidence:.1%}")}

  {_c(_BOLD, "Synthesised Research Report:")}
  {_c(_DIM, "─" * 60)}
"""
    sys.stdout.write(header)

    # Print synthesis with a typewriter effect
    words = synthesis.split()
    line_len = 0
    sys.stdout.write("  ")
    for word in words:
        if line_len + len(word) + 1 > 66:
            sys.stdout.write("\n  ")
            line_len = 0
        sys.stdout.write(word + " ")
        line_len += len(word) + 1
        sys.stdout.flush()
        time.sleep(0.02)

    footer = f"""

  {_c(_DIM, "─" * 60)}
  {_c(_DIM, "Powered by Felix Agent SDK — helical multi-agent orchestration")}
  {_c(_DIM, "github.com/AppSprout-dev/felix-agent-sdk")}
  {_c(_BOLD + _GREEN, separator)}
"""
    sys.stdout.write(footer)
    sys.stdout.flush()
