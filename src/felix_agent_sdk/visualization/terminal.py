"""ANSI terminal rendering primitives for the helix visualizer.

Pure-stdlib helpers for colour output, cursor control, and simple widgets.
No external dependencies.
"""

from __future__ import annotations

import os
import sys
from typing import Dict

# ---------------------------------------------------------------------------
# ANSI colour constants
# ---------------------------------------------------------------------------

CYAN: str = "\033[96m"
YELLOW: str = "\033[93m"
GREEN: str = "\033[92m"
RED: str = "\033[91m"
MAGENTA: str = "\033[95m"
DIM: str = "\033[2m"
BOLD: str = "\033[1m"
RESET: str = "\033[0m"

# Phase-to-colour mapping
PHASE_COLORS: Dict[str, str] = {
    "exploration": CYAN,
    "analysis": YELLOW,
    "synthesis": GREEN,
}


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------


def supports_color() -> bool:
    """Return True if the terminal is likely to support ANSI colour codes.

    Checks, in order:
    - ``NO_COLOR`` env var (https://no-color.org/) — disables colour.
    - ``FORCE_COLOR`` env var — forces colour on.
    - ``sys.stdout.isatty()`` — colour only when attached to a real terminal.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def colorize(code: str, text: str) -> str:
    """Wrap *text* in ANSI *code* if colour is supported.

    Args:
        code: One or more ANSI escape sequences (e.g. ``BOLD + CYAN``).
        text: The string to colour.

    Returns:
        The coloured string, or the plain *text* when colour is disabled.
    """
    if supports_color():
        return f"{code}{text}{RESET}"
    return text


def clear_screen() -> str:
    """Return the ANSI escape sequence to clear the terminal and home the cursor."""
    return "\033[2J\033[H"


def hide_cursor() -> str:
    """Return the ANSI escape sequence to hide the text cursor."""
    return "\033[?25l"


def show_cursor() -> str:
    """Return the ANSI escape sequence to show the text cursor."""
    return "\033[?25h"


def progress_bar(value: float, width: int = 20) -> str:
    """Render a simple progress bar using block characters.

    Args:
        value: Progress fraction in ``[0.0, 1.0]``.
        width: Total character width of the bar.

    Returns:
        A string like ``"████████░░░░░░░░░░░░"``.
    """
    value = max(0.0, min(1.0, value))
    filled = int(value * width)
    return "\u2588" * filled + "\u2591" * (width - filled)
