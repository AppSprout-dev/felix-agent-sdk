"""Shared fixtures for integration tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the deep-research example's sibling modules importable in tests.
_example_dir = str(
    Path(__file__).resolve().parent.parent.parent / "examples" / "05_deep_research_live"
)
if _example_dir not in sys.path:
    sys.path.insert(0, _example_dir)
