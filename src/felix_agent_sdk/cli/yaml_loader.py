"""YAML config loader for the Felix CLI.

Parses a ``felix.yaml`` file into a :class:`WorkflowConfig`, a task
description string, and provider configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from felix_agent_sdk.core.helix import HelixConfig
from felix_agent_sdk.workflows.config import SynthesisStrategy, WorkflowConfig

_HELIX_PRESETS = {
    "default": HelixConfig.default,
    "research_heavy": HelixConfig.research_heavy,
    "fast_convergence": HelixConfig.fast_convergence,
}


def load_workflow_yaml(
    path: str | Path,
) -> Tuple[WorkflowConfig, str, Dict[str, Any]]:
    """Parse a ``felix.yaml`` into SDK-native types.

    Args:
        path: Path to the YAML config file.

    Returns:
        Tuple of (WorkflowConfig, task_description, provider_info).
        ``provider_info`` is a dict with keys ``provider`` and ``model``.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required fields are missing or invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping, got {type(raw).__name__}")

    # --- Task (required) ---
    task = raw.get("task")
    if not task:
        raise ValueError("Missing required 'task' field in config")

    # --- Helix ---
    helix_config = _parse_helix(raw.get("helix", "default"))

    # --- Team composition ---
    team = _parse_team(raw.get("team"))

    # --- Scalars ---
    kwargs: Dict[str, Any] = {}
    if "confidence_threshold" in raw:
        kwargs["confidence_threshold"] = float(raw["confidence_threshold"])
    if "max_rounds" in raw:
        kwargs["max_rounds"] = int(raw["max_rounds"])
    if "max_agents" in raw:
        kwargs["max_agents"] = int(raw["max_agents"])
    if "synthesis_strategy" in raw:
        kwargs["synthesis_strategy"] = SynthesisStrategy(raw["synthesis_strategy"])
    if "enable_dynamic_spawning" in raw:
        kwargs["enable_dynamic_spawning"] = bool(raw["enable_dynamic_spawning"])
    if "max_dynamic_agents" in raw:
        kwargs["max_dynamic_agents"] = int(raw["max_dynamic_agents"])

    config = WorkflowConfig(
        helix_config=helix_config,
        team_composition=team,
        **kwargs,
    )

    # --- Provider info ---
    provider_info = {
        "provider": raw.get("provider", ""),
        "model": raw.get("model", ""),
    }

    return config, str(task), provider_info


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _parse_helix(value: Any) -> HelixConfig:
    """Parse helix field — string preset or dict of params."""
    if isinstance(value, str):
        factory = _HELIX_PRESETS.get(value)
        if factory is None:
            valid = ", ".join(_HELIX_PRESETS)
            raise ValueError(f"Unknown helix preset '{value}'. Valid: {valid}")
        return factory()
    if isinstance(value, dict):
        return HelixConfig(
            top_radius=float(value.get("top_radius", 3.0)),
            bottom_radius=float(value.get("bottom_radius", 0.5)),
            height=float(value.get("height", 8.0)),
            turns=float(value.get("turns", 2.0)),
        )
    raise ValueError(f"'helix' must be a preset name or dict, got {type(value).__name__}")


def _parse_team(
    value: Any,
) -> list[tuple[str, dict[str, Any]]]:
    """Parse team field — list of {type: ...} dicts."""
    if value is None:
        return [("research", {}), ("analysis", {}), ("critic", {})]
    if not isinstance(value, list):
        raise ValueError(f"'team' must be a list, got {type(value).__name__}")

    result: list[tuple[str, dict[str, Any]]] = []
    for entry in value:
        if isinstance(entry, str):
            result.append((entry, {}))
        elif isinstance(entry, dict):
            agent_type = entry.pop("type", "llm")
            result.append((str(agent_type), dict(entry)))
        else:
            raise ValueError(f"Team entry must be a string or dict, got {type(entry).__name__}")
    return result
