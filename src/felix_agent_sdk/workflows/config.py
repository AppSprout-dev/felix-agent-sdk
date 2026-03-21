"""Workflow configuration and result types for the Felix Agent SDK.

Defines the declarative configuration that parameterises the workflow
runner and the structured result it produces.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from felix_agent_sdk.core.helix import HelixConfig
from felix_agent_sdk.memory.compression import CompressionConfig


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------


class WorkflowPhase(Enum):
    """High-level workflow phases (mirrors helix position phases)."""

    EXPLORATION = "exploration"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"


class SynthesisStrategy(Enum):
    """How the workflow synthesises agent results into a final output."""

    BEST_RESULT = "best_result"
    COMPRESSED_MERGE = "compressed_merge"
    ROUND_ROBIN = "round_robin"


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass
class WorkflowConfig:
    """Declarative configuration for a :class:`FelixWorkflow` run.

    Templates produce instances of this class via ``create_config()``
    factory functions.
    """

    helix_config: HelixConfig = field(default_factory=HelixConfig.default)
    team_composition: list[tuple[str, dict[str, Any]]] = field(
        default_factory=lambda: [
            ("research", {}),
            ("analysis", {}),
            ("critic", {}),
        ]
    )
    confidence_threshold: float = 0.80
    max_rounds: int = 3
    synthesis_strategy: SynthesisStrategy = SynthesisStrategy.COMPRESSED_MERGE
    max_agents: int = 10
    enable_context_compression: bool = True
    compression_config: Optional[CompressionConfig] = None
    enable_dynamic_spawning: bool = False
    max_dynamic_agents: int = 3


# ------------------------------------------------------------------
# Result
# ------------------------------------------------------------------


@dataclass
class WorkflowResult:
    """Structured output from a :class:`FelixWorkflow` run."""

    synthesis: str
    agent_results: list[Any] = field(default_factory=list)
    total_rounds: int = 0
    final_confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
