"""Analysis workflow template — data-focused convergence.

Uses a narrow helix (``fast_convergence``) with emphasis on analysis
agents for precision-oriented tasks.
"""

from __future__ import annotations

from felix_agent_sdk.core.helix import HelixConfig
from felix_agent_sdk.workflows.config import SynthesisStrategy, WorkflowConfig


def create_config(
    confidence_threshold: float = 0.85,
    max_rounds: int = 3,
) -> WorkflowConfig:
    """Create an analysis-focused workflow configuration.

    Team: 1 Research + 2 Analysis + 1 Critic.
    Higher confidence threshold for data-focused precision.
    """
    return WorkflowConfig(
        helix_config=HelixConfig.fast_convergence(),
        team_composition=[
            ("research", {}),
            ("analysis", {}),
            ("analysis", {}),
            ("critic", {}),
        ],
        confidence_threshold=confidence_threshold,
        max_rounds=max_rounds,
        synthesis_strategy=SynthesisStrategy.COMPRESSED_MERGE,
        max_agents=10,
    )
