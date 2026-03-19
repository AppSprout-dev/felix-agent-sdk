"""Review workflow template — multi-perspective critique.

Uses a balanced helix (``default``) with emphasis on critic agents
for thorough multi-perspective review and refinement.
"""

from __future__ import annotations

from felix_agent_sdk.core.helix import HelixConfig
from felix_agent_sdk.workflows.config import SynthesisStrategy, WorkflowConfig


def create_config(
    confidence_threshold: float = 0.80,
    max_rounds: int = 3,
) -> WorkflowConfig:
    """Create a review-focused workflow configuration.

    Team: 1 Research + 1 Analysis + 2 Critic.
    Multi-perspective critique with balanced helix.
    """
    return WorkflowConfig(
        helix_config=HelixConfig.default(),
        team_composition=[
            ("research", {}),
            ("analysis", {}),
            ("critic", {}),
            ("critic", {}),
        ],
        confidence_threshold=confidence_threshold,
        max_rounds=max_rounds,
        synthesis_strategy=SynthesisStrategy.COMPRESSED_MERGE,
        max_agents=10,
    )
