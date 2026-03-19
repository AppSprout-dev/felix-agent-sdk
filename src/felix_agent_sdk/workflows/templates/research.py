"""Research workflow template — broad exploration to synthesis.

Uses a wide helix (``research_heavy``) with multiple research agents
for deep exploration before analysis and critique.
"""

from __future__ import annotations

from felix_agent_sdk.core.helix import HelixConfig
from felix_agent_sdk.workflows.config import SynthesisStrategy, WorkflowConfig


def create_config(
    confidence_threshold: float = 0.75,
    max_rounds: int = 4,
) -> WorkflowConfig:
    """Create a research-focused workflow configuration.

    Team: 2 Research + 1 Analysis + 1 Critic.
    Lower confidence threshold and more rounds allow deeper exploration.
    """
    return WorkflowConfig(
        helix_config=HelixConfig.research_heavy(),
        team_composition=[
            ("research", {}),
            ("research", {}),
            ("analysis", {}),
            ("critic", {}),
        ],
        confidence_threshold=confidence_threshold,
        max_rounds=max_rounds,
        synthesis_strategy=SynthesisStrategy.COMPRESSED_MERGE,
        max_agents=10,
    )
