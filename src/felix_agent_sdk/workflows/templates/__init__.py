"""Pre-built workflow templates for common patterns."""

from felix_agent_sdk.workflows.templates.analysis import (
    create_config as analysis_config,
)
from felix_agent_sdk.workflows.templates.research import (
    create_config as research_config,
)
from felix_agent_sdk.workflows.templates.review import (
    create_config as review_config,
)

__all__ = [
    "research_config",
    "analysis_config",
    "review_config",
]
