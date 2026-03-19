"""Felix workflow orchestration.

High-level workflow runner and pre-built templates for common patterns.
"""

from felix_agent_sdk.workflows.config import (
    SynthesisStrategy,
    WorkflowConfig,
    WorkflowPhase,
    WorkflowResult,
)
from felix_agent_sdk.workflows.context_builder import CollaborativeContextBuilder
from felix_agent_sdk.workflows.runner import FelixWorkflow, run_felix_workflow
from felix_agent_sdk.workflows.synthesizer import WorkflowSynthesizer

__all__ = [
    "FelixWorkflow",
    "run_felix_workflow",
    "WorkflowConfig",
    "WorkflowResult",
    "WorkflowPhase",
    "SynthesisStrategy",
    "CollaborativeContextBuilder",
    "WorkflowSynthesizer",
]
