"""Position-aware token budget derivation for Felix agents.

Derives input/output token allocations from an agent's helix position:
- Exploration (t near 0): larger input budget for broad context ingestion.
- Synthesis (t near 1): larger output budget for comprehensive generation.
"""

from __future__ import annotations

from dataclasses import dataclass

from felix_agent_sdk.core.helix import HelixPosition


@dataclass(frozen=True)
class TokenBudget:
    """Token allocation derived from helix position.

    Attributes:
        max_input_tokens: Maximum tokens for the prompt/context.
        max_output_tokens: Maximum tokens for generation.
        reserved_tokens: Tokens reserved for system overhead.
    """

    max_input_tokens: int
    max_output_tokens: int
    reserved_tokens: int = 0

    @property
    def total(self) -> int:
        """Total usable tokens (input + output)."""
        return self.max_input_tokens + self.max_output_tokens

    @classmethod
    def from_helix_position(
        cls,
        position: HelixPosition,
        model_context_window: int = 128_000,
        reserved_tokens: int = 512,
    ) -> TokenBudget:
        """Derive a budget from the agent's current helix position.

        At t=0 (exploration): 70% input / 30% output.
        At t=1 (synthesis):   40% input / 60% output.
        Linear interpolation between these extremes.

        Args:
            position: Current position on the helix.
            model_context_window: Total context window of the target model.
            reserved_tokens: Tokens reserved for formatting overhead.
        """
        usable = model_context_window - reserved_tokens
        t = position.t

        # Input share decreases linearly from 0.70 to 0.40
        input_share = 0.70 - 0.30 * t
        output_share = 1.0 - input_share

        return cls(
            max_input_tokens=int(usable * input_share),
            max_output_tokens=int(usable * output_share),
            reserved_tokens=reserved_tokens,
        )

    @classmethod
    def default(cls) -> TokenBudget:
        """Reasonable default for standalone use (balanced 60/40 split)."""
        return cls(max_input_tokens=76_800, max_output_tokens=51_200, reserved_tokens=512)
