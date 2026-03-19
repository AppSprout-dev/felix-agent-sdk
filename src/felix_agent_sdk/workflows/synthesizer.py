"""Workflow synthesis for the Felix Agent SDK.

Lightweight replacement for Felix's full SynthesisEngine. Combines agent
results into a single final output via one of three strategies.

Synthesis patterns derived from CalebisGross/felix workflow orchestration.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from felix_agent_sdk.agents.factory import AgentFactory
from felix_agent_sdk.agents.llm_agent import LLMResult, LLMTask
from felix_agent_sdk.memory.compression import (
    CompressionConfig,
    CompressionStrategy,
    ContextCompressor,
)
from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.workflows.config import SynthesisStrategy, WorkflowConfig

logger = logging.getLogger(__name__)


class WorkflowSynthesizer:
    """Synthesises agent results into a final workflow output.

    Args:
        provider: LLM provider for the synthesis call.
        config: Workflow configuration (determines strategy and compression).
    """

    def __init__(self, provider: BaseProvider, config: WorkflowConfig) -> None:
        self._provider = provider
        self._config = config

    def synthesize(
        self,
        results: list[LLMResult],
        task_description: str,
    ) -> str:
        """Produce a final synthesis string from *results*.

        Strategy dispatch:

        * ``BEST_RESULT`` — return the highest-confidence result.
        * ``COMPRESSED_MERGE`` — compress all results then run a
          synthesiser agent at ``t = 1.0``.
        * ``ROUND_ROBIN`` — return the last round's best result.
        """
        if not results:
            return ""

        strategy = self._config.synthesis_strategy

        if strategy == SynthesisStrategy.BEST_RESULT:
            return self._best_result(results)
        if strategy == SynthesisStrategy.ROUND_ROBIN:
            return self._round_robin(results)
        # Default: COMPRESSED_MERGE
        return self._compressed_merge(results, task_description)

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _best_result(results: list[LLMResult]) -> str:
        """Pick the single highest-confidence result."""
        best = max(results, key=lambda r: r.confidence)
        return best.content

    @staticmethod
    def _round_robin(results: list[LLMResult]) -> str:
        """Return the last result (most recent round's output)."""
        return results[-1].content

    def _compressed_merge(
        self,
        results: list[LLMResult],
        task_description: str,
    ) -> str:
        """Compress all results, then run a synthesis agent."""
        # Build context from all agent outputs
        context_dict: dict[str, Any] = {
            "task": task_description,
        }
        for r in results:
            key = f"{r.agent_id}_{r.position_info.get('phase', 'unknown')}"
            context_dict[key] = r.content

        # Optionally compress
        if self._config.enable_context_compression:
            compressor = ContextCompressor(
                config=self._config.compression_config
                or CompressionConfig(
                    strategy=CompressionStrategy.HIERARCHICAL_SUMMARY,
                )
            )
            compressed = compressor.compress_context(context_dict)
            context_text = json.dumps(compressed.content, indent=2)
        else:
            context_text = "\n\n".join(f"[{r.agent_id}]: {r.content}" for r in results)

        # Create a synthesis agent at the bottom of the helix (t=1.0)
        factory = AgentFactory(
            self._provider,
            helix_config=self._config.helix_config,
        )
        synth_agent = factory.create_agent(
            agent_type="llm",
            agent_id="synthesizer",
            spawn_time=0.0,
        )
        # Move to synthesis phase
        synth_agent.spawn(current_time=0.0)
        synth_agent._progress = 1.0  # noqa: SLF001 — place at synthesis end

        task = LLMTask(
            task_id="synthesis",
            description=(
                f"Synthesise the following agent outputs into a coherent, "
                f"concise final answer for the task: {task_description}"
            ),
            context=context_text,
        )

        result = synth_agent.process_task(task)
        return result.content
