"""LLM-powered agent for Felix Agent SDK.

Ported from CalebisGross/felix src/agents/llm_agent.py.
Refactored to use the provider-agnostic BaseProvider interface in place of
the original LMStudioClient coupling.

The agent's behaviour adapts based on its position on the helix:
- Top (wide, t near 0): broad exploration, high temperature.
- Middle: focused analysis, balanced processing.
- Bottom (narrow, t near 1): precise synthesis, low temperature.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from felix_agent_sdk.agents.base import Agent
from felix_agent_sdk.core.helix import HelixGeometry
from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.providers.types import ChatMessage, CompletionResult, MessageRole
from felix_agent_sdk.tokens.budget import TokenBudget

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LLMTask:
    """A unit of work for an LLM agent."""

    task_id: str
    description: str
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LLMResult:
    """Structured result from LLM agent task processing."""

    agent_id: str
    task_id: str
    content: str
    position_info: Dict[str, Any]
    completion_result: CompletionResult
    processing_time: float
    confidence: float
    temperature_used: float
    token_budget_used: int


# ---------------------------------------------------------------------------
# Temperature ranges per agent type (ported from Felix defaults)
# ---------------------------------------------------------------------------

_DEFAULT_TEMPERATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "research": (0.4, 0.9),
    "analysis": (0.2, 0.7),
    "synthesis": (0.1, 0.5),
    "critic": (0.1, 0.6),
}

_DEFAULT_TEMP_RANGE: Tuple[float, float] = (0.1, 0.9)


# ---------------------------------------------------------------------------
# LLMAgent
# ---------------------------------------------------------------------------


class LLMAgent(Agent):
    """LLM-powered agent that processes tasks via a provider-agnostic interface.

    Extends :class:`Agent` with LLM capabilities: adaptive temperature,
    position-aware prompting, content-quality confidence scoring, and
    helical-checkpoint gating.
    """

    HELICAL_CHECKPOINTS: ClassVar[List[float]] = [0.0, 0.3, 0.5, 0.7, 0.9]

    def __init__(
        self,
        agent_id: str,
        provider: BaseProvider,
        helix: HelixGeometry,
        *,
        spawn_time: float = 0.0,
        velocity: Optional[float] = None,
        agent_type: str = "general",
        temperature_range: Optional[Tuple[float, float]] = None,
        max_tokens: Optional[int] = None,
        token_budget: Optional[TokenBudget] = None,
    ) -> None:
        super().__init__(agent_id, helix, spawn_time=spawn_time, velocity=velocity)

        self.provider = provider
        self.agent_type = agent_type

        self.temperature_range = temperature_range or _DEFAULT_TEMPERATURE_RANGES.get(
            agent_type, _DEFAULT_TEMP_RANGE
        )

        self.max_tokens = max_tokens or 4096
        self.token_budget = token_budget

        # Processing state
        self.processing_results: List[LLMResult] = []
        self.total_tokens_used: int = 0
        self.total_processing_time: float = 0.0
        self._last_checkpoint_processed: int = -1

        # Feedback calibration (Phase 5+ integration point)
        self._confidence_calibration_offset: float = 0.0

    # ------------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------------

    def get_adaptive_temperature(self) -> float:
        """Map the current helix position to a temperature value.

        Top of helix (t = 0) yields the high end of the range;
        bottom (t = 1) yields the low end.
        """
        hint = self.position.temperature_hint  # 1.0 at t=0, 0.0 at t=1
        min_temp, max_temp = self.temperature_range
        temperature = min_temp + (max_temp - min_temp) * hint
        return max(min_temp, min(max_temp, temperature))

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def calculate_confidence(self, content: str) -> float:
        """Score confidence based on agent type, progress, and content quality.

        Agent-type confidence ranges (ported from Felix):
        - research:  0.30 – 0.60
        - analysis:  0.40 – 0.80
        - synthesis: 0.60 – 0.95
        - critic:    0.50 – 0.80
        - default:   0.30 – 0.70
        """
        depth = self._progress

        if self.agent_type == "research":
            base = 0.3 + depth * 0.3
            cap = 0.6
        elif self.agent_type == "analysis":
            base = 0.4 + depth * 0.4
            cap = 0.8
        elif self.agent_type == "synthesis":
            base = 0.6 + depth * 0.35
            cap = 0.95
        elif self.agent_type == "critic":
            base = 0.5 + depth * 0.3
            cap = 0.8
        else:
            base = 0.3 + depth * 0.4
            cap = 0.7

        content_bonus = self._analyze_content_quality(content) * 0.1
        consistency_bonus = self._calculate_consistency_bonus() * 0.05

        total = base + content_bonus + consistency_bonus + self._confidence_calibration_offset
        return min(max(total, 0.0), cap)

    def _analyze_content_quality(self, content: str) -> float:
        """Heuristic content-quality score (0.0 – 1.0)."""
        if not content or not content.strip():
            return 0.0

        lower = content.lower()
        score = 0.0

        # Length appropriateness (0.25 weight)
        length = len(content)
        if 100 <= length <= 2000:
            length_score = 1.0
        elif length < 100:
            length_score = length / 100.0
        else:
            length_score = max(0.3, 2000.0 / length)
        score += length_score * 0.25

        # Structure indicators (0.25 weight)
        structure_indicators = [
            "\n\n" in content,
            "." in content,
            any(w in lower for w in ("analysis", "research", "conclusion", "summary")),
            content.count(".") >= 3,
        ]
        score += (sum(structure_indicators) / len(structure_indicators)) * 0.25

        # Depth indicators (0.25 weight)
        depth_indicators = [
            any(w in lower for w in ("because", "therefore", "however", "moreover", "furthermore")),
            any(w in lower for w in ("data", "evidence", "study", "research", "analysis")),
            any(w in lower for w in ("consider", "suggest", "indicate", "demonstrate")),
            len(content.split()) > 50,
        ]
        score += (sum(depth_indicators) / len(depth_indicators)) * 0.25

        # Specificity indicators (0.25 weight)
        specificity_indicators = [
            any(c.isdigit() for c in content),
            content.count(",") > 2,
            any(w in lower for w in ("specific", "particular", "detail", "example")),
            '"' in content or "'" in content,
        ]
        score += (sum(specificity_indicators) / len(specificity_indicators)) * 0.25

        return min(score, 1.0)

    def _calculate_consistency_bonus(self) -> float:
        """Confidence-history stability bonus (0.0 – 1.0)."""
        if len(self._confidence_history) < 3:
            return 0.5
        recent = self._confidence_history[-3:]
        avg = sum(recent) / len(recent)
        variance = sum((c - avg) ** 2 for c in recent) / len(recent)
        return min(max(0.0, 1.0 - variance * 10), 1.0)

    # ------------------------------------------------------------------
    # Position-aware prompting
    # ------------------------------------------------------------------

    def create_position_aware_prompt(self, task: LLMTask) -> Tuple[str, str]:
        """Build (system_prompt, user_prompt) adapted to the agent's helix phase.

        - exploration: encourage breadth, divergent thinking.
        - analysis: encourage comparison, pattern recognition.
        - synthesis: encourage integration, concise conclusions.
        """
        phase = self.position.phase
        progress_pct = int(self._progress * 100)

        if phase == "exploration":
            directive = (
                "You are in the EXPLORATION phase. Generate diverse ideas, explore "
                "multiple angles, and gather broad information. Prioritise breadth "
                "over depth. Be creative and consider unconventional perspectives."
            )
        elif phase == "analysis":
            directive = (
                "You are in the ANALYSIS phase. Compare approaches, identify patterns, "
                "evaluate trade-offs, and organise findings. Balance breadth with depth. "
                "Focus on relationships between ideas."
            )
        else:
            directive = (
                "You are in the SYNTHESIS phase. Integrate previous findings into a "
                "coherent, concise conclusion. Prioritise clarity and actionable "
                "insight. Be precise and decisive."
            )

        system_prompt = (
            f"You are a {self.agent_type} agent at {progress_pct}% progress "
            f"({phase} phase).\n\n{directive}"
        )

        # Build user prompt from task
        parts = [task.description]
        if task.context:
            parts.append(f"\nAdditional context:\n{task.context}")
        if task.context_history:
            parts.append("\nPrevious agent outputs:")
            for entry in task.context_history:
                agent = entry.get("agent_id", "unknown")
                text = entry.get("content", "")
                parts.append(f"  [{agent}]: {text[:300]}")

        user_prompt = "\n".join(parts)
        return system_prompt, user_prompt

    # ------------------------------------------------------------------
    # Provider interaction
    # ------------------------------------------------------------------

    def _call_provider(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> CompletionResult:
        """Translate agent prompts into a BaseProvider call."""
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]
        return self.provider.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Main task processing
    # ------------------------------------------------------------------

    def process_task(self, task: LLMTask) -> LLMResult:
        """Process a task: prompt ➜ provider call ➜ confidence ➜ result.

        This is the primary entry point for running an agent on a task.
        """
        start = time.monotonic()

        temperature = self.get_adaptive_temperature()
        system_prompt, user_prompt = self.create_position_aware_prompt(task)
        max_tokens = self.max_tokens

        completion = self._call_provider(system_prompt, user_prompt, temperature, max_tokens)

        confidence = self.calculate_confidence(completion.content)
        self.record_confidence(confidence)

        elapsed = time.monotonic() - start
        self.total_tokens_used += completion.total_tokens
        self.total_processing_time += elapsed

        result = LLMResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            content=completion.content,
            position_info=self.get_position_info(),
            completion_result=completion,
            processing_time=elapsed,
            confidence=confidence,
            temperature_used=temperature,
            token_budget_used=completion.total_tokens,
        )
        self.processing_results.append(result)
        return result

    # ------------------------------------------------------------------
    # Checkpoint gating
    # ------------------------------------------------------------------

    def should_process_at_checkpoint(self) -> bool:
        """Return ``True`` if the agent has crossed a new helical checkpoint."""
        current_index = -1
        for i, cp in enumerate(self.HELICAL_CHECKPOINTS):
            if self._progress >= cp:
                current_index = i
        return current_index > self._last_checkpoint_processed

    def mark_checkpoint_processed(self) -> None:
        """Mark the current checkpoint as handled."""
        for i, cp in enumerate(self.HELICAL_CHECKPOINTS):
            if self._progress >= cp:
                self._last_checkpoint_processed = i

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LLMAgent(id={self.agent_id!r}, type={self.agent_type!r}, "
            f"state={self._state.value!r}, progress={self._progress:.3f})"
        )
