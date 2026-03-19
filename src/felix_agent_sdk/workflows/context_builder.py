"""Collaborative context builder for Felix Agent SDK.

Accumulates agent outputs and builds enriched context for subsequent
agent processing rounds. Supports relevance scoring, deduplication,
and version tracking.

Algorithms ported from CalebisGross/felix ``src/workflows/context_builder.py``.
Refactored to remove CentralPost/MemoryFacade coupling.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Contribution:
    """A single agent's contribution to the collaborative context."""

    agent_id: str
    agent_type: str
    content: str
    confidence: float
    phase: str
    timestamp: float = field(default_factory=time.time)


class CollaborativeContextBuilder:
    """Builds enriched context from multiple agent contributions.

    Each processing round, agents add their results via
    :meth:`add_contribution`. The builder then provides merged context
    for the next round via :meth:`build_context` and
    :meth:`get_context_history`.
    """

    def __init__(self) -> None:
        self._contributions: list[Contribution] = []
        self._version: int = 0

    # ------------------------------------------------------------------
    # Add / query contributions
    # ------------------------------------------------------------------

    @property
    def version(self) -> int:
        """Context version — incremented on each contribution."""
        return self._version

    @property
    def contribution_count(self) -> int:
        return len(self._contributions)

    def add_contribution(
        self,
        agent_id: str,
        agent_type: str,
        content: str,
        confidence: float,
        phase: str = "exploration",
    ) -> None:
        """Record an agent's output as a contribution."""
        self._contributions.append(
            Contribution(
                agent_id=agent_id,
                agent_type=agent_type,
                content=content,
                confidence=confidence,
                phase=phase,
            )
        )
        self._version += 1

    def add_from_result(self, result: Any) -> None:
        """Add a contribution from an :class:`LLMResult` object."""
        self.add_contribution(
            agent_id=result.agent_id,
            agent_type=result.position_info.get("phase", "exploration"),
            content=result.content,
            confidence=result.confidence,
            phase=result.position_info.get("phase", "exploration"),
        )

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def build_context(self, max_entries: int = 10) -> str:
        """Build a merged context string, most relevant first.

        Contributions are scored by recency and confidence, then the top
        *max_entries* are selected and formatted.
        """
        scored = self._score_contributions()
        scored.sort(key=lambda pair: pair[1], reverse=True)
        selected = scored[:max_entries]

        if not selected:
            return ""

        parts: list[str] = []
        for contrib, _score in selected:
            parts.append(
                f"[{contrib.agent_id} ({contrib.phase}), "
                f"confidence={contrib.confidence:.2f}]: "
                f"{contrib.content}"
            )
        return "\n\n".join(parts)

    def get_context_history(self, max_entries: int = 10) -> list[dict[str, Any]]:
        """Return contributions formatted for ``LLMTask.context_history``.

        Each entry is ``{"agent_id": ..., "content": ...}``.
        """
        scored = self._score_contributions()
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [{"agent_id": c.agent_id, "content": c.content} for c, _ in scored[:max_entries]]

    def merge_contributions(self) -> dict[str, Any]:
        """Merge all contributions into a dict suitable for
        :meth:`ContextCompressor.compress_context`.
        """
        merged: dict[str, Any] = {}
        for contrib in self._contributions:
            key = f"{contrib.agent_id}_{contrib.phase}"
            merged[key] = contrib.content
        return merged

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def deduplicate(self, similarity_threshold: float = 0.6) -> int:
        """Remove near-duplicate contributions.

        Uses Jaccard similarity on keyword sets. Returns the number of
        contributions removed.
        """
        if len(self._contributions) < 2:
            return 0

        to_keep: list[Contribution] = [self._contributions[0]]
        removed = 0

        for contrib in self._contributions[1:]:
            is_dup = False
            kw_new = self._extract_keywords(contrib.content)
            for existing in to_keep:
                kw_existing = self._extract_keywords(existing.content)
                sim = self._jaccard(kw_new, kw_existing)
                if sim >= similarity_threshold:
                    is_dup = True
                    break
            if is_dup:
                removed += 1
            else:
                to_keep.append(contrib)

        self._contributions = to_keep
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_contributions(self) -> list[tuple[Contribution, float]]:
        """Score contributions by recency + confidence."""
        if not self._contributions:
            return []

        now = time.time()
        scored: list[tuple[Contribution, float]] = []
        for i, contrib in enumerate(self._contributions):
            # Recency: more recent = higher (0.0 – 0.5)
            age = now - contrib.timestamp
            recency = max(0.0, 0.5 - age * 0.01)
            # Confidence weight (0.0 – 0.5)
            conf = contrib.confidence * 0.5
            scored.append((contrib, recency + conf))
        return scored

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        words = re.findall(r"\b\w{4,}\b", text.lower())
        return set(words)

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)
