"""Content gap analysis for dynamic spawning decisions.

Examines agent outputs to identify topic coverage gaps and recommend
which agent type would best address them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from felix_agent_sdk.utils.text import STOPWORDS

# Minimum keyword length after stopword removal
_MIN_KEYWORD_LEN = 3

# Agent type recommendation thresholds
_LOW_COVERAGE = 0.3
_MID_COVERAGE = 0.6


@dataclass(frozen=True)
class CoverageReport:
    """Result of content gap analysis.

    Attributes:
        topics_covered: Keywords extracted from agent outputs.
        topics_sparse: Keywords that appear in fewer than 2 outputs.
        coverage_score: Ratio of well-covered topics to total topics.
        recommended_type: The agent type best suited to fill gaps.
    """

    topics_covered: Set[str] = field(default_factory=set)
    topics_sparse: Set[str] = field(default_factory=set)
    coverage_score: float = 0.0
    recommended_type: str = "research"


class ContentAnalyzer:
    """Analyse agent outputs to identify coverage gaps.

    Uses keyword extraction and topic overlap to determine whether
    the team's outputs collectively cover the problem space or if
    specific agent types are needed to fill gaps.
    """

    def analyze_coverage(self, results: List[Dict[str, Any]]) -> CoverageReport:
        """Analyse a list of agent result dicts for topic coverage.

        Each dict should have at least ``content`` (str) and
        ``agent_type`` (str) keys — matching the shape of
        ``LLMResult`` position_info dicts or simple test dicts.

        Returns:
            :class:`CoverageReport` with coverage metrics.
        """
        if not results:
            return CoverageReport()

        all_keywords: Set[str] = set()
        per_result_keywords: List[Set[str]] = []

        for r in results:
            content = r.get("content", "")
            kw = self._extract_keywords(content)
            all_keywords.update(kw)
            per_result_keywords.append(kw)

        # A topic is "sparse" if it appears in fewer than 2 results
        topic_counts: Dict[str, int] = {}
        for kw_set in per_result_keywords:
            for kw in kw_set:
                topic_counts[kw] = topic_counts.get(kw, 0) + 1

        sparse = {kw for kw, count in topic_counts.items() if count < 2}
        covered = all_keywords - sparse

        score = len(covered) / max(len(all_keywords), 1)
        recommended = self._recommend_type(score, results)

        return CoverageReport(
            topics_covered=covered,
            topics_sparse=sparse,
            coverage_score=round(score, 4),
            recommended_type=recommended,
        )

    def recommend_agent_type(self, coverage: CoverageReport) -> str:
        """Return the agent type string from a coverage report."""
        return coverage.recommended_type

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        words = re.findall(r"[a-zA-Z]+", text.lower())
        return {
            w for w in words
            if w not in STOPWORDS and len(w) >= _MIN_KEYWORD_LEN
        }

    def _recommend_type(
        self, score: float, results: List[Dict[str, Any]]
    ) -> str:
        """Decide which agent type to spawn based on coverage score."""
        if score < _LOW_COVERAGE:
            return "research"
        if score < _MID_COVERAGE:
            # Check if we're light on analysis
            types = [r.get("agent_type", "") for r in results]
            analysis_count = sum(1 for t in types if t == "analysis")
            if analysis_count < 2:
                return "analysis"
            return "research"
        # High coverage — add a critic to validate
        return "critic"
