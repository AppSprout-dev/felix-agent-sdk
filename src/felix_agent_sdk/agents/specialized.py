"""Specialized LLM agent types for Felix Agent SDK.

Ported from CalebisGross/felix src/agents/specialized_agents.py.
Provides role-specific prompt overrides and behaviour for Research,
Analysis, and Critic agents.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from felix_agent_sdk.agents.llm_agent import LLMAgent, LLMTask
from felix_agent_sdk.core.helix import HelixGeometry
from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.tokens.budget import TokenBudget

logger = logging.getLogger(__name__)


class ResearchAgent(LLMAgent):
    """Research agent specialising in broad information gathering.

    High creativity at the top of the helix, gradually narrowing focus
    as it descends.
    """

    def __init__(
        self,
        agent_id: str,
        provider: BaseProvider,
        helix: HelixGeometry,
        *,
        spawn_time: float = 0.0,
        velocity: Optional[float] = None,
        research_domain: str = "general",
        temperature_range: Optional[Tuple[float, float]] = None,
        max_tokens: Optional[int] = None,
        token_budget: Optional[TokenBudget] = None,
    ) -> None:
        super().__init__(
            agent_id,
            provider,
            helix,
            spawn_time=spawn_time,
            velocity=velocity,
            agent_type="research",
            temperature_range=temperature_range,
            max_tokens=max_tokens,
            token_budget=token_budget,
        )
        self.research_domain = research_domain

    def create_position_aware_prompt(self, task: LLMTask) -> Tuple[str, str]:
        """Research-specific prompt framing per helix phase."""
        phase = self.position.phase
        progress_pct = int(self._progress * 100)

        if phase == "exploration":
            directive = (
                "You are a research agent in the EXPLORATION phase. Cast a wide net: "
                "survey the landscape, identify relevant sources, generate diverse "
                "hypotheses, and surface unconventional angles. Breadth is more "
                "important than depth at this stage."
            )
        elif phase == "analysis":
            directive = (
                "You are a research agent in the ANALYSIS phase. Compare findings "
                "from your earlier exploration, identify patterns and contradictions, "
                "and begin prioritising the most promising lines of inquiry."
            )
        else:
            directive = (
                "You are a research agent in the SYNTHESIS phase. Distil your "
                "research into a concise summary of key findings. Highlight the "
                "strongest evidence and most actionable insights."
            )

        domain_note = (
            f" Your domain focus is '{self.research_domain}'."
            if self.research_domain != "general"
            else ""
        )
        system_prompt = (
            f"Research agent at {progress_pct}% progress ({phase} phase).{domain_note}"
            f"\n\n{directive}"
        )

        parts = [task.description]
        if task.context:
            parts.append(f"\nAdditional context:\n{task.context}")
        if task.context_history:
            parts.append("\nPrevious agent outputs:")
            for entry in task.context_history:
                agent = entry.get("agent_id", "unknown")
                text = entry.get("content", "")
                parts.append(f"  [{agent}]: {text[:300]}")

        return system_prompt, "\n".join(parts)


class AnalysisAgent(LLMAgent):
    """Analysis agent specialising in processing and organising information.

    Tracks accumulated patterns and insights across multiple task invocations.
    """

    def __init__(
        self,
        agent_id: str,
        provider: BaseProvider,
        helix: HelixGeometry,
        *,
        spawn_time: float = 0.0,
        velocity: Optional[float] = None,
        analysis_type: str = "general",
        temperature_range: Optional[Tuple[float, float]] = None,
        max_tokens: Optional[int] = None,
        token_budget: Optional[TokenBudget] = None,
    ) -> None:
        super().__init__(
            agent_id,
            provider,
            helix,
            spawn_time=spawn_time,
            velocity=velocity,
            agent_type="analysis",
            temperature_range=temperature_range,
            max_tokens=max_tokens,
            token_budget=token_budget,
        )
        self.analysis_type = analysis_type
        self.patterns_found: List[str] = []
        self.insights: List[str] = []

    def create_position_aware_prompt(self, task: LLMTask) -> Tuple[str, str]:
        """Analysis-specific prompt framing per helix phase."""
        phase = self.position.phase
        progress_pct = int(self._progress * 100)

        if phase == "exploration":
            directive = (
                "You are an analysis agent in the EXPLORATION phase. Identify the "
                "key dimensions and frameworks relevant to this problem. Begin "
                "categorising information and spotting early patterns."
            )
        elif phase == "analysis":
            directive = (
                "You are an analysis agent in the ANALYSIS phase. Perform deep "
                "comparative analysis. Evaluate trade-offs, rank alternatives, "
                "and identify causal relationships."
            )
        else:
            directive = (
                "You are an analysis agent in the SYNTHESIS phase. Consolidate your "
                "analysis into clear conclusions. Present the strongest findings "
                "with supporting evidence."
            )

        type_note = (
            f" Analysis type: '{self.analysis_type}'." if self.analysis_type != "general" else ""
        )
        system_prompt = (
            f"Analysis agent at {progress_pct}% progress ({phase} phase).{type_note}\n\n{directive}"
        )

        parts = [task.description]
        if task.context:
            parts.append(f"\nAdditional context:\n{task.context}")
        if task.context_history:
            parts.append("\nPrevious agent outputs:")
            for entry in task.context_history:
                agent = entry.get("agent_id", "unknown")
                text = entry.get("content", "")
                parts.append(f"  [{agent}]: {text[:300]}")

        return system_prompt, "\n".join(parts)


class CriticAgent(LLMAgent):
    """Critic agent specialising in quality assurance and review.

    Includes meta-cognitive evaluation of other agents' reasoning processes.
    """

    def __init__(
        self,
        agent_id: str,
        provider: BaseProvider,
        helix: HelixGeometry,
        *,
        spawn_time: float = 0.0,
        velocity: Optional[float] = None,
        review_focus: str = "general",
        temperature_range: Optional[Tuple[float, float]] = None,
        max_tokens: Optional[int] = None,
        token_budget: Optional[TokenBudget] = None,
    ) -> None:
        super().__init__(
            agent_id,
            provider,
            helix,
            spawn_time=spawn_time,
            velocity=velocity,
            agent_type="critic",
            temperature_range=temperature_range,
            max_tokens=max_tokens,
            token_budget=token_budget,
        )
        self.review_focus = review_focus
        self.identified_issues: List[str] = []
        self.suggestions: List[str] = []

    def create_position_aware_prompt(self, task: LLMTask) -> Tuple[str, str]:
        """Critic-specific prompt framing per helix phase."""
        phase = self.position.phase
        progress_pct = int(self._progress * 100)

        if phase == "exploration":
            directive = (
                "You are a critic agent in the EXPLORATION phase. Survey the work "
                "produced so far and identify potential issues, gaps, or areas that "
                "deserve deeper scrutiny."
            )
        elif phase == "analysis":
            directive = (
                "You are a critic agent in the ANALYSIS phase. Systematically "
                "evaluate the quality of reasoning, evidence, and methodology. "
                "Identify logical fallacies, unsupported claims, and weaknesses."
            )
        else:
            directive = (
                "You are a critic agent in the SYNTHESIS phase. Deliver a final "
                "quality assessment. Score the overall reasoning, flag any "
                "remaining issues, and provide concise improvement recommendations."
            )

        focus_note = (
            f" Review focus: '{self.review_focus}'." if self.review_focus != "general" else ""
        )
        system_prompt = (
            f"Critic agent at {progress_pct}% progress ({phase} phase).{focus_note}\n\n{directive}"
        )

        parts = [task.description]
        if task.context:
            parts.append(f"\nAdditional context:\n{task.context}")
        if task.context_history:
            parts.append("\nPrevious agent outputs:")
            for entry in task.context_history:
                agent = entry.get("agent_id", "unknown")
                text = entry.get("content", "")
                parts.append(f"  [{agent}]: {text[:300]}")

        return system_prompt, "\n".join(parts)

    # ------------------------------------------------------------------
    # Meta-cognitive evaluation (ported from Felix CriticAgent)
    # ------------------------------------------------------------------

    def evaluate_reasoning_process(
        self,
        agent_output: Dict[str, Any],
        agent_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate how another agent reasoned, not just what it produced.

        Returns a dictionary with scores and actionable feedback.
        """
        result_text = agent_output.get("result", "")
        confidence = agent_output.get("confidence", 0.5)
        agent_id = agent_output.get("agent_id", "unknown")

        issues: List[str] = []
        recommendations: List[str] = []
        scores = {"logical_coherence": 0.5, "evidence_quality": 0.5, "methodology": 0.5}

        if self._has_logical_fallacies(result_text):
            issues.append("Contains potential logical fallacies")
            scores["logical_coherence"] = 0.4
            recommendations.append("Review reasoning chain for logical consistency")
        else:
            scores["logical_coherence"] = 0.8

        if self._has_weak_evidence(result_text):
            issues.append("Evidence appears weak or unsupported")
            scores["evidence_quality"] = 0.4
            recommendations.append("Strengthen claims with more reliable evidence")
        else:
            scores["evidence_quality"] = 0.8

        if agent_metadata:
            atype = agent_metadata.get("agent_type", "unknown")
            if not self._methodology_appropriate(result_text, atype):
                issues.append(f"Methodology not well-suited for {atype} agent")
                scores["methodology"] = 0.4
                recommendations.append(f"Consider approaches more aligned with {atype} role")
            else:
                scores["methodology"] = 0.8
        else:
            scores["methodology"] = 0.6

        if len(result_text.split()) < 50:
            issues.append("Reasoning appears shallow — insufficient depth")
            recommendations.append("Provide more detailed reasoning and analysis")
            for key in scores:
                scores[key] *= 0.9

        avg_score = sum(scores.values()) / len(scores)
        gap = abs(confidence - avg_score)
        if gap > 0.3:
            if confidence > avg_score:
                issues.append(
                    f"Agent appears overconfident "
                    f"(confidence={confidence:.2f} vs quality={avg_score:.2f})"
                )
                recommendations.append("Calibrate confidence based on reasoning quality")
            else:
                issues.append(
                    f"Agent appears underconfident "
                    f"(confidence={confidence:.2f} vs quality={avg_score:.2f})"
                )
                recommendations.append("Increase confidence when reasoning is solid")

        reasoning_quality = sum(scores.values()) / len(scores)
        re_evaluation_needed = reasoning_quality < 0.5 or len(issues) >= 3

        return {
            "reasoning_quality_score": reasoning_quality,
            "logical_coherence": scores["logical_coherence"],
            "evidence_quality": scores["evidence_quality"],
            "methodology_appropriateness": scores["methodology"],
            "identified_issues": issues,
            "improvement_recommendations": recommendations,
            "re_evaluation_needed": re_evaluation_needed,
            "agent_id": agent_id,
        }

    # -- helper heuristics (ported from Felix) --

    @staticmethod
    def _has_logical_fallacies(text: str) -> bool:
        indicators = [
            "everyone knows",
            "obviously",
            "clearly",
            "it goes without saying",
            "all experts agree",
            "no one would disagree",
            "always",
            "never",
        ]
        lower = text.lower()
        return any(ind in lower for ind in indicators)

    @staticmethod
    def _has_weak_evidence(text: str) -> bool:
        weak = [
            "i think",
            "i believe",
            "probably",
            "maybe",
            "might be",
            "could be",
            "seems like",
            "appears to",
        ]
        lower = text.lower()
        return sum(1 for w in weak if w in lower) > 3

    @staticmethod
    def _methodology_appropriate(text: str, agent_type: str) -> bool:
        lower = text.lower()
        if agent_type == "research":
            return "perspective" in lower or "source" in lower
        if agent_type == "analysis":
            return "because" in lower or "therefore" in lower
        if agent_type == "critic":
            return "issue" in lower or "problem" in lower or "improve" in lower
        return True
