"""Rich mock provider for the deep research demo.

Returns phase-aware, substantive responses that make the terminal
visualisation feel like a real multi-agent research run — no API key needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.providers.types import CompletionResult

# ---------------------------------------------------------------------------
# Canned responses keyed by (agent_type, phase)
# ---------------------------------------------------------------------------

_RESPONSES: dict[tuple[str, str], list[str]] = {
    # ── Research agents ────────────────────────────────────────────────
    ("research", "exploration"): [
        (
            "Initial survey reveals three major research clusters around this topic. "
            "First, the computational approach — leveraging transformer architectures "
            "for pattern detection — has seen 47% growth in published papers since 2023. "
            "Second, the human-in-the-loop paradigm shows promise in domains where "
            "ground truth is scarce (medical imaging, legal analysis). Third, emerging "
            "work on synthetic data generation suggests we can overcome data scarcity "
            "bottlenecks. Key researchers to track: Chen et al. (2024) on adaptive "
            "attention, Patel & Rodriguez (2024) on domain transfer, and the DeepMind "
            "team's recent work on self-supervised pre-training."
        ),
        (
            "Expanding search to adjacent domains. Cross-referencing the computational "
            "cluster with deployment data reveals a gap: 78% of published architectures "
            "have no production deployment evidence. The human-in-the-loop work shows "
            "stronger real-world adoption (estimated 34% of Fortune 500 companies have "
            "piloted such systems). Notable outlier: the open-source community has "
            "produced 12 actively maintained frameworks, 3 of which have >10k GitHub "
            "stars, suggesting grassroots momentum independent of corporate R&D."
        ),
    ],
    ("research", "analysis"): [
        (
            "Narrowing focus based on exploration findings. The production-readiness "
            "gap is the critical insight. Comparing the top 5 frameworks by adoption "
            "metrics: Framework A leads in enterprise (42% market share) but lags in "
            "developer satisfaction (3.2/5). Framework C, despite being newest, shows "
            "the steepest adoption curve and highest satisfaction (4.6/5). This suggests "
            "architectural decisions in newer frameworks better align with practitioner "
            "needs. Cost analysis shows 60% reduction in compute requirements when "
            "using distilled models — a key enabler for broader adoption."
        ),
    ],
    ("research", "synthesis"): [
        (
            "Consolidating research findings into key evidence streams. Three factors "
            "drive adoption: (1) compute cost — must be <$0.01 per inference for mass "
            "market; (2) developer experience — frameworks with <30 min onboarding "
            "time see 5x adoption; (3) regulatory clarity — markets with clear AI "
            "governance see 2.3x faster enterprise deployment. The data strongly "
            "suggests that technical capability is no longer the bottleneck — "
            "infrastructure, tooling, and policy are."
        ),
    ],
    # ── Analysis agents ────────────────────────────────────────────────
    ("analysis", "exploration"): [
        (
            "Examining the structural relationships in the research landscape. "
            "Network analysis of citation graphs reveals two distinct communities "
            "that rarely cross-reference: the ML systems community (focused on "
            "efficiency, deployment, MLOps) and the AI research community (focused "
            "on novel architectures, benchmarks). This bifurcation may explain the "
            "production-readiness gap identified by the research agents. "
            "Preliminary pattern: breakthroughs in the research community take "
            "18-24 months to appear in systems-focused papers."
        ),
    ],
    ("analysis", "analysis"): [
        (
            "Deepening the structural analysis. The 18-24 month transfer lag has "
            "three identified bottlenecks: (a) reproducibility — only 63% of papers "
            "provide code, and only 41% of those reproduce within 10% of reported "
            "metrics; (b) hardware assumptions — 72% of novel architectures assume "
            "A100-class GPUs, while median enterprise deployment is on T4/L4-class; "
            "(c) evaluation mismatch — academic benchmarks measure accuracy, but "
            "production systems optimise for latency, cost, and reliability. "
            "Proposed synthesis: the winning strategy bridges these communities "
            "rather than choosing one side."
        ),
    ],
    ("analysis", "synthesis"): [
        (
            "Final analysis: the evidence converges on a 'bridge strategy' as the "
            "optimal approach. Organisations that maintain both a research pipeline "
            "(for capability advancement) and a systems pipeline (for deployment) "
            "outperform those that focus on either alone — by approximately 40% on "
            "composite metrics. The critical success factor is the translation layer "
            "between research and production, which requires dedicated tooling and "
            "personnel. Recommendation: invest 60% in systems, 30% in research, "
            "10% in the bridge function."
        ),
    ],
    # ── Critic agents ──────────────────────────────────────────────────
    ("critic", "exploration"): [
        (
            "Evaluating the methodological rigour of the research stream. Potential "
            "weaknesses identified: (1) the 47% growth figure may include pre-prints "
            "and non-peer-reviewed work; (2) the Fortune 500 adoption estimate lacks "
            "a clear source — could be survivorship bias in survey respondents; "
            "(3) GitHub stars as a proxy for adoption is unreliable (correlation "
            "with actual production use is ~0.3). These don't invalidate the "
            "conclusions but suggest confidence intervals should be wider than "
            "the point estimates imply."
        ),
    ],
    ("critic", "analysis"): [
        (
            "The analysis agent's 'bridge strategy' thesis is well-supported but "
            "has a key assumption worth challenging: the 40% outperformance figure "
            "comes from organisations that can afford dual pipelines, introducing "
            "selection bias (larger, better-resourced orgs). For smaller teams, the "
            "optimal strategy may differ. Additionally, the 60/30/10 split is "
            "presented without sensitivity analysis. However, the core insight — "
            "that the translation layer is the bottleneck — holds across multiple "
            "independent data points and is the strongest claim in the analysis."
        ),
    ],
    ("critic", "synthesis"): [
        (
            "Final assessment: the research is methodologically sound with noted "
            "caveats. Confidence level: HIGH for the core thesis (translation layer "
            "as bottleneck), MEDIUM for specific metrics (adoption rates, cost "
            "figures), LOW for the prescriptive allocation (60/30/10 split). "
            "The analysis would benefit from longitudinal data and controlled "
            "comparisons, but as a strategic framework it provides actionable "
            "guidance. Overall recommendation: proceed with the bridge strategy "
            "thesis as the primary finding."
        ),
    ],
    # ── General / synthesis agent (used by WorkflowSynthesizer) ──────
    ("llm", "synthesis"): [
        (
            "## AI Agent Frameworks: State of the Field\n\n"
            "### Key Findings\n\n"
            "Our multi-agent research team analysed the current landscape of AI agent "
            "frameworks through exploration, structured analysis, and critical review. "
            "Three convergent findings emerged:\n\n"
            "**1. The Production-Readiness Gap is the Central Challenge**\n"
            "78% of published agent architectures have no evidence of production deployment. "
            "The research-to-deployment pipeline takes 18-24 months, with bottlenecks in "
            "reproducibility (only 63% of papers provide code), hardware assumptions "
            "(72% target A100-class GPUs vs. median enterprise T4/L4), and evaluation "
            "mismatch (academic benchmarks vs. production metrics).\n\n"
            "**2. Developer Experience Drives Adoption More Than Capability**\n"
            "Frameworks with <30 minute onboarding see 5x higher adoption. The newest "
            "framework in our analysis (Framework C) has the steepest adoption curve and "
            "highest developer satisfaction (4.6/5) despite being least mature. Compute "
            "cost must reach <$0.01 per inference for mass-market viability.\n\n"
            "**3. The 'Bridge Strategy' is Optimal — With Caveats**\n"
            "Organisations maintaining both research and systems pipelines outperform "
            "single-focus teams by ~40% on composite metrics. The critical success factor "
            "is the translation layer between research and production. However, this "
            "finding carries selection bias toward larger, well-resourced organisations.\n\n"
            "### Confidence Assessment\n"
            "- **HIGH**: Translation layer as primary bottleneck\n"
            "- **MEDIUM**: Specific adoption rates and cost projections\n"
            "- **LOW**: Prescriptive resource allocation (60/30/10 split)\n\n"
            "### Recommendation\n"
            "Teams building multi-agent systems in 2025 should prioritise developer "
            "experience and deployment tooling over novel architectures. Invest in the "
            "bridge between research and production — this is where the highest leverage "
            "exists. Start with frameworks that optimise for onboarding speed and "
            "iterate toward capability, not the reverse."
        ),
    ],
}

# Fallback if an exact (type, phase) key isn't found
_FALLBACK = (
    "Processing the available information and contributing analysis. "
    "The evidence suggests multiple viable approaches, each with distinct "
    "trade-offs in cost, complexity, and scalability. Further investigation "
    "of the specific constraints would help narrow the recommendation space."
)


def _detect_phase(kwargs: dict) -> str:
    """Best-effort phase detection from the system prompt."""
    for msg in kwargs.get("messages", []):
        content = getattr(msg, "content", "") or ""
        lower = content.lower()
        if "exploration" in lower:
            return "exploration"
        if "analysis" in lower:
            return "analysis"
        if "synthesis" in lower:
            return "synthesis"
    return "exploration"


def _detect_agent_type(kwargs: dict) -> str:
    """Best-effort agent-type detection from the system prompt."""
    all_text = ""
    for msg in kwargs.get("messages", []):
        all_text += " " + (getattr(msg, "content", "") or "")
    lower = all_text.lower()

    # The WorkflowSynthesizer sends "Synthesise the following agent outputs"
    # via a general ("llm") agent at t=1.0 — detect this as the "llm" type
    # so it hits the dedicated synthesis response.
    if "synthesise the following" in lower or "synthesize the following" in lower:
        return "llm"
    if "research" in lower:
        return "research"
    if "analysis" in lower:
        return "analysis"
    if "critic" in lower:
        return "critic"
    return "research"


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_research_mock_provider() -> BaseProvider:
    """Build a mock provider that returns rich, phase-aware responses."""
    provider = MagicMock(spec=BaseProvider)
    call_state: dict[tuple[str, str], int] = {}

    def _complete(messages, **kwargs):
        # Reconstruct kwargs with messages for detection helpers
        full_kw = {**kwargs, "messages": messages}
        agent_type = _detect_agent_type(full_kw)
        phase = _detect_phase(full_kw)

        key = (agent_type, phase)
        responses = _RESPONSES.get(key, [_FALLBACK])
        idx = call_state.get(key, 0) % len(responses)
        call_state[key] = idx + 1

        content = responses[idx]
        token_est = len(content.split())
        return CompletionResult(
            content=content,
            model="felix-research-mock",
            usage={
                "prompt_tokens": 120,
                "completion_tokens": token_est,
                "total_tokens": 120 + token_est,
            },
        )

    provider.complete.side_effect = _complete
    provider.count_tokens.return_value = 120
    return provider
