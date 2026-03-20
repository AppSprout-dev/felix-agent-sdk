#!/usr/bin/env python3
"""Felix Deep Research Demo — live terminal visualisation.

Runs a multi-agent research workflow with a real-time ASCII helix showing
agents spiralling from broad exploration to focused synthesis.

No API key required — uses a rich mock provider with realistic responses.

Usage:
    python examples/05_deep_research_live/run.py
    python examples/05_deep_research_live/run.py --topic "quantum computing in drug discovery"
    python examples/05_deep_research_live/run.py --fast          # skip animations
    python examples/05_deep_research_live/run.py --provider openai  # use real provider
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Enable UTF-8 output on Windows so box-drawing chars render correctly
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Allow running as: python examples/05_deep_research_live/run.py
# by adding the example directory to sys.path for sibling imports.
_example_dir = str(Path(__file__).resolve().parent)
if _example_dir not in sys.path:  # noqa: E402
    sys.path.insert(0, _example_dir)

from felix_agent_sdk.agents.base import AgentState  # noqa: E402
from felix_agent_sdk.agents.factory import AgentFactory  # noqa: E402
from felix_agent_sdk.agents.llm_agent import LLMAgent, LLMResult, LLMTask  # noqa: E402
from felix_agent_sdk.communication.central_post import CentralPost  # noqa: E402
from felix_agent_sdk.communication.messages import MessageType  # noqa: E402
from felix_agent_sdk.communication.spoke import SpokeManager  # noqa: E402
from felix_agent_sdk.core.helix import HelixConfig  # noqa: E402
from felix_agent_sdk.providers.base import BaseProvider  # noqa: E402
from felix_agent_sdk.workflows.config import SynthesisStrategy, WorkflowConfig  # noqa: E402
from felix_agent_sdk.workflows.context_builder import CollaborativeContextBuilder  # noqa: E402
from felix_agent_sdk.workflows.synthesizer import WorkflowSynthesizer  # noqa: E402

from _mock_research import make_research_mock_provider  # noqa: E402
from helix_visualizer import (  # noqa: E402
    AgentSnapshot,
    HelixVisualizer,
    print_intro,
    print_phase_transition,
    print_synthesis_result,
)

# ---------------------------------------------------------------------------
# Default topic
# ---------------------------------------------------------------------------

DEFAULT_TOPIC = (
    "Analyse the current state of AI agent frameworks: what architectural "
    "patterns are emerging, what gaps exist between research and production "
    "deployment, and what strategies should teams adopt for building reliable "
    "multi-agent systems in 2025?"
)

# ---------------------------------------------------------------------------
# Demo config
# ---------------------------------------------------------------------------


def build_config() -> WorkflowConfig:
    """Research-heavy workflow: 2 Research + 1 Analysis + 1 Critic, 4 rounds."""
    return WorkflowConfig(
        helix_config=HelixConfig.research_heavy(),
        team_composition=[
            ("research", {}),
            ("research", {}),
            ("analysis", {}),
            ("critic", {}),
        ],
        confidence_threshold=0.78,
        max_rounds=4,
        synthesis_strategy=SynthesisStrategy.COMPRESSED_MERGE,
        max_agents=10,
    )


# ---------------------------------------------------------------------------
# Snapshot extraction
# ---------------------------------------------------------------------------


def agent_to_snapshot(agent: LLMAgent, last_content: str = "") -> AgentSnapshot:
    """Convert a live agent to a renderable snapshot."""
    return AgentSnapshot(
        agent_id=agent.agent_id,
        agent_type=agent.agent_type,
        progress=agent.progress,
        confidence=agent.confidence,
        temperature=agent.get_adaptive_temperature(),
        phase=agent.position.phase,
        content_preview=last_content[:80] if last_content else "",
    )


# ---------------------------------------------------------------------------
# Per-agent processing step (extracted to reduce cognitive complexity)
# ---------------------------------------------------------------------------


@dataclass
class _RoundContext:
    """Mutable state passed through a single processing round."""

    agents: list[LLMAgent]
    spoke_mgr: SpokeManager
    builder: CollaborativeContextBuilder
    viz: HelixVisualizer
    topic: str
    round_num: int
    max_rounds: int
    last_contents: dict[str, str]
    prev_phase: str | None
    fast: bool


def _process_agent(
    agent: LLMAgent,
    current_time: float,
    ctx: _RoundContext,
) -> LLMResult | None:
    """Advance a single agent and return its result, or None if it didn't produce one."""
    # Spawn if ready
    if agent.state == AgentState.WAITING and agent.can_spawn(current_time):
        agent.spawn(current_time)

    if agent.state != AgentState.ACTIVE:
        return None

    agent.update_position(current_time)

    # Phase transition animation
    phase = agent.position.phase
    if ctx.prev_phase and phase != ctx.prev_phase and not ctx.fast:
        print_phase_transition(phase)
    ctx.prev_phase = phase

    if not agent.should_process_at_checkpoint():
        return None

    # Show "thinking" frame
    _render_frame(ctx, f"{agent.agent_id} ({agent.agent_type}) processing...")
    _pause(0.6, ctx.fast)

    # Process task
    task = LLMTask(
        task_id=f"{agent.agent_id}-r{ctx.round_num}",
        description=ctx.topic,
        context="",
        context_history=ctx.builder.get_context_history(),
    )
    result = agent.process_task(task)
    agent.mark_checkpoint_processed()
    ctx.builder.add_from_result(result)

    # Route message through hub
    spoke = ctx.spoke_mgr.get_spoke(agent.agent_id)
    if spoke:
        spoke.send_message(
            message_type=MessageType.STATUS_UPDATE,
            content={
                "confidence": result.confidence,
                "phase": result.position_info.get("phase", ""),
                "progress": result.position_info.get("progress", 0.0),
            },
        )

    ctx.last_contents[agent.agent_id] = result.content

    # Render updated frame
    _render_frame(ctx, f"{agent.agent_id} done — confidence {result.confidence:.2f}")
    _pause(0.8, ctx.fast)

    return result


def _render_frame(ctx: _RoundContext, status: str) -> None:
    """Build snapshots from live agents and render a visualiser frame."""
    snapshots = [agent_to_snapshot(a, ctx.last_contents.get(a.agent_id, "")) for a in ctx.agents]
    ctx.viz.print_frame(snapshots, ctx.round_num, ctx.max_rounds, status)


# ---------------------------------------------------------------------------
# Main orchestration with visualisation
# ---------------------------------------------------------------------------


def run_demo(
    topic: str,
    provider: BaseProvider,
    fast: bool = False,
) -> None:
    """Run the full demo with live terminal visualisation."""
    config = build_config()
    helix = config.helix_config.to_geometry()
    viz = HelixVisualizer(helix)

    if not fast:
        print_intro()

    # --- Setup ---
    factory = AgentFactory(provider, helix_config=config.helix_config)
    hub = CentralPost(max_agents=config.max_agents)
    spoke_mgr = SpokeManager(hub=hub)
    builder = CollaborativeContextBuilder()
    all_results: list[LLMResult] = []

    # Create team
    composition = config.team_composition
    agents: list[LLMAgent] = []
    for i, (agent_type, kwargs) in enumerate(composition):
        spawn_time = i / max(len(composition), 1)
        agent = factory.create_agent(
            agent_type=agent_type,
            spawn_time=spawn_time,
            **kwargs,
        )
        spoke_mgr.create_spoke(agent.agent_id, agent=agent)
        agents.append(agent)

    # Initial frame
    snapshots = [agent_to_snapshot(a) for a in agents]
    viz.print_frame(snapshots, 0, config.max_rounds, "Initialising agents...")
    _pause(1.0, fast)

    # --- Processing rounds ---
    ctx = _RoundContext(
        agents=agents,
        spoke_mgr=spoke_mgr,
        builder=builder,
        viz=viz,
        topic=topic,
        round_num=0,
        max_rounds=config.max_rounds,
        last_contents={},
        prev_phase=None,
        fast=fast,
    )
    converged = False
    rounds_completed = 0

    try:
        for round_num in range(1, config.max_rounds + 1):
            current_time = round_num / max(config.max_rounds, 1)
            ctx.round_num = round_num
            round_results: list[LLMResult] = []

            for agent in agents:
                result = _process_agent(agent, current_time, ctx)
                if result:
                    round_results.append(result)

            all_results.extend(round_results)
            spoke_mgr.process_all_messages()
            rounds_completed = round_num

            # Convergence check
            if round_results:
                avg = sum(r.confidence for r in round_results) / len(round_results)
                if avg >= config.confidence_threshold:
                    _render_frame(
                        ctx,
                        f"Converged at confidence {avg:.2f} — moving to synthesis",
                    )
                    _pause(1.5, fast)
                    converged = True
                    break

    finally:
        spoke_mgr.shutdown_all()
        hub.shutdown()

    # --- Synthesis ---
    if not fast:
        print_phase_transition("synthesis")

    synthesizer = WorkflowSynthesizer(provider, config)
    synthesis = synthesizer.synthesize(all_results, topic)
    final_confidence = (
        sum(r.confidence for r in all_results) / len(all_results) if all_results else 0
    )

    print_synthesis_result(synthesis, final_confidence)

    # Summary stats
    total_tokens = sum(r.token_budget_used for r in all_results)
    print(
        f"\n  Agents: {len(agents)}  |  Rounds: {rounds_completed}  |  "
        f"Tokens: {total_tokens:,}  |  Converged: {converged}"
    )
    print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pause(seconds: float, fast: bool) -> None:
    if not fast:
        time.sleep(seconds)
    else:
        time.sleep(0.05)


def _resolve_provider(provider_name: str | None) -> BaseProvider:
    """Resolve a provider by name, defaulting to mock."""
    if provider_name is None or provider_name == "mock":
        return make_research_mock_provider()

    if provider_name == "anthropic":
        from felix_agent_sdk.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    if provider_name == "openai":
        from felix_agent_sdk.providers.openai_provider import OpenAIProvider

        return OpenAIProvider()
    if provider_name == "local":
        from felix_agent_sdk.providers.local import LocalProvider

        return LocalProvider()

    print(f"Unknown provider: {provider_name}. Using mock.")
    return make_research_mock_provider()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Felix Deep Research Demo — live helical visualisation",
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help="Research topic for the agents to investigate",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["mock", "anthropic", "openai", "local"],
        help="LLM provider (default: mock, no API key needed)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip animations for quick testing",
    )
    args = parser.parse_args()

    provider = _resolve_provider(args.provider)

    try:
        run_demo(topic=args.topic, provider=provider, fast=args.fast)
    except KeyboardInterrupt:
        print("\n\n  Demo interrupted. Thanks for watching!")
        sys.exit(0)


if __name__ == "__main__":
    main()
