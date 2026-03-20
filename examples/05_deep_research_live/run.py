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
import os
import sys
import time

# Enable UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    # Enable ANSI escape processing on Windows 10+
    os.system("")

# Ensure the examples directory and src are importable
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "examples"))

from felix_agent_sdk.agents.base import AgentState
from felix_agent_sdk.agents.factory import AgentFactory
from felix_agent_sdk.agents.llm_agent import LLMAgent, LLMResult, LLMTask
from felix_agent_sdk.communication.central_post import CentralPost
from felix_agent_sdk.communication.messages import MessageType
from felix_agent_sdk.communication.spoke import SpokeManager
from felix_agent_sdk.core.helix import HelixConfig
from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.workflows.config import SynthesisStrategy, WorkflowConfig
from felix_agent_sdk.workflows.context_builder import CollaborativeContextBuilder
from felix_agent_sdk.workflows.synthesizer import WorkflowSynthesizer

from _mock_research import make_research_mock_provider
from helix_visualizer import (
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


def agent_to_snapshot(
    agent: LLMAgent, last_content: str = ""
) -> AgentSnapshot:
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
    last_contents: dict[str, str] = {}
    prev_phase: str | None = None
    converged = False

    try:
        for round_num in range(1, config.max_rounds + 1):
            current_time = round_num / max(config.max_rounds, 1)
            round_results: list[LLMResult] = []

            for agent in agents:
                # Spawn
                if agent.state == AgentState.WAITING and agent.can_spawn(current_time):
                    agent.spawn(current_time)

                if agent.state != AgentState.ACTIVE:
                    continue

                # Advance position
                agent.update_position(current_time)

                # Phase transition animation
                phase = agent.position.phase
                if prev_phase and phase != prev_phase and not fast:
                    print_phase_transition(phase)
                prev_phase = phase

                # Checkpoint gating
                if not agent.should_process_at_checkpoint():
                    continue

                # Show "thinking" frame
                snapshots = [
                    agent_to_snapshot(a, last_contents.get(a.agent_id, ""))
                    for a in agents
                ]
                viz.print_frame(
                    snapshots,
                    round_num,
                    config.max_rounds,
                    f"{agent.agent_id} ({agent.agent_type}) processing...",
                )
                _pause(0.6, fast)

                # Process task
                task = LLMTask(
                    task_id=f"{agent.agent_id}-r{round_num}",
                    description=topic,
                    context="",
                    context_history=builder.get_context_history(),
                )
                result = agent.process_task(task)
                agent.mark_checkpoint_processed()
                builder.add_from_result(result)

                # Route message
                spoke = spoke_mgr.get_spoke(agent.agent_id)
                if spoke:
                    spoke.send_message(
                        message_type=MessageType.STATUS_UPDATE,
                        content={
                            "confidence": result.confidence,
                            "phase": result.position_info.get("phase", ""),
                            "progress": result.position_info.get("progress", 0.0),
                        },
                    )

                round_results.append(result)
                last_contents[agent.agent_id] = result.content

                # Render updated frame
                snapshots = [
                    agent_to_snapshot(a, last_contents.get(a.agent_id, ""))
                    for a in agents
                ]
                viz.print_frame(
                    snapshots,
                    round_num,
                    config.max_rounds,
                    f"{agent.agent_id} done — confidence {result.confidence:.2f}",
                )
                _pause(0.8, fast)

            all_results.extend(round_results)
            spoke_mgr.process_all_messages()

            # Convergence check
            if round_results:
                avg = sum(r.confidence for r in round_results) / len(round_results)
                if avg >= config.confidence_threshold:
                    snapshots = [
                        agent_to_snapshot(a, last_contents.get(a.agent_id, ""))
                        for a in agents
                    ]
                    viz.print_frame(
                        snapshots,
                        round_num,
                        config.max_rounds,
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
    print(f"\n  Agents: {len(agents)}  |  Rounds: {round_num}  |  "
          f"Tokens: {total_tokens:,}  |  Converged: {converged}")
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
        print(f"\n\n  Demo interrupted. Thanks for watching!")
        sys.exit(0)


if __name__ == "__main__":
    main()
