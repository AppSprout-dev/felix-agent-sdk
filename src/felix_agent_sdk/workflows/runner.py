"""Felix workflow orchestration runner.

Wires agents, communication hub, and memory into a complete multi-agent
workflow. This is the top-level integration point of the SDK.

Core orchestration loop ported from CalebisGross/felix
``src/workflows/felix_workflow.py``. Refactored to use discrete rounds,
the SDK's provider abstraction, and the pluggable backend stack.
"""

from __future__ import annotations

import logging
import time

from typing import Optional

from felix_agent_sdk.agents.base import AgentState
from felix_agent_sdk.agents.factory import AgentFactory
from felix_agent_sdk.agents.llm_agent import LLMAgent, LLMResult, LLMTask
from felix_agent_sdk.communication.central_post import CentralPost
from felix_agent_sdk.communication.messages import MessageType
from felix_agent_sdk.communication.spoke import SpokeManager
from felix_agent_sdk.events.bus import EventBus
from felix_agent_sdk.events.mixins import EventEmitterMixin
from felix_agent_sdk.events.types import EventType
from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.spawning.spawner import DynamicSpawner
from felix_agent_sdk.workflows.config import WorkflowConfig, WorkflowResult
from felix_agent_sdk.workflows.context_builder import CollaborativeContextBuilder
from felix_agent_sdk.workflows.synthesizer import WorkflowSynthesizer

logger = logging.getLogger(__name__)


class FelixWorkflow(EventEmitterMixin):
    """Multi-agent workflow runner using helical agent progression.

    Orchestrates agent team creation, discrete processing rounds with
    helix-driven temperature/prompting, message routing through the
    CentralPost hub, and final synthesis.

    Args:
        config: Workflow configuration (team composition, thresholds, …).
        provider: LLM provider shared by all agents.
        event_bus: Optional event bus for observability.
    """

    def __init__(
        self,
        config: WorkflowConfig,
        provider: BaseProvider,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._factory = AgentFactory(provider, helix_config=config.helix_config)
        self.set_event_bus(event_bus)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, task_description: str, context: str = "") -> WorkflowResult:
        """Execute the workflow end-to-end.

        1. Setup — hub, spokes, agents.
        2. Processing rounds — advance time, process tasks, route messages.
        3. Synthesis — combine results.
        4. Cleanup — shutdown hub and spokes.

        Args:
            task_description: The task for agents to work on.
            context: Optional initial context / background.

        Returns:
            :class:`WorkflowResult` with synthesised output and metadata.
        """
        start = time.monotonic()

        # --- Setup ---
        hub = CentralPost(max_agents=self._config.max_agents, event_bus=self._event_bus)
        spoke_mgr = SpokeManager(hub=hub)
        agents = self._create_team(spoke_mgr)
        builder = CollaborativeContextBuilder()
        all_results: list[LLMResult] = []

        spawner: Optional[DynamicSpawner] = None
        if self._config.enable_dynamic_spawning:
            spawner = DynamicSpawner(
                factory=self._factory,
                spoke_mgr=spoke_mgr,
                max_spawned=self._config.max_dynamic_agents,
                event_bus=getattr(self, "_event_bus", None),
            )

        self.emit_event(
            EventType.WORKFLOW_STARTED,
            {
                "task": task_description,
                "max_rounds": self._config.max_rounds,
                "agents_count": len(agents),
                "team_composition": [(a.agent_type, a.agent_id) for a in agents],
            },
        )

        try:
            # --- Processing rounds ---
            rounds_completed = 0
            converged = False
            time_step = 1.0 / max(self._config.max_rounds, 1)

            for round_num in range(self._config.max_rounds):
                current_time = (round_num + 1) * time_step
                rounds_completed = round_num + 1

                self.emit_event(
                    EventType.WORKFLOW_ROUND_STARTED,
                    {"round": rounds_completed, "current_time": round(current_time, 4)},
                )

                round_results = self._run_round(
                    agents=agents,
                    spoke_mgr=spoke_mgr,
                    builder=builder,
                    task_description=task_description,
                    context=context,
                    current_time=current_time,
                )
                all_results.extend(round_results)

                avg_confidence = 0.0
                if round_results:
                    avg_confidence = sum(r.confidence for r in round_results) / len(round_results)

                self.emit_event(
                    EventType.WORKFLOW_ROUND_COMPLETED,
                    {
                        "round": rounds_completed,
                        "results_count": len(round_results),
                        "avg_confidence": round(avg_confidence, 4),
                    },
                )

                # Dynamic spawning
                if spawner is not None:
                    new_agents = spawner.check_and_spawn(
                        agents, round_results, current_time
                    )
                    agents.extend(new_agents)

                # Convergence check
                if round_results and avg_confidence >= self._config.confidence_threshold:
                    logger.info(
                        "Workflow converged at round %d (confidence=%.3f)",
                        rounds_completed,
                        avg_confidence,
                    )
                    self.emit_event(
                        EventType.WORKFLOW_CONVERGED,
                        {"round": rounds_completed, "confidence": round(avg_confidence, 4)},
                    )
                    converged = True
                    break

            # --- Synthesis ---
            self.emit_event(EventType.WORKFLOW_SYNTHESIS_STARTED, {})
            synthesizer = WorkflowSynthesizer(self._provider, self._config)
            synthesis = synthesizer.synthesize(all_results, task_description)

            # --- Result ---
            final_confidence = 0.0
            if all_results:
                final_confidence = sum(r.confidence for r in all_results) / len(all_results)

            elapsed = time.monotonic() - start
            total_tokens = sum(r.token_budget_used for r in all_results)

            result = WorkflowResult(
                synthesis=synthesis,
                agent_results=all_results,
                total_rounds=rounds_completed,
                final_confidence=final_confidence,
                metadata={
                    "elapsed_seconds": round(elapsed, 3),
                    "total_tokens": total_tokens,
                    "converged": converged,
                    "agents_count": len(agents),
                    "team_composition": [(a.agent_type, a.agent_id) for a in agents],
                },
            )

            self.emit_event(
                EventType.WORKFLOW_COMPLETED,
                {
                    "rounds": rounds_completed,
                    "converged": converged,
                    "final_confidence": round(final_confidence, 4),
                    "total_tokens": total_tokens,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )

            return result
        finally:
            spoke_mgr.shutdown_all()
            hub.shutdown()

    # ------------------------------------------------------------------
    # Internal — team setup
    # ------------------------------------------------------------------

    def _create_team(self, spoke_mgr: SpokeManager) -> list[LLMAgent]:
        """Spawn agents from config and connect them to the hub."""
        composition = self._config.team_composition
        n = len(composition)
        agents: list[LLMAgent] = []

        for i, (agent_type, kwargs) in enumerate(composition):
            spawn_time = i / max(n, 1)
            agent = self._factory.create_agent(
                agent_type=agent_type,
                spawn_time=spawn_time,
                **kwargs,
            )
            # Propagate event bus to each agent
            if self._event_bus is not None:
                agent.set_event_bus(self._event_bus)
            spoke_mgr.create_spoke(agent.agent_id, agent=agent)
            agents.append(agent)

        return agents

    # ------------------------------------------------------------------
    # Internal — processing round
    # ------------------------------------------------------------------

    def _run_round(
        self,
        agents: list[LLMAgent],
        spoke_mgr: SpokeManager,
        builder: CollaborativeContextBuilder,
        task_description: str,
        context: str,
        current_time: float,
    ) -> list[LLMResult]:
        """Execute one processing round across all agents."""
        round_results: list[LLMResult] = []

        for agent in agents:
            # Spawn if eligible and still waiting
            if agent.state == AgentState.WAITING and agent.can_spawn(current_time):
                agent.spawn(current_time)

            if agent.state != AgentState.ACTIVE:
                continue

            # Advance position
            agent.update_position(current_time)

            # Check helical checkpoint gating
            if not agent.should_process_at_checkpoint():
                continue

            # Build task with collaborative context
            task = LLMTask(
                task_id=f"{agent.agent_id}-r{current_time:.2f}",
                description=task_description,
                context=context,
                context_history=builder.get_context_history(),
            )

            result = agent.process_task(task)
            agent.mark_checkpoint_processed()

            # Feed result back to collaborative context
            builder.add_from_result(result)

            # Send status message through hub
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

        # Route messages through the hub
        spoke_mgr.process_all_messages()

        return round_results

    # ------------------------------------------------------------------
    # Convenience function
    # ------------------------------------------------------------------


def run_felix_workflow(
    config: WorkflowConfig,
    provider: BaseProvider,
    task_description: str,
    context: str = "",
    event_bus: Optional[EventBus] = None,
) -> WorkflowResult:
    """Run a Felix workflow in one call.

    Thin wrapper around :class:`FelixWorkflow` for simple use cases.
    """
    workflow = FelixWorkflow(config, provider, event_bus=event_bus)
    return workflow.run(task_description, context=context)
