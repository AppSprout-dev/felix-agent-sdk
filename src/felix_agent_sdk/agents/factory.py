"""Agent factory for helix-aware team creation.

New SDK code — consolidates Felix's scattered ``create_*`` functions
into a single factory with a clean, extensible interface.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

from felix_agent_sdk.agents.base import generate_spawn_times
from felix_agent_sdk.agents.llm_agent import LLMAgent
from felix_agent_sdk.agents.specialized import AnalysisAgent, CriticAgent, ResearchAgent
from felix_agent_sdk.core.helix import HelixConfig
from felix_agent_sdk.providers.base import BaseProvider


class AgentFactory:
    """Factory for creating helix-aware agents and teams.

    The provider is injected once at construction time; all agents
    created by this factory share the same provider instance.

    Example::

        factory = AgentFactory(provider)
        team = factory.create_specialized_team("moderate")
    """

    _agent_types: ClassVar[Dict[str, Type[LLMAgent]]] = {
        "llm": LLMAgent,
        "research": ResearchAgent,
        "analysis": AnalysisAgent,
        "critic": CriticAgent,
    }

    def __init__(
        self,
        provider: BaseProvider,
        helix_config: Optional[HelixConfig] = None,
    ) -> None:
        self._provider = provider
        self._config = helix_config or HelixConfig.default()
        self._helix = self._config.to_geometry()
        self._agent_counter: int = 0

    # ------------------------------------------------------------------
    # Single-agent creation
    # ------------------------------------------------------------------

    def create_agent(
        self,
        agent_type: str = "llm",
        agent_id: Optional[str] = None,
        spawn_time: float = 0.0,
        **kwargs: Any,
    ) -> LLMAgent:
        """Create a single agent at a specific helix position.

        Args:
            agent_type: Registered type name (``"llm"``, ``"research"``, etc.).
            agent_id: Optional custom identifier.  Auto-generated when ``None``.
            spawn_time: Position on the spawn timeline (0.0 – 1.0).
            **kwargs: Forwarded to the agent constructor (e.g. ``research_domain``).

        Raises:
            ValueError: If *agent_type* is not registered.
        """
        cls = self._agent_types.get(agent_type)
        if cls is None:
            available = ", ".join(sorted(self._agent_types))
            raise ValueError(f"Unknown agent type {agent_type!r}. Available: {available}")

        aid = agent_id or self._generate_id(agent_type)
        return cls(
            agent_id=aid,
            provider=self._provider,
            helix=self._helix,
            spawn_time=spawn_time,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Team creation
    # ------------------------------------------------------------------

    def create_team(
        self,
        team_size: int = 3,
        agent_type: str = "llm",
        **kwargs: Any,
    ) -> List[LLMAgent]:
        """Create a homogeneous team evenly distributed along the helix.

        Spawn times are spaced as ``[0/n, 1/n, 2/n, …, (n-1)/n]``.
        """
        agents: List[LLMAgent] = []
        for i in range(team_size):
            spawn_time = i / max(team_size, 1)
            agents.append(
                self.create_agent(
                    agent_type=agent_type,
                    spawn_time=spawn_time,
                    **kwargs,
                )
            )
        return agents

    def create_specialized_team(
        self,
        complexity: str = "moderate",
        seed: Optional[int] = None,
    ) -> List[LLMAgent]:
        """Create a balanced team of specialized agents.

        Complexity levels:
        - ``"simple"``:   1 Research + 1 Analysis
        - ``"moderate"``: 1 Research + 1 Analysis + 1 Critic
        - ``"complex"``:  2 Research + 1 Analysis + 1 Critic + 1 LLM (synthesiser)

        Args:
            complexity: Task complexity level.
            seed: Optional random seed for reproducible spawn times.

        Raises:
            ValueError: If *complexity* is not recognised.
        """
        compositions: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {
            "simple": [
                ("research", {}),
                ("analysis", {}),
            ],
            "moderate": [
                ("research", {}),
                ("analysis", {}),
                ("critic", {}),
            ],
            "complex": [
                ("research", {"research_domain": "general"}),
                ("research", {"research_domain": "technical"}),
                ("analysis", {}),
                ("critic", {}),
                ("llm", {}),
            ],
        }

        spec = compositions.get(complexity)
        if spec is None:
            available = ", ".join(sorted(compositions))
            raise ValueError(f"Unknown complexity {complexity!r}. Available: {available}")

        spawn_times = generate_spawn_times(len(spec), seed=seed)
        spawn_times.sort()

        agents: List[LLMAgent] = []
        for (atype, extra_kwargs), st in zip(spec, spawn_times):
            agents.append(
                self.create_agent(
                    agent_type=atype,
                    spawn_time=st,
                    **extra_kwargs,
                )
            )
        return agents

    # ------------------------------------------------------------------
    # Type registration
    # ------------------------------------------------------------------

    @classmethod
    def register_agent_type(cls, name: str, agent_class: Type[LLMAgent]) -> None:
        """Register a custom agent type for factory creation.

        Raises:
            TypeError: If *agent_class* does not extend :class:`LLMAgent`.
        """
        if not issubclass(agent_class, LLMAgent):
            raise TypeError(f"{agent_class.__name__} must be a subclass of LLMAgent")
        cls._agent_types[name] = agent_class

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_id(self, prefix: str = "agent") -> str:
        self._agent_counter += 1
        return f"{prefix}-{self._agent_counter:03d}"
