# Felix Agent SDK

**Helical multi-agent orchestration for convergent collaboration.**

Felix is a Python SDK for building multi-agent AI systems that progressively converge from broad exploration to focused synthesis. Unlike graph-based frameworks that route tasks through static DAGs, Felix models agent collaboration as movement along a helical geometry — agents start wide (exploring diverse perspectives) and spiral inward toward consensus, with a central coordination hub managing communication at O(N) efficiency.

```
pip install felix-agent-sdk
```

---

## Quickstart

### Your First Felix Workflow (< 10 lines)

```python
from felix_agent_sdk import run_felix_workflow, WorkflowConfig
from felix_agent_sdk.providers import OpenAIProvider

provider = OpenAIProvider(model="gpt-4o")
config = WorkflowConfig()  # 1 Research + 1 Analysis + 1 Critic

result = run_felix_workflow(config, provider, "Evaluate the state of multi-agent AI frameworks")
print(result.synthesis)
```

That's it. Felix handles agent spawning positions, position-aware prompting (research agents get higher temperature, critic agents get lower), hub-spoke message routing, and final synthesis.

---

## Why Felix?

### The Problem with Existing Frameworks

Most multi-agent frameworks model collaboration as graph traversal: define nodes, wire edges, pass state. This works for deterministic pipelines, but it doesn't model how expert teams actually converge on complex problems — starting with broad exploration, iteratively refining, and progressively narrowing toward synthesis.

### What Felix Does Differently

**Helical Geometry as a First-Class Primitive.** Agent progression isn't a flat graph — it's a 3D helix where position determines behavior. Agents at the top (wide radius) explore broadly with high creativity. As they descend and the helix narrows, they focus, critique, and synthesize. This isn't a metaphor; it's implemented mathematically with parametric equations that control temperature, token budgets, and prompting strategies based on position.

**Hub-Spoke Communication via CentralPost.** Instead of requiring developers to wire message-passing between every agent pair (O(N²) mesh), Felix routes all communication through a central coordination hub at O(N) complexity. CentralPost manages message queuing, phase-aware routing, agent awareness, and synthesis.

**Dynamic Spawning with Confidence Monitoring.** Felix agents don't just execute predefined tasks — the system monitors confidence levels and content gaps in real time, spawning additional agents when the team needs reinforcement. *(Dynamic spawning module coming in a future release.)*

**Provider-Agnostic by Design.** Swap LLM providers with a single configuration change. Use Claude for research agents and GPT-4 for analysis agents. Run locally with LM Studio during development and switch to production APIs for deployment. The provider abstraction is clean and extensible.

### Framework Comparison

| Capability                        | Felix Agent SDK       | LangGraph             | Claude Agent SDK      |
|-----------------------------------|-----------------------|-----------------------|-----------------------|
| **Agent topology**                | Helical progression   | Directed graph (DAG)  | Single agent loop     |
| **Multi-agent native**            | Core architecture     | Via nodes/edges       | Single agent          |
| **Communication model**           | Hub-spoke O(N)        | Shared state          | N/A                   |
| **Dynamic team composition**      | Confidence-based      | Static graph          | Fixed                 |
| **Position-aware behavior**       | Geometry-driven       | Manual                | N/A                   |
| **Provider-agnostic**             | Any LLM               | Any LLM               | Claude only           |
| **Memory & knowledge persistence**| Built-in              | Checkpointing         | Via tools             |
| **Progressive convergence**       | Architectural         | Manual loops          | N/A                   |

---

## Installation

### Basic (core + no provider dependencies)

```bash
pip install felix-agent-sdk
```

### With a specific LLM provider

```bash
pip install felix-agent-sdk[anthropic]   # Claude
pip install felix-agent-sdk[openai]      # OpenAI / OpenAI-compatible
pip install felix-agent-sdk[local]       # LM Studio, Ollama (uses OpenAI-compat API)
```

### Everything

```bash
pip install felix-agent-sdk[all]
```

### Requirements

Python 3.10 or higher. Core dependencies: `pydantic`, `httpx`, `numpy`, `pyyaml`.

---

## Core Concepts

### The Helix

The helix is Felix's central abstraction. It's a 3D parametric curve defined by four parameters:

```python
from felix_agent_sdk.core.helix import HelixGeometry

helix = HelixGeometry(
    top_radius=3.0,     # Breadth of exploration (wider = more diverse)
    bottom_radius=0.5,  # Precision of synthesis (narrower = more focused)
    height=8.0,         # Total progression depth
    turns=2             # Number of complete spirals
)
```

Each agent is positioned on this helix at parameter `t in [0, 1]`, where `t=0` is the top (wide exploration) and `t=1` is the bottom (narrow synthesis). The agent's position determines its behavior: temperature, token budget, and prompting strategy are all derived from the geometry.

**Named presets** are available for common scenarios:

```python
from felix_agent_sdk.core.helix import HelixConfig

config = HelixConfig.default()              # Balanced general-purpose
config = HelixConfig.research_heavy()       # Wide top, many turns
config = HelixConfig.fast_convergence()     # Steep descent, quick synthesis
```

### Agents

Felix provides three specialized agent types, each designed for a phase of the convergence process:

**ResearchAgent** — Operates near the top of the helix. High temperature, broad exploration, gathers diverse perspectives. Spawned early in the workflow.

**AnalysisAgent** — Operates in the middle band. Moderate temperature, evaluates research findings, identifies patterns and gaps.

**CriticAgent** — Operates in the lower band. Low temperature, challenges assumptions, validates reasoning, ensures rigor.

```python
from felix_agent_sdk import AgentFactory
from felix_agent_sdk.providers import AnthropicProvider

provider = AnthropicProvider(model="claude-sonnet-4-5")
factory = AgentFactory(provider)

team = factory.create_specialized_team("moderate")
# Returns: [ResearchAgent, AnalysisAgent, CriticAgent]
```

### CentralPost (Hub-Spoke Communication)

CentralPost is the coordination hub through which all agent communication flows. It handles message routing, agent registration, and phase tracking.

```python
from felix_agent_sdk import CentralPost, Spoke
from felix_agent_sdk.communication import SpokeManager

hub = CentralPost(max_agents=10)
spoke_mgr = SpokeManager(hub=hub)

# Create spokes for each agent
for agent in team:
    spoke_mgr.create_spoke(agent.agent_id, agent=agent)

# Route messages
spoke_mgr.process_all_messages()
```

### Memory

Felix includes built-in memory systems that persist knowledge across workflow runs:

**KnowledgeStore** — Long-term storage for insights and findings, tagged with confidence levels and knowledge types.

**TaskMemory** — Pattern recognition for recurring tasks. Tracks successful strategies and outcomes.

**ContextCompressor** — Manages growing context windows by intelligently compressing earlier conversation history.

```python
from felix_agent_sdk.memory import KnowledgeStore, KnowledgeType, ConfidenceLevel

store = KnowledgeStore()  # In-memory default; pass SQLiteBackend for persistence
kid = store.add_entry(
    knowledge_type=KnowledgeType.AGENT_INSIGHT,
    content={"text": "Solar capacity grew 40% year-over-year"},
    confidence_level=ConfidenceLevel.HIGH,
    source_agent="research-001",
    domain="energy",
)
entry = store.get_entry_by_id(kid)
```

---

## Workflow Patterns

### Research Pipeline

A team investigates a topic, analyzes findings, and produces a synthesized report:

```python
from felix_agent_sdk import FelixWorkflow
from felix_agent_sdk.workflows.templates import research_config

config = research_config()  # 2 Research + 1 Analysis + 1 Critic, 4 rounds
workflow = FelixWorkflow(config, provider)
result = workflow.run("What are the implications of quantum computing for cryptography?")

print(result.synthesis)
print(f"Confidence: {result.final_confidence:.2f}")
print(f"Rounds: {result.total_rounds}")
```

### Critique-and-Refine

An iterative pattern where critic agents challenge analysis agents:

```python
from felix_agent_sdk.workflows.templates import review_config

config = review_config()  # 1 Research + 1 Analysis + 2 Critic
workflow = FelixWorkflow(config, provider)
result = workflow.run("Review this technical specification for logical consistency")
```

### Custom Workflow

For full control, configure the workflow directly:

```python
from felix_agent_sdk import FelixWorkflow, WorkflowConfig
from felix_agent_sdk.core.helix import HelixConfig
from felix_agent_sdk.workflows.config import SynthesisStrategy

config = WorkflowConfig(
    helix_config=HelixConfig(top_radius=4.0, bottom_radius=0.3, height=10.0, turns=3),
    team_composition=[
        ("research", {}),
        ("research", {}),
        ("analysis", {}),
        ("critic", {}),
    ],
    confidence_threshold=0.85,
    max_rounds=5,
    synthesis_strategy=SynthesisStrategy.COMPRESSED_MERGE,
)

workflow = FelixWorkflow(config, provider)
result = workflow.run("Comprehensive analysis of the AI agent SDK market landscape")
print(f"Synthesis (confidence: {result.final_confidence:.2f}):")
print(result.synthesis)
```

---

## Provider Configuration

Felix supports multiple LLM providers through a clean abstraction layer:

```python
from felix_agent_sdk.providers import AnthropicProvider, OpenAIProvider, LocalProvider

# Anthropic Claude
provider = AnthropicProvider(model="claude-sonnet-4-5", api_key="sk-...")

# OpenAI
provider = OpenAIProvider(model="gpt-4o", api_key="sk-...")

# Local (LM Studio, Ollama)
provider = LocalProvider(
    base_url="http://localhost:1234/v1",
    model="llama-3.1-8b",
)
```

### Environment Variables

```bash
export FELIX_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export FELIX_MODEL=claude-sonnet-4-5
```

```python
from felix_agent_sdk.providers import auto_detect_provider

provider = auto_detect_provider()  # Reads from environment
```

---

## Roadmap

**Phase 1 — Core SDK (v0.1.0):** Pip-installable package with provider abstraction, core primitives, and documentation.

**Phase 2 — Developer Experience (v0.2.0, Current):** Event system, structured logging, streaming, dynamic spawning, CLI tooling (`felix init`, `felix run`), expanded examples.

**Phase 3 — Community & Ecosystem:** MCP server integration, vector database connectors, observability adapters, community contribution framework.

---

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/AppSprout-dev/felix-agent-sdk.git
cd felix-agent-sdk
pip install -e ".[dev]"
pytest
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Built with care by the [AppSprout](https://appsprout.dev) team.
