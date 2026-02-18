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
from felix import HelixGeometry, CentralPost, AgentFactory
from felix.providers import OpenAIProvider

# 1. Configure the helix (exploration → synthesis geometry)
helix = HelixGeometry.default()  # top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2

# 2. Connect an LLM provider
provider = OpenAIProvider(model="gpt-4o")

# 3. Stand up the coordination hub and spawn a team
central = CentralPost(helix, provider=provider)
factory = AgentFactory(helix, provider)
team = factory.spawn_team(roles=["research", "analysis", "critic"])

# 4. Register agents and run
for agent in team:
    central.register_agent(agent)

result = central.run(task="Evaluate the current state of multi-agent AI frameworks")
print(result.synthesis)
```

That's it. Felix handles the agent spawning positions, position-aware prompting (research agents get higher temperature, critic agents get lower), hub-spoke message routing, and final synthesis.

---

## Why Felix?

### The Problem with Existing Frameworks

Most multi-agent frameworks model collaboration as graph traversal: define nodes, wire edges, pass state. This works for deterministic pipelines, but it doesn't model how expert teams actually converge on complex problems — starting with broad exploration, iteratively refining, and progressively narrowing toward synthesis.

### What Felix Does Differently

**Helical Geometry as a First-Class Primitive.** Agent progression isn't a flat graph — it's a 3D helix where position determines behavior. Agents at the top (wide radius) explore broadly with high creativity. As they descend and the helix narrows, they focus, critique, and synthesize. This isn't a metaphor; it's implemented mathematically with parametric equations that control temperature, token budgets, and prompting strategies based on position.

**Hub-Spoke Communication via CentralPost.** Instead of requiring developers to wire message-passing between every agent pair (O(N²) mesh), Felix routes all communication through a central coordination hub at O(N) complexity. CentralPost manages message queuing, phase-aware routing, agent awareness, and synthesis. It's both simpler and more efficient.

**Dynamic Spawning with Confidence Monitoring.** Felix agents don't just execute predefined tasks — the system monitors confidence levels and content gaps in real time, spawning additional agents when the team needs reinforcement. If research agents produce low-confidence results, the system can automatically spawn additional researchers or critics to address the gaps.

**Provider-Agnostic by Design.** Swap LLM providers with a single configuration change. Use Claude for research agents and GPT-4 for analysis agents. Run locally with LM Studio during development and switch to Bedrock in production. The provider abstraction is clean and extensible.

### Framework Comparison

| Capability                        | Felix Agent SDK       | LangGraph             | Claude Agent SDK      |
|-----------------------------------|-----------------------|-----------------------|-----------------------|
| **Agent topology**                | Helical progression   | Directed graph (DAG)  | Single agent loop     |
| **Multi-agent native**            | ✅ Core architecture  | ✅ Via nodes/edges    | ❌ Single agent       |
| **Communication model**           | Hub-spoke O(N)        | Shared state          | N/A                   |
| **Dynamic team composition**      | ✅ Confidence-based   | ❌ Static graph       | ❌ Fixed              |
| **Position-aware behavior**       | ✅ Geometry-driven    | ❌ Manual             | ❌ N/A                |
| **Provider-agnostic**             | ✅ Any LLM            | ✅ Any LLM            | ❌ Claude only        |
| **Memory & knowledge persistence**| ✅ Built-in           | ✅ Checkpointing      | ⚠️ Via tools          |
| **Progressive convergence**       | ✅ Architectural      | ❌ Manual loops       | ❌ N/A                |
| **Scientific workflow alignment** | ✅ By design          | ⚠️ Possible           | ❌ Not designed for   |

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
pip install felix-agent-sdk[bedrock]     # AWS Bedrock
pip install felix-agent-sdk[vertex]      # Google Vertex AI
```

### Everything

```bash
pip install felix-agent-sdk[all]
```

### Requirements

Python 3.10 or higher is required. The core SDK has minimal dependencies: `pydantic`, `httpx`, `numpy`, and `pyyaml`.

---

## Core Concepts

### The Helix

The helix is Felix's central abstraction. It's a 3D parametric curve defined by four parameters:

```python
from felix import HelixGeometry

helix = HelixGeometry(
    top_radius=3.0,     # Breadth of exploration (wider = more diverse)
    bottom_radius=0.5,  # Precision of synthesis (narrower = more focused)
    height=8.0,         # Total progression depth
    turns=2             # Number of complete spirals
)
```

Each agent is positioned on this helix at parameter `t ∈ [0, 1]`, where `t=0` is the top (wide exploration) and `t=1` is the bottom (narrow synthesis). The agent's position determines its behavior: temperature, token budget, and prompting strategy are all derived from the geometry.

**Default presets** are available for common scenarios:

```python
helix = HelixGeometry.default()              # Balanced general-purpose
helix = HelixGeometry.research_heavy()       # Wide top, many turns
helix = HelixGeometry.fast_convergence()     # Steep descent, quick synthesis
```

### Agents

Felix provides three specialized agent types, each designed for a phase of the convergence process:

**ResearchAgent** — Operates near the top of the helix. High temperature, broad exploration, gathers diverse perspectives and information. Spawned early in the workflow.

**AnalysisAgent** — Operates in the middle band. Moderate temperature, synthesizes and evaluates research findings, identifies patterns and gaps. Spawned after initial research completes.

**CriticAgent** — Operates in the lower band. Low temperature, challenges assumptions, validates reasoning, ensures rigor. Spawned once analysis has produced substantive claims.

You can also define custom agent types:

```python
from felix.agents import Agent, AgentRole

class FactCheckAgent(Agent):
    role = AgentRole.CUSTOM
    temperature_range = (0.1, 0.4)   # Low creativity, high precision
    max_tokens = 300
    spawn_time_range = (0.5, 0.9)    # Mid-to-late in workflow

    def build_prompt(self, task, context):
        return f"Verify the factual accuracy of: {context.latest_claims}"
```

### CentralPost (Hub-Spoke Communication)

CentralPost is the coordination hub through which all agent communication flows. It handles message routing, agent registration, phase tracking, and final synthesis.

```python
from felix import CentralPost, HelixGeometry
from felix.providers import AnthropicProvider

helix = HelixGeometry.default()
provider = AnthropicProvider(model="claude-sonnet-4-5")

central = CentralPost(
    helix,
    provider=provider,
    max_agents=15,
    enable_memory=True,
    enable_metrics=True,
)

# Register agents
central.register_agent(research_agent)
central.register_agent(analysis_agent)

# Query team awareness
status = central.query_team_awareness("convergence_status")

# Run to synthesis
result = central.run(task="Analyze competitive landscape of AI agent frameworks")
```

### Memory

Felix includes built-in memory systems that persist knowledge across workflow runs:

**KnowledgeStore** — Long-term storage for insights, findings, and synthesized knowledge. Entries are tagged with confidence levels and knowledge types. Supports retrieval by relevance.

**TaskMemory** — Pattern recognition for recurring tasks. Stores successful strategies and outcomes so the system can improve over time.

**ContextCompressor** — Manages growing context windows by intelligently compressing earlier conversation history while preserving critical information.

```python
from felix.memory import KnowledgeStore

store = KnowledgeStore("project_knowledge.db")
store.add(content="Felix SDK shows 20% improvement in workload distribution",
          confidence=0.85, knowledge_type="benchmark_result")

relevant = store.query("workload distribution performance", top_k=5)
```

### Dynamic Spawning

Felix can adaptively grow and modify its agent team based on real-time confidence monitoring:

```python
from felix.spawning import DynamicSpawning, ConfidenceMonitor

monitor = ConfidenceMonitor(
    threshold=0.8,            # Spawn when confidence drops below this
    volatility_threshold=0.15, # Spawn stabilizers on high variance
    window_minutes=5.0,        # Rolling analysis window
)

spawner = DynamicSpawning(
    factory=agent_factory,
    monitor=monitor,
    max_agents=15,
)

# The spawner hooks into CentralPost and reacts automatically
central.attach_spawner(spawner)
```

---

## Workflow Patterns

### Research Pipeline

The most common pattern — a team investigates a topic, analyzes findings, and produces a synthesized report:

```python
from felix.workflows.templates import ResearchWorkflow

workflow = ResearchWorkflow(
    provider=provider,
    helix=HelixGeometry.research_heavy(),
    num_researchers=3,
    num_analysts=2,
    num_critics=1,
)

result = workflow.run("What are the implications of quantum computing for cryptography?")
print(result.synthesis)
print(result.confidence)
print(result.sources)
```

### Critique-and-Refine Loop

An iterative pattern where critic agents challenge analysis agents, who refine their work until confidence thresholds are met:

```python
from felix.workflows.templates import ReviewWorkflow

workflow = ReviewWorkflow(
    provider=provider,
    max_iterations=5,
    confidence_target=0.9,
)

result = workflow.run("Review this technical specification for logical consistency",
                      context=spec_document)
```

### Custom Workflow

For full control, use the lower-level primitives directly:

```python
from felix import HelixGeometry, CentralPost, AgentFactory
from felix.providers import AnthropicProvider
from felix.spawning import DynamicSpawning, ConfidenceMonitor

# Setup
helix = HelixGeometry(top_radius=4.0, bottom_radius=0.3, height=10.0, turns=3)
provider = AnthropicProvider(model="claude-sonnet-4-5")
central = CentralPost(helix, provider=provider, enable_memory=True)
factory = AgentFactory(helix, provider)

# Spawn initial team
researchers = [factory.create_research_agent(domain=d)
               for d in ["technical", "market", "regulatory"]]
analyst = factory.create_analysis_agent()
critic = factory.create_critic_agent()

for agent in [*researchers, analyst, critic]:
    central.register_agent(agent)

# Attach dynamic spawning
spawner = DynamicSpawning(factory=factory,
                          monitor=ConfidenceMonitor(threshold=0.8),
                          max_agents=12)
central.attach_spawner(spawner)

# Run with streaming callbacks
def on_progress(agent_id, phase, message):
    print(f"[{phase}] {agent_id}: {message[:80]}...")

result = central.run(
    task="Comprehensive analysis of the AI agent SDK market landscape",
    on_progress=on_progress,
)

print(f"\nSynthesis (confidence: {result.confidence:.2f}):")
print(result.synthesis)
```

---

## Provider Configuration

Felix supports multiple LLM providers through a clean abstraction layer. Configure via code or environment variables:

### Code Configuration

```python
from felix.providers import AnthropicProvider, OpenAIProvider, LocalProvider

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
from felix.providers import auto_detect_provider

provider = auto_detect_provider()  # Reads from environment
```

### Mixed Providers

Use different providers for different agent roles:

```python
from felix.providers import OpenAIProvider, AnthropicProvider

central = CentralPost(helix, provider=OpenAIProvider(model="gpt-4o"))

# Override for specific agents
research_agent = factory.create_research_agent(
    provider_override=AnthropicProvider(model="claude-sonnet-4-5")
)
```

---

## Streaming

Felix supports real-time streaming of agent outputs:

```python
from felix.streaming import StreamHandler

handler = StreamHandler()

@handler.on("agent_started")
def on_start(event):
    print(f"Agent {event.agent_id} began {event.phase}")

@handler.on("token")
def on_token(event):
    print(event.text, end="", flush=True)

@handler.on("synthesis_complete")
def on_done(event):
    print(f"\n\nFinal confidence: {event.confidence}")

result = central.run(task="...", stream_handler=handler)
```

---

## Benchmarks

Felix includes built-in benchmarking to validate its three core hypotheses:

| Hypothesis | Claim | Validated Improvement |
|------------|-------|----------------------|
| **H1** | Helical progression enhances workload distribution | 20% gain vs. linear |
| **H2** | Hub-spoke communication optimizes resource allocation | 15% gain vs. mesh |
| **H3** | Memory compression reduces latency with attention focus | 25% gain |

Run benchmarks:

```bash
python -m felix.benchmarks.run_all
```

---

## Roadmap

**Phase 1 — Core SDK (Current):** Pip-installable package with provider abstraction, core primitives, and documentation. Focus on making Felix accessible to external developers.

**Phase 2 — Developer Experience:** CLI tooling (`felix init`, `felix run`), helix visualization dashboard, structured logging with observability integrations, and expanded examples covering common use cases.

**Phase 3 — Community & Ecosystem:** MCP server integration, vector database connectors, LangSmith/Phoenix observability adapters, and community contribution framework.

---

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setting up a development environment, running tests, and submitting pull requests.

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