# Changelog

All notable changes to Felix Agent SDK will be documented in this file.

## [0.1.0] — 2026-03-19

Initial public release of the Felix Agent SDK.

### Core Geometry
- `HelixGeometry` — 3D parametric helix with `get_position(t)`, `get_radius(z)`, `get_angle_at_t(t)`, `get_tangent_vector(t)`, `approximate_arc_length()`
- `HelixConfig` — named presets: `default()`, `research_heavy()`, `fast_convergence()`
- `HelixPosition` — phase-aware wrapper with exploration/analysis/synthesis boundaries

### Agents
- `Agent` base class with helix-aware positioning, temperature, and lifecycle
- `LLMAgent` with provider integration, position-aware prompting, meta-cognitive evaluation
- `ResearchAgent`, `AnalysisAgent`, `CriticAgent` specialized agents
- `AgentFactory` with `create_agent()`, `create_specialized_team()`, custom type registration

### Communication
- `CentralPost` hub with O(N) message routing, phase-aware delivery, performance tracking
- `AgentRegistry` with capacity management, role indexing, performance history
- `Spoke` + `SpokeManager` with per-type handler registration, lazy async support
- `Message` and `MessageType` enums for structured inter-agent communication

### Memory
- `KnowledgeStore` with confidence-tagged entries, domain filtering, relationship graph
- `TaskMemory` with pattern recognition, strategy recommendations, execution history
- `ContextCompressor` with hierarchical summarization strategies
- Pluggable backend interface with `SQLiteBackend` as default

### Providers
- `BaseProvider` abstract interface: `complete()`, `stream()`, `count_tokens()`
- `AnthropicProvider` — Claude models
- `OpenAIProvider` — GPT models and OpenAI-compatible APIs
- `LocalProvider` — LM Studio, Ollama via OpenAI-compatible endpoint
- `auto_detect_provider()` — environment-based provider selection

### Workflows
- `FelixWorkflow` orchestrator: team spawn → rounds → convergence check → synthesis
- `WorkflowConfig` with helix config, team composition, confidence threshold, synthesis strategy
- `CollaborativeContextBuilder` with relevance scoring and deduplication
- `WorkflowSynthesizer` with BEST_RESULT, COMPRESSED_MERGE, ROUND_ROBIN strategies
- Pre-built templates: `research_config()`, `analysis_config()`, `review_config()`

### Tokens
- `TokenBudget` with position-aware allocation (exploration vs synthesis split)

### Infrastructure
- Pip-installable package with optional provider dependencies
- 659 tests (unit + integration)
- Python 3.10+ support
- MIT license
