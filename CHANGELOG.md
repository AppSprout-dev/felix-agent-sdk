# Changelog

All notable changes to Felix Agent SDK will be documented in this file.

## [0.2.0] — 2026-03-21

Phase 2: Developer Experience. Zero breaking changes to v0.1.0 API.

### Event System
- `EventBus` — synchronous pub/sub with exact, prefix (`"agent.*"`), and catch-all subscriptions
- `FelixEvent` — frozen dataclass with event_type, source, data, timestamp
- `EventType` — enum covering agent, workflow, task, message, stream, and spawn events
- `EventEmitterMixin` — opt-in mixin for event emission from any component
- Wired into `FelixWorkflow`, `LLMAgent`, and `CentralPost`

### Structured Logging
- `configure_logging()` — one-call setup for all `felix_agent_sdk.*` loggers
- `FelixLogConfig` — text or JSON format, per-subsystem level overrides
- `JSONFormatter` — structured JSON output with event metadata fields
- `EventLogBridge` — subscribes to EventBus and auto-logs events

### Streaming
- `StreamEvent` / `StreamEventType` — token-level streaming events
- `StreamHandler`, `CallbackStreamHandler`, `EventBusStreamHandler`
- `StreamAccumulator` — bridges provider `stream()` to SDK `StreamEvent`
- `LLMAgent.process_task_streaming()` — full streaming with same lifecycle as `process_task()`

### Dynamic Spawning
- `ConfidenceMonitor` — per-agent/team confidence tracking, stagnation detection
- `ContentAnalyzer` — keyword-based topic coverage gap analysis
- `TeamSizeOptimizer` — heuristic team size recommendations
- `DynamicSpawner` — orchestrates monitor + analyzer, emits spawn events
- `WorkflowConfig.enable_dynamic_spawning` and `max_dynamic_agents` fields
- Wired into `FelixWorkflow` round loop

### CLI
- `felix init <name> [--template]` — scaffold projects (research, analysis, review templates)
- `felix run <config.yaml> [--provider] [--verbose]` — run workflows from YAML
- `felix version` — print SDK version
- YAML config loader with helix presets, team composition, all config fields

### Examples
- `06_event_system.py` — event subscriptions and workflow timeline
- `07_streaming_output.py` — token-level streaming demo
- `08_dynamic_spawning.py` — confidence-driven agent creation
- `09_structured_logging.py` — JSON and text log output
- `10_yaml_workflow/` — CLI workflow from YAML config

### Infrastructure
- Shared `STOPWORDS` extracted to `utils/text.py` (deduplicated from 3 modules)
- EventBus history capped at 10,000 events (configurable)
- 792+ tests, Python 3.10-3.12

---

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
