# Changelog

All notable changes to Felix Agent SDK will be documented in this file.

## [0.3.0] — 2026-06-10

Phase 3: Hardening & Async. Driven by a full-repo code review and a follow-up
adversarial re-review from the RLE-consumer perspective.

### Added
- **Async provider interface**: `AnthropicProvider`, `OpenAIProvider`, and
  `LocalProvider` now implement `acomplete()` and `astream()` using the
  vendor SDKs' async clients.
- `Agent.set_progress()` — public API to place an agent at a specific helix
  position (replaces orchestrators poking `_progress`). Placing at `t=1.0`
  transitions the agent to `COMPLETED`, making the placement stable across
  subsequent position updates.
- `HubCapacityError` (subclass of `RuntimeError`), exported from
  `felix_agent_sdk` and `felix_agent_sdk.communication`.
- `extract_status_code()` / `extract_retry_after()` helpers in
  `providers.errors`; `RateLimitError.retry_after` is now populated from
  the Retry-After header when present.
- `SQLiteBackend` supports the context manager protocol and is now safe
  for multi-threaded use (shared connection + internal lock).

### Changed
- **Breaking**: `CentralPost.register_agent()` raises `HubCapacityError` at
  capacity instead of returning `None`, matching `register_agent_id()`
  (which now raises the same subclass — `except RuntimeError` still works).
- **Behavior**: `CentralPost` processed-message history is now bounded
  (`message_history_limit=1000` by default); previously it grew without
  limit. Pass `message_history_limit=None` to restore unbounded history.
- `KnowledgeStore.add_relationship()` returns `False` instead of storing a
  relationship when either entry is missing or soft-deleted (such
  relationships were invisible to `get_relationships()` anyway).
- `HelixGeometry`/`HelixConfig` `turns` is typed `float` (fractional turns
  are geometrically valid and the YAML loader already produced floats).
- `LLMAgent.get_position_info()` now includes `agent_type`.
- `ProviderConfig.__repr__` masks the API key.
- Provider error translation prefers HTTP status codes from vendor SDK
  exceptions over message string matching; all translated errors carry
  `status_code` when available.
- `ProviderRegistry` detection priorities are stored per provider and
  honored during `auto_detect()` (previously priorities were reset on
  re-registration and ignored by detection).
- Default Anthropic model updated to `claude-sonnet-4-6`.
- `felix run` now prints the underlying cause when provider creation fails.
- `SpokeManager.create_spoke(auto_connect=True)` raises `HubCapacityError`
  when the hub is at capacity instead of silently returning a disconnected
  spoke (the spoke is not retained on failure).
- SQL identifier validation uses `re.fullmatch` (closing a trailing-newline
  bypass) and explicitly rejects `]` so it cannot break out of `[..]`
  bracket quoting.

### Fixed
- Dynamic spawning gap analysis received an empty `agent_type` for every
  result, so type-based spawn recommendations could never trigger.
- `SQLiteBackend.query()` interpolated `order_by` into SQL unvalidated;
  identifiers are now strictly validated and checked against known columns
  (session schema plus on-disk columns, so schema evolution still sorts).
- `KnowledgeStore.get_relationships()` no longer returns relationships
  involving soft-deleted entries.
- OpenAI/Local `stream()`/`astream()` stop consuming only at a true terminal
  chunk (usage with empty choices or a finish_reason), so servers that emit
  running usage on content chunks (e.g. vLLM `continuous_usage_stats`) are no
  longer silently truncated. A `close()` failure during stream teardown is
  swallowed (debug-logged) instead of masking a successful stream as a
  `ProviderError`.
- Provider error translation maps HTTP 404 to `ModelNotFoundError`, and the
  SDK exception class name (`NotFoundError`) is now matched correctly (the
  previous `"not_found"` substring check could never match).
- CentralPost async queue is created eagerly (removes a lazy-init race).
- 32 mypy strict errors across 13 files; `mypy src/` now gates CI.

### Infrastructure
- CI: Windows + Ubuntu matrix, Python 3.10–3.14, coverage report. A dedicated
  type-check job installs the provider SDKs (`.[all,dev]`) so mypy checks the
  provider layer against real vendor types rather than `Any`.
- Publish workflow verifies the release tag matches `_version.py`.
- `types-PyYAML` added to dev dependencies; explicit `asyncio_mode = "strict"`.

### Notes (behavior worth knowing)
- Provider HTTP 403 now maps to `AuthenticationError` — a per-model permission
  denial aborts rather than being retried as a generic error.
- `CentralPost.register_agent` no longer substitutes `id(agent)` for an
  empty-string `agent_id`; an empty id registers verbatim as `""`.
- A stream that completes without the server ever sending usage emits no
  `is_final` event (there is no terminal usage chunk to mark).
- `extract_retry_after` reads the numeric `Retry-After` header form only;
  `retry-after-ms` and HTTP-date forms are not yet parsed.

---

## [0.2.2] — 2026-06-09

### Fixed
- `AnthropicProvider`: omit sampling params for Claude Fable 5 / Opus 4.7+ (these
  models reject `temperature`/`top_p`/`top_k` with HTTP 400).
- Param-rejection errors are no longer misreported as `ModelNotFoundError`.

---

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
