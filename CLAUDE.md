# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Hard Rules

1. Use a venv at `.venv` for all development, testing, and execution.
2. When porting code from the original Felix framework (`https://github.com/CalebisGross/felix`), Caleb's architecture decisions take precedence. Core algorithms stay identical; only the interface layer changes.
3. Git attribution for ported/refactored code:
   - **Ported code** (Caleb's algorithms, unchanged): `git commit --author="Caleb Gross <209704970+CalebisGross@users.noreply.github.com>"`
   - **Refactored code** (Caleb's logic, new interfaces): add `Co-authored-by: Caleb Gross <209704970+CalebisGross@users.noreply.github.com>` trailer
   - **New code** (provider layer, configs, tests, docs): normal commits
4. Never commit `.env`, credentials, `*.db` files, or `felix-agent-sdk-structure.md` (internal design doc).
5. All PRs require review from `@CalebisGross` (enforced via CODEOWNERS).

## Project Overview

`felix-agent-sdk` is a pip-installable Python SDK that extracts and repackages the internal Felix Framework into a provider-agnostic, open-source package for external developers.

- **GitHub org:** [AppSprout-dev](https://github.com/AppSprout-dev)
- **Public SDK repo:** `AppSprout-dev/felix-agent-sdk`
- **Private Felix source:** `AppSprout-dev/felix` (original by Caleb Gross)
- **Team:** Jason Bennett (@jkbennitt) — SDK lead. Caleb Gross (@CalebisGross) — original Felix author, required reviewer.

## Build & Run Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Install with specific provider
pip install -e ".[anthropic]"   # Claude models
pip install -e ".[openai]"      # OpenAI models
pip install -e ".[local]"       # LM Studio / Ollama
pip install -e ".[all]"         # Everything

# Testing
pytest tests/
pytest tests/unit/
pytest tests/ -v --cov=src

# Linting & type checking
ruff check src/
ruff format src/
mypy src/
```

## Architecture

### Package Layout

```
src/felix_agent_sdk/
├── __init__.py          # Public API surface + version
├── _version.py          # Single source of truth for version
├── core/                # HelixGeometry, HelixConfig, HelixPosition
├── agents/              # Agent, LLMAgent, specialized agents, AgentFactory
├── communication/       # CentralPost, Spoke, messages, AgentRegistry
├── memory/              # KnowledgeStore, TaskMemory, compression, pluggable backends
├── providers/           # BaseProvider, Anthropic, OpenAI, Local (IMPLEMENTED)
├── spawning/            # DynamicSpawning, ConfidenceMonitor, ContentAnalyzer
├── workflows/           # Felix workflow runner + templates
├── tokens/              # TokenBudgetManager, provider-aware counting
├── streaming/           # StreamEvent types, StreamHandler
└── utils/               # Logging, config loading, common types
```

### Key Design Decisions

- **Provider abstraction** is the critical layer. Every provider implements `BaseProvider` with `complete()`, `stream()`, `count_tokens()`. This decouples all agent logic from specific LLM APIs.
- **`openai_provider.py`** (not `openai.py`) avoids shadowing the `openai` package import.
- **Helix model:** agents traverse `t ∈ [0,1]` along a 3D spiral. Top (t=0) = exploration, high temperature, broad search. Bottom (t=1) = synthesis, low temperature, focused output. Default: top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2.0.
- **Hub-spoke O(N)** communication via CentralPost. All messages route through the hub, never direct agent-to-agent mesh (O(N²)).
- **Memory** uses pluggable backends with SQLite as default.
- **Dynamic spawning** is confidence-driven. If team confidence drops below threshold, new agents spawn automatically.

### Porting Reference

The original Felix source is at `https://github.com/CalebisGross/felix`. Key files to port:

| Felix source | SDK target | Notes |
|---|---|---|
| `src/core/helix_geometry.py` (186 lines) | `core/helix.py` | Add HelixConfig presets, HelixPosition |
| `src/agents/agent.py` (372 lines) | `agents/base.py` | Extract AgentState, clean interface |
| `src/agents/llm_agent.py` (2130 lines) | `agents/llm_agent.py` | Wire to BaseProvider (biggest refactor) |
| `src/agents/specialized_agents.py` (578 lines) | `agents/specialized.py` | Keep Research, Analysis, Critic |
| `src/agents/dynamic_spawning.py` (1204 lines) | `spawning/` | Split into focused modules |
| `src/communication/central_post.py` (3210 lines) | `communication/` | Extract AgentRegistry, decouple memory |
| `src/communication/spoke.py` (514 lines) | `communication/spoke.py` | Minimal changes |
| `src/memory/knowledge_store.py` (2953 lines) | `memory/` | Add pluggable backend interface |
| `src/memory/task_memory.py` (705 lines) | `memory/` | Pattern recognition |
| `src/memory/context_compression.py` (564 lines) | `memory/` | Compression strategies |
| `src/llm/token_budget.py` (331 lines) | `tokens/budget.py` | Provider-aware counting |
| `src/workflows/felix_workflow.py` | `workflows/` | Add template system |

### What NOT to Port

- Tkinter/PySide6 GUI code
- `exp/` benchmarking scripts (write new ones)
- `linear_pipeline.py` (research baseline)
- `lm_studio_client.py` / `multi_server_client.py` (replaced by provider layer)
- CLI, API, execution/trust system, knowledge daemon

## Phased Implementation

- **Phase 1** (current): Package skeleton + provider layer
- **Phase 2** `feat/core-geometry`: Port HelixGeometry, add HelixConfig/HelixPosition
- **Phase 3** `feat/providers`: Tests for provider layer
- **Phase 4** `feat/agents`: Port agent classes, wire to BaseProvider
- **Phase 5** `feat/communication`: Port CentralPost, Spoke, messages
- **Phase 6** `feat/memory`: Port KnowledgeStore, TaskMemory, ContextCompressor
- **Phase 7** `feat/workflows`: Port workflow runner + templates
