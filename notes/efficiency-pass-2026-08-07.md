# Felix efficiency pass — 2026-08-07

Hard task: research the helix multi-agent concept and iteratively improve efficiency. Yield Framework applied throughout.

**Actors:** human-directed session + Prime Agent (WSL, `openrouter/deepseek/deepseek-v4-flash`) for exploration; concrete patches landed in-repo after agent stalled on Windows/WSL venv paths.

---

## Concept (research summary)

Felix models multi-agent work as **motion on a helix**:

| Idea | Mechanism |
|------|-----------|
| Explore → synthesize | Helix time `t` drives temperature / prompt stance |
| O(N) comms | Hub-spoke via `CentralPost` (not full mesh) |
| Dynamic capacity | `DynamicSpawner` + `ConfidenceMonitor` + `ContentAnalyzer` |
| Context growth | `CollaborativeContextBuilder` scores recency + confidence |
| Token awareness | `TokenBudget.from_helix_position` shifts input/output share with `t` |

Efficiency bottlenecks (Yield lens):

1. **Cleverness debt in team sizing** — magic length/confidence thresholds.
2. **Unnecessary agents** — length growth even when confidence is already high.
3. **Context bloat** — unlimited contribution text re-injected each round.
4. **Spawn path work** — coverage analysis after confidence already says HOLD (code already short-circuits; comment clarified).

---

## Yield audit

### AMPLIFY
- Hub-spoke O(N) design and helix-position token budgets.
- Config-driven `TeamSizeConfig` + high-confidence short-circuit.
- Per-entry context char caps + empty contribution skip.
- Keyword cache on dedupe.

### DELETE
- None of product surface; removed *reliance* on hardcoded optimizer thresholds as the only API.

### REPLACE
- Magic numbers in `TeamSizeOptimizer` → `TeamSizeConfig` dataclass.
- Unbounded context payload → truncated `build_context` bodies.

### DEFER
- Learned team-size policy (RL / bandit) instead of any hand threshold.
  ✅ 2026-08-07: `_score_contributions` magic numbers made configurable.
- Streaming context compression tied to live token meters.
- Cross-project Continual Harness isolation for multi-CEM agents.

---

## Changes made

### 1. `TeamSizeOptimizer` (`spawning/optimizer.py`)
- Added frozen `TeamSizeConfig` with prior defaults.
- **High-confidence path:** if average confidence ≥ `high_confidence_skip_length` (default 0.85), **skip length-based headcount growth** so long prompts do not spawn extra agents when quality is already high.
- Constructor accepts optional `config=`.

### 2. `CollaborativeContextBuilder` (`workflows/context_builder.py`)
- `max_chars_per_entry` (default 4000) truncates bodies in `build_context`.
- `skip_empty=True` drops blank contributions.
- Keyword cache for faster `deduplicate`.

### 3. `DynamicSpawner`
- Comment-only clarification of the existing early return before coverage analysis when confidence says HOLD.

### 4. `CollaborativeContextBuilder` — configurable scoring parameters
- Added `recency_decay_rate`, `recency_max_weight`, `confidence_weight` constructor
  parameters to replace hardcoded magic numbers in `_score_contributions`.
- Backward-compatible: defaults (`0.01`, `0.5`, `0.5`) preserve original behaviour
  exactly (confirmed by `test_defaults_match_original_behavior`).
- Removes three hardcoded values flagged by Yield audit without breaking API.

### Tests
- `test_high_confidence_skips_length_growth`
- `test_config_overrides_thresholds`
- `test_skips_empty_content`, `test_truncates_long_content_in_build`, `test_no_truncate_when_disabled`
- `test_custom_recency_decay`, `test_custom_confidence_weight`, `test_defaults_match_original_behavior`

---

## Efficiency impact (reasoned)

| Change | Expected effect |
|--------|-----------------|
| High-conf skip length growth | Fewer LLM agents on long, already-solved tasks → lower multi-agent token multiply |
| Context char cap | Bounds prompt growth from verbose agents |
| Skip empty | Fewer no-op history rows |
| Keyword cache | Cheaper dedupe on large contribution sets |
| Configurable scoring params | Lets callers tune recency half-life and confidence weight for their task duration, avoiding stale-context or noise problems without forking the builder |

Not measured end-to-end with live providers in this pass (cost/latency depends on model).

---

## Prime Agent trial notes

- Install + WSL path works with DeepSeek V4 Flash via OpenRouter.
- Agent successfully: loaded yield skill, scanned repo, read core modules.
- Agent stalled: Windows `.venv` on `/mnt/c` vs WSL (pytest path hell).
- Mitigation: Linux venv at `~/felix-venv` (897 unit tests green baseline); restart prompt forces that interpreter.

---

## Remaining work

- Wire `TeamSizeConfig` into `WorkflowConfig` / CLI defaults.
- Auto-dedupe each round before `build_context` in runner.
- Optional soft token budget hard-stop in runner when over allocation.
- Prime Agent: document WSL+Windows monorepo venv recipe in Felix README.
