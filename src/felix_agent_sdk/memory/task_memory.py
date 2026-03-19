"""Task memory system for the Felix Agent SDK.

Pattern recognition, success/failure tracking, and adaptive strategy
selection based on historical task execution data.

Algorithms ported from CalebisGross/felix ``src/memory/task_memory.py``.
Refactored to use the pluggable :class:`BaseBackend` interface.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

from felix_agent_sdk.memory.backends.base import BaseBackend
from felix_agent_sdk.memory.backends.sqlite import SQLiteBackend


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------


class TaskOutcome(Enum):
    """Possible outcomes for task execution."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


class TaskComplexity(Enum):
    """Task complexity levels."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_STOPWORDS: set[str] = {
    "the",
    "and",
    "for",
    "are",
    "but",
    "not",
    "you",
    "all",
    "can",
    "had",
    "her",
    "was",
    "one",
    "our",
    "out",
    "day",
    "get",
    "has",
    "him",
    "his",
    "how",
    "its",
    "may",
    "new",
    "now",
    "old",
    "see",
    "two",
    "who",
    "boy",
    "did",
    "man",
    "she",
    "use",
    "way",
    "oil",
    "sit",
    "set",
    "run",
}


def _get_enum_value(value: Any) -> str:
    """Safely get string value from an enum or string."""
    if isinstance(value, str):
        return value
    return getattr(value, "value", str(value))


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from *text*."""
    words = re.findall(r"\b\w{3,}\b", text.lower())
    keywords = [w for w in words if w not in _STOPWORDS and len(w) > 3]
    return list(set(keywords))


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class TaskPattern:
    """Pattern extracted from task execution history."""

    pattern_id: str
    task_type: str
    complexity: TaskComplexity
    keywords: list[str]
    typical_duration: float
    success_rate: float
    failure_modes: list[str]
    optimal_strategies: list[str]
    required_agents: list[str]
    context_requirements: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    usage_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["complexity"] = _get_enum_value(self.complexity)
        data["keywords"] = json.dumps(self.keywords)
        data["failure_modes"] = json.dumps(self.failure_modes)
        data["optimal_strategies"] = json.dumps(self.optimal_strategies)
        data["required_agents"] = json.dumps(self.required_agents)
        data["context_requirements"] = json.dumps(self.context_requirements)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskPattern:
        d = dict(data)
        if "_id" in d:
            d.setdefault("pattern_id", d.pop("_id"))
        else:
            d.pop("_id", None)
        d["complexity"] = TaskComplexity(d["complexity"])
        for list_field in ("keywords", "failure_modes", "optimal_strategies", "required_agents"):
            if isinstance(d.get(list_field), str):
                d[list_field] = json.loads(d[list_field])
        if isinstance(d.get("context_requirements"), str):
            d["context_requirements"] = json.loads(d["context_requirements"])
        return cls(**d)


@dataclass
class TaskExecution:
    """Record of a single task execution."""

    execution_id: str
    task_description: str
    task_type: str
    complexity: TaskComplexity
    outcome: TaskOutcome
    duration: float
    agents_used: list[str]
    strategies_used: list[str]
    context_size: int
    error_messages: list[str]
    success_metrics: dict[str, float]
    patterns_matched: list[str]
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["complexity"] = _get_enum_value(self.complexity)
        data["outcome"] = _get_enum_value(self.outcome)
        data["agents_used"] = json.dumps(self.agents_used)
        data["strategies_used"] = json.dumps(self.strategies_used)
        data["error_messages"] = json.dumps(self.error_messages)
        data["success_metrics"] = json.dumps(self.success_metrics)
        data["patterns_matched"] = json.dumps(self.patterns_matched)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskExecution:
        d = dict(data)
        if "_id" in d:
            d.setdefault("execution_id", d.pop("_id"))
        else:
            d.pop("_id", None)
        d["complexity"] = TaskComplexity(d["complexity"])
        d["outcome"] = TaskOutcome(d["outcome"])
        for list_field in ("agents_used", "strategies_used", "error_messages", "patterns_matched"):
            if isinstance(d.get(list_field), str):
                d[list_field] = json.loads(d[list_field])
        if isinstance(d.get("success_metrics"), str):
            d["success_metrics"] = json.loads(d["success_metrics"])
        return cls(**d)


@dataclass
class TaskMemoryQuery:
    """Query structure for task memory retrieval."""

    task_types: Optional[list[str]] = None
    complexity_levels: Optional[list[TaskComplexity]] = None
    keywords: Optional[list[str]] = None
    min_success_rate: Optional[float] = None
    max_duration: Optional[float] = None
    time_range: Optional[tuple[float, float]] = None
    limit: int = 10

    def build_filters(self) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if self.task_types:
            filters["task_type"] = {"$in": self.task_types}
        if self.complexity_levels:
            filters["complexity"] = {"$in": [_get_enum_value(c) for c in self.complexity_levels]}
        if self.min_success_rate is not None:
            filters["success_rate"] = {"$gte": self.min_success_rate}
        if self.max_duration is not None:
            filters["typical_duration"] = {"$lte": self.max_duration}
        if self.time_range:
            filters["created_at"] = {"$gte": self.time_range[0], "$lte": self.time_range[1]}
        return filters


# ------------------------------------------------------------------
# Schema definitions
# ------------------------------------------------------------------

_PATTERNS_SCHEMA: dict[str, str] = {
    "task_type": "TEXT",
    "complexity": "TEXT",
    "keywords": "TEXT",
    "typical_duration": "REAL",
    "success_rate": "REAL",
    "failure_modes": "TEXT",
    "optimal_strategies": "TEXT",
    "required_agents": "TEXT",
    "context_requirements": "TEXT",
    "created_at": "REAL",
    "updated_at": "REAL",
    "usage_count": "INTEGER",
}

_EXECUTIONS_SCHEMA: dict[str, str] = {
    "task_description": "TEXT",
    "task_type": "TEXT",
    "complexity": "TEXT",
    "outcome": "TEXT",
    "duration": "REAL",
    "agents_used": "TEXT",
    "strategies_used": "TEXT",
    "context_size": "INTEGER",
    "error_messages": "TEXT",
    "success_metrics": "TEXT",
    "patterns_matched": "TEXT",
    "created_at": "REAL",
}

_PAT_TABLE = "task_patterns"
_EXE_TABLE = "task_executions"


# ------------------------------------------------------------------
# TaskMemory
# ------------------------------------------------------------------


class TaskMemory:
    """Task memory for pattern recognition and adaptive strategy selection.

    Args:
        backend: Storage backend. Defaults to an in-memory
            :class:`SQLiteBackend` if not supplied.
    """

    def __init__(self, backend: Optional[BaseBackend] = None) -> None:
        self._backend = backend or SQLiteBackend()
        self._backend.initialize(_PAT_TABLE, _PATTERNS_SCHEMA)
        self._backend.initialize(_EXE_TABLE, _EXECUTIONS_SCHEMA)

    # ------------------------------------------------------------------
    # ID generation (ported from Felix)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_execution_id(task_description: str) -> str:
        hash_input = f"{task_description}:{time.time()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    @staticmethod
    def _generate_pattern_id(
        task_type: str, complexity: TaskComplexity, keywords: list[str]
    ) -> str:
        keywords_str = ":".join(sorted(keywords))
        hash_input = f"{task_type}:{_get_enum_value(complexity)}:{keywords_str}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------

    def record_task_execution(
        self,
        task_description: str,
        task_type: str,
        complexity: TaskComplexity,
        outcome: TaskOutcome,
        duration: float,
        agents_used: list[str],
        strategies_used: list[str],
        context_size: int,
        error_messages: Optional[list[str]] = None,
        success_metrics: Optional[dict[str, float]] = None,
    ) -> str:
        """Record a task execution and update patterns. Returns execution ID."""
        error_messages = error_messages or []
        success_metrics = success_metrics or {}

        execution_id = self._generate_execution_id(task_description)

        execution = TaskExecution(
            execution_id=execution_id,
            task_description=task_description,
            task_type=task_type,
            complexity=complexity,
            outcome=outcome,
            duration=duration,
            agents_used=agents_used,
            strategies_used=strategies_used,
            context_size=context_size,
            error_messages=error_messages,
            success_metrics=success_metrics,
            patterns_matched=[],
        )

        # Find matching patterns
        matched = self._find_matching_patterns(execution)
        execution.patterns_matched = [p.pattern_id for p in matched]

        # Store execution
        self._backend.store(_EXE_TABLE, execution_id, execution.to_dict())

        # Update or create patterns
        self._update_patterns_from_execution(execution)

        return execution_id

    # ------------------------------------------------------------------
    # Pattern matching (ported from Felix — 50% keyword overlap)
    # ------------------------------------------------------------------

    def _find_matching_patterns(self, execution: TaskExecution) -> list[TaskPattern]:
        query = TaskMemoryQuery(
            task_types=[execution.task_type],
            complexity_levels=[execution.complexity],
        )
        patterns = self.get_patterns(query)
        task_keywords = _extract_keywords(execution.task_description)

        matched: list[TaskPattern] = []
        for pattern in patterns:
            if not pattern.keywords:
                continue
            overlap = len(set(task_keywords) & set(pattern.keywords))
            if overlap >= len(pattern.keywords) * 0.5:
                matched.append(pattern)
        return matched

    def _update_patterns_from_execution(self, execution: TaskExecution) -> None:
        task_keywords = _extract_keywords(execution.task_description)
        if not task_keywords:
            return

        pattern_id = self._generate_pattern_id(
            execution.task_type, execution.complexity, task_keywords
        )

        existing = self._get_pattern_by_id(pattern_id)
        if existing:
            self._update_existing_pattern(existing, execution)
        else:
            self._create_new_pattern(pattern_id, execution, task_keywords)

    def _get_pattern_by_id(self, pattern_id: str) -> Optional[TaskPattern]:
        data = self._backend.get(_PAT_TABLE, pattern_id)
        if data is None:
            return None
        return TaskPattern.from_dict(data)

    def _update_existing_pattern(self, pattern: TaskPattern, execution: TaskExecution) -> None:
        """Update existing pattern with new execution data."""
        executions = self._get_executions_for_pattern(pattern.pattern_id)
        executions.append(execution)

        # Recalculate success rate
        successes = sum(
            1 for e in executions if e.outcome in (TaskOutcome.SUCCESS, TaskOutcome.PARTIAL_SUCCESS)
        )
        pattern.success_rate = successes / len(executions)

        # Recalculate typical duration
        pattern.typical_duration = sum(e.duration for e in executions) / len(executions)

        # Update failure modes
        failures = [e for e in executions if e.outcome in (TaskOutcome.FAILURE, TaskOutcome.ERROR)]
        all_failures: list[str] = []
        for f in failures:
            all_failures.extend(f.error_messages)
        pattern.failure_modes = list(set(all_failures))

        # Update optimal strategies (from successful executions)
        success_execs = [e for e in executions if e.outcome == TaskOutcome.SUCCESS]
        strategy_counts: dict[str, int] = {}
        for s in success_execs:
            for strategy in s.strategies_used:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        pattern.optimal_strategies = sorted(
            strategy_counts, key=lambda x: strategy_counts[x], reverse=True
        )[:5]

        # Update required agents
        agent_counts: dict[str, int] = {}
        for s in success_execs:
            for agent in s.agents_used:
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
        pattern.required_agents = sorted(agent_counts, key=lambda x: agent_counts[x], reverse=True)[
            :3
        ]

        pattern.updated_at = time.time()
        pattern.usage_count += 1
        self._backend.store(_PAT_TABLE, pattern.pattern_id, pattern.to_dict())

    def _create_new_pattern(
        self, pattern_id: str, execution: TaskExecution, keywords: list[str]
    ) -> None:
        is_success = execution.outcome in (TaskOutcome.SUCCESS, TaskOutcome.PARTIAL_SUCCESS)
        is_failure = execution.outcome in (TaskOutcome.FAILURE, TaskOutcome.ERROR)

        pattern = TaskPattern(
            pattern_id=pattern_id,
            task_type=execution.task_type,
            complexity=execution.complexity,
            keywords=keywords,
            typical_duration=execution.duration,
            success_rate=1.0 if is_success else 0.0,
            failure_modes=execution.error_messages if is_failure else [],
            optimal_strategies=execution.strategies_used
            if execution.outcome == TaskOutcome.SUCCESS
            else [],
            required_agents=execution.agents_used
            if execution.outcome == TaskOutcome.SUCCESS
            else [],
            context_requirements={
                "min_context_size": execution.context_size,
                "success_metrics": execution.success_metrics,
            },
            usage_count=1,
        )
        self._backend.store(_PAT_TABLE, pattern_id, pattern.to_dict())

    def _get_executions_for_pattern(self, pattern_id: str) -> list[TaskExecution]:
        rows = self._backend.query(
            _EXE_TABLE,
            filters={"patterns_matched": {"$contains": pattern_id}},
        )
        return [TaskExecution.from_dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_patterns(self, query: TaskMemoryQuery) -> list[TaskPattern]:
        """Retrieve task patterns matching *query*."""
        rows = self._backend.query(
            _PAT_TABLE,
            filters=query.build_filters(),
            order_by="success_rate",
            ascending=False,
            limit=query.limit,
        )
        patterns: list[TaskPattern] = []
        for row in rows:
            pattern = TaskPattern.from_dict(row)
            # Post-filter by keywords if specified
            if query.keywords:
                pattern_kw_lower = [k.lower() for k in pattern.keywords]
                if not any(kw.lower() in pattern_kw_lower for kw in query.keywords):
                    continue
            patterns.append(pattern)
        return patterns

    # ------------------------------------------------------------------
    # Strategy recommendation (ported from Felix)
    # ------------------------------------------------------------------

    def recommend_strategy(
        self,
        task_description: str,
        task_type: str,
        complexity: TaskComplexity,
    ) -> dict[str, Any]:
        """Recommend optimal strategy based on historical patterns."""
        keywords = _extract_keywords(task_description)
        query = TaskMemoryQuery(
            task_types=[task_type],
            complexity_levels=[complexity],
            keywords=keywords,
            min_success_rate=0.5,
            limit=5,
        )
        patterns = self.get_patterns(query)

        if not patterns:
            return {
                "strategies": [],
                "agents": [],
                "estimated_duration": None,
                "success_probability": 0.0,
                "recommendations": "No similar patterns found. Proceeding with default strategy.",
                "potential_issues": [],
            }

        all_strategies: list[str] = []
        all_agents: list[str] = []
        durations: list[float] = []
        success_rates: list[float] = []
        potential_issues: list[str] = []

        for pattern in patterns:
            all_strategies.extend(pattern.optimal_strategies)
            all_agents.extend(pattern.required_agents)
            durations.append(pattern.typical_duration)
            success_rates.append(pattern.success_rate)
            potential_issues.extend(pattern.failure_modes)

        # Most common strategies and agents
        strategy_counts: dict[str, int] = {}
        for s in all_strategies:
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
        agent_counts: dict[str, int] = {}
        for a in all_agents:
            agent_counts[a] = agent_counts.get(a, 0) + 1

        recommended_strategies = sorted(
            strategy_counts, key=lambda x: strategy_counts[x], reverse=True
        )[:3]
        recommended_agents = sorted(agent_counts, key=lambda x: agent_counts[x], reverse=True)[:3]

        avg_duration = sum(durations) / len(durations) if durations else None
        avg_success = sum(success_rates) / len(success_rates) if success_rates else 0.0

        recommendations: list[str] = []
        if recommended_strategies:
            recommendations.append(
                f"Use proven strategies: {', '.join(recommended_strategies[:2])}"
            )
        if recommended_agents:
            recommendations.append(f"Deploy agents: {', '.join(recommended_agents[:2])}")
        if avg_duration:
            recommendations.append(f"Expected duration: {avg_duration:.1f} seconds")

        return {
            "strategies": recommended_strategies,
            "agents": recommended_agents,
            "estimated_duration": avg_duration,
            "success_probability": avg_success,
            "recommendations": ". ".join(recommendations),
            "potential_issues": list(set(potential_issues))[:3],
            "patterns_used": len(patterns),
        }

    # ------------------------------------------------------------------
    # Summary / cleanup
    # ------------------------------------------------------------------

    def get_memory_summary(self) -> dict[str, Any]:
        """Aggregate statistics about task memory."""
        total_patterns = self._backend.count(_PAT_TABLE)
        total_executions = self._backend.count(_EXE_TABLE)
        return {
            "total_patterns": total_patterns,
            "total_executions": total_executions,
        }

    def cleanup_old_patterns(self, max_age_days: int = 60, min_usage_count: int = 2) -> int:
        """Remove old or unused patterns."""
        cutoff = time.time() - max_age_days * 86400
        old = self._backend.query(
            _PAT_TABLE,
            filters={"created_at": {"$lt": cutoff}, "usage_count": {"$lt": min_usage_count}},
        )
        zero = self._backend.query(
            _PAT_TABLE,
            filters={"success_rate": 0.0, "usage_count": 1},
        )
        to_delete = {row["_id"] for row in old} | {row["_id"] for row in zero}
        for pid in to_delete:
            self._backend.delete(_PAT_TABLE, pid)
        return len(to_delete)
