"""Tests for the TaskMemory pattern recognition system."""

from __future__ import annotations

import pytest

from felix_agent_sdk.memory.task_memory import (
    TaskComplexity,
    TaskExecution,
    TaskMemory,
    TaskMemoryQuery,
    TaskOutcome,
    TaskPattern,
    _extract_keywords,
)


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


class TestTaskPattern:
    def test_roundtrip(self):
        pattern = TaskPattern(
            pattern_id="p1",
            task_type="analysis",
            complexity=TaskComplexity.COMPLEX,
            keywords=["data", "statistics"],
            typical_duration=30.0,
            success_rate=0.85,
            failure_modes=["timeout"],
            optimal_strategies=["parallel"],
            required_agents=["research"],
            context_requirements={"min_context_size": 1000},
        )
        d = pattern.to_dict()
        restored = TaskPattern.from_dict(d)
        assert restored.pattern_id == "p1"
        assert restored.complexity == TaskComplexity.COMPLEX
        assert restored.keywords == ["data", "statistics"]


class TestTaskExecution:
    def test_roundtrip(self):
        exe = TaskExecution(
            execution_id="e1",
            task_description="analyse data",
            task_type="analysis",
            complexity=TaskComplexity.MODERATE,
            outcome=TaskOutcome.SUCCESS,
            duration=10.0,
            agents_used=["research-1"],
            strategies_used=["parallel"],
            context_size=500,
            error_messages=[],
            success_metrics={"accuracy": 0.95},
            patterns_matched=["p1"],
        )
        d = exe.to_dict()
        restored = TaskExecution.from_dict(d)
        assert restored.outcome == TaskOutcome.SUCCESS
        assert restored.success_metrics == {"accuracy": 0.95}


class TestKeywordExtraction:
    def test_extracts_meaningful_words(self):
        kw = _extract_keywords("Analyse the complex statistical data patterns")
        assert "analyse" in kw
        assert "complex" in kw
        assert "the" not in kw  # stopword

    def test_filters_short_words(self):
        kw = _extract_keywords("a an the big cat sat on mat")
        assert "the" not in kw
        assert "sat" not in kw  # 3 chars, but > 3 required

    def test_empty_input(self):
        assert _extract_keywords("") == []


# ------------------------------------------------------------------
# TaskMemory
# ------------------------------------------------------------------


class TestTaskMemoryRecording:
    def test_record_creates_execution(self, task_memory):
        eid = task_memory.record_task_execution(
            task_description="Analyse renewable energy data patterns",
            task_type="analysis",
            complexity=TaskComplexity.MODERATE,
            outcome=TaskOutcome.SUCCESS,
            duration=15.0,
            agents_used=["research-1"],
            strategies_used=["parallel"],
            context_size=500,
        )
        assert isinstance(eid, str)
        assert len(eid) == 16

    def test_record_creates_pattern(self, task_memory):
        task_memory.record_task_execution(
            task_description="Analyse renewable energy data patterns",
            task_type="analysis",
            complexity=TaskComplexity.MODERATE,
            outcome=TaskOutcome.SUCCESS,
            duration=15.0,
            agents_used=["research-1"],
            strategies_used=["parallel"],
            context_size=500,
        )
        patterns = task_memory.get_patterns(
            TaskMemoryQuery(task_types=["analysis"])
        )
        assert len(patterns) >= 1

    def test_second_execution_updates_pattern(self, task_memory):
        """Two executions with same keywords should update the same pattern."""
        task_memory.record_task_execution(
            task_description="Analyse renewable energy data patterns",
            task_type="analysis",
            complexity=TaskComplexity.MODERATE,
            outcome=TaskOutcome.SUCCESS,
            duration=10.0,
            agents_used=["research-1"],
            strategies_used=["parallel"],
            context_size=500,
        )
        task_memory.record_task_execution(
            task_description="Analyse renewable energy data patterns",
            task_type="analysis",
            complexity=TaskComplexity.MODERATE,
            outcome=TaskOutcome.FAILURE,
            duration=20.0,
            agents_used=["research-1"],
            strategies_used=["sequential"],
            context_size=600,
            error_messages=["timeout exceeded"],
        )
        patterns = task_memory.get_patterns(
            TaskMemoryQuery(task_types=["analysis"])
        )
        assert len(patterns) >= 1
        # Pattern should have been updated (usage_count > 1)
        p = patterns[0]
        assert p.usage_count >= 2


class TestTaskMemoryPatternMatching:
    def test_keyword_overlap_threshold(self, task_memory):
        """Pattern matching requires 50% keyword overlap."""
        task_memory.record_task_execution(
            task_description="Analyse complex statistical data patterns",
            task_type="analysis",
            complexity=TaskComplexity.COMPLEX,
            outcome=TaskOutcome.SUCCESS,
            duration=10.0,
            agents_used=["research"],
            strategies_used=["parallel"],
            context_size=500,
        )
        # This should NOT match (completely different keywords)
        task_memory.record_task_execution(
            task_description="Write marketing blog post content",
            task_type="writing",
            complexity=TaskComplexity.SIMPLE,
            outcome=TaskOutcome.SUCCESS,
            duration=5.0,
            agents_used=["writer"],
            strategies_used=["sequential"],
            context_size=200,
        )
        analysis_patterns = task_memory.get_patterns(
            TaskMemoryQuery(task_types=["analysis"])
        )
        writing_patterns = task_memory.get_patterns(
            TaskMemoryQuery(task_types=["writing"])
        )
        # They should be separate patterns
        assert len(analysis_patterns) >= 1
        assert len(writing_patterns) >= 1


class TestStrategyRecommendation:
    def test_recommend_with_history(self, task_memory):
        task_memory.record_task_execution(
            task_description="Analyse complex statistical data patterns",
            task_type="analysis",
            complexity=TaskComplexity.COMPLEX,
            outcome=TaskOutcome.SUCCESS,
            duration=15.0,
            agents_used=["research", "analysis"],
            strategies_used=["parallel", "deep_dive"],
            context_size=1000,
        )
        rec = task_memory.recommend_strategy(
            "Analyse statistical data distributions",
            "analysis",
            TaskComplexity.COMPLEX,
        )
        assert rec["success_probability"] > 0
        assert len(rec["strategies"]) > 0

    def test_recommend_without_history(self, task_memory):
        rec = task_memory.recommend_strategy(
            "Completely new task type",
            "unknown",
            TaskComplexity.SIMPLE,
        )
        assert abs(rec["success_probability"]) < 1e-9
        assert rec["strategies"] == []


class TestTaskMemorySummaryAndCleanup:
    def test_summary(self, task_memory):
        task_memory.record_task_execution(
            task_description="test task for summary",
            task_type="test",
            complexity=TaskComplexity.SIMPLE,
            outcome=TaskOutcome.SUCCESS,
            duration=1.0,
            agents_used=["a"],
            strategies_used=["s"],
            context_size=100,
        )
        summary = task_memory.get_memory_summary()
        assert summary["total_patterns"] >= 1
        assert summary["total_executions"] >= 1

    def test_cleanup(self, task_memory):
        task_memory.record_task_execution(
            task_description="old pattern cleanup test task",
            task_type="cleanup",
            complexity=TaskComplexity.SIMPLE,
            outcome=TaskOutcome.FAILURE,
            duration=1.0,
            agents_used=["a"],
            strategies_used=["s"],
            context_size=100,
            error_messages=["fail"],
        )
        # With max_age_days=0, should clean up the single-use failure pattern
        removed = task_memory.cleanup_old_patterns(max_age_days=0, min_usage_count=2)
        assert removed >= 1
