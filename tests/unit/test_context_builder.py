"""Tests for CollaborativeContextBuilder."""

from __future__ import annotations


from felix_agent_sdk.workflows.context_builder import (
    CollaborativeContextBuilder,
    Contribution,
)


class TestContribution:
    def test_construction(self):
        c = Contribution(
            agent_id="agent-1",
            agent_type="research",
            content="Some findings",
            confidence=0.75,
            phase="exploration",
        )
        assert c.agent_id == "agent-1"
        assert c.timestamp > 0


class TestContextBuilderBasics:
    def test_empty(self):
        builder = CollaborativeContextBuilder()
        assert builder.contribution_count == 0
        assert builder.version == 0
        assert builder.build_context() == ""
        assert builder.get_context_history() == []

    def test_add_contribution(self):
        builder = CollaborativeContextBuilder()
        builder.add_contribution("a1", "research", "findings", 0.7, "exploration")
        assert builder.contribution_count == 1
        assert builder.version == 1

    def test_version_increments(self):
        builder = CollaborativeContextBuilder()
        builder.add_contribution("a1", "research", "x", 0.5, "exploration")
        builder.add_contribution("a2", "analysis", "y", 0.6, "analysis")
        assert builder.version == 2


class TestBuildContext:
    def test_builds_formatted_string(self):
        builder = CollaborativeContextBuilder()
        builder.add_contribution("a1", "research", "Finding A", 0.8, "exploration")
        builder.add_contribution("a2", "analysis", "Finding B", 0.6, "analysis")

        ctx = builder.build_context()
        assert "a1" in ctx
        assert "Finding A" in ctx
        assert "a2" in ctx

    def test_max_entries(self):
        builder = CollaborativeContextBuilder()
        for i in range(20):
            builder.add_contribution(f"a{i}", "research", f"content {i}", 0.5, "exploration")

        ctx = builder.build_context(max_entries=3)
        # Should not contain all 20 agents
        assert ctx.count("[a") <= 3

    def test_higher_confidence_ranked_higher(self):
        builder = CollaborativeContextBuilder()
        builder.add_contribution("low", "research", "low conf", 0.1, "exploration")
        builder.add_contribution("high", "research", "high conf", 0.9, "exploration")

        history = builder.get_context_history(max_entries=1)
        assert len(history) == 1
        assert history[0]["agent_id"] == "high"


class TestContextHistory:
    def test_format(self):
        builder = CollaborativeContextBuilder()
        builder.add_contribution("a1", "research", "content here", 0.7, "exploration")

        history = builder.get_context_history()
        assert len(history) == 1
        assert history[0]["agent_id"] == "a1"
        assert history[0]["content"] == "content here"


class TestMergeContributions:
    def test_merge_keys(self):
        builder = CollaborativeContextBuilder()
        builder.add_contribution("a1", "research", "Alpha", 0.7, "exploration")
        builder.add_contribution("a2", "analysis", "Beta", 0.8, "analysis")

        merged = builder.merge_contributions()
        assert "a1_exploration" in merged
        assert "a2_analysis" in merged
        assert merged["a1_exploration"] == "Alpha"


class TestDeduplication:
    def test_removes_duplicates(self):
        builder = CollaborativeContextBuilder()
        builder.add_contribution("a1", "research", "The renewable energy data shows patterns", 0.7, "exploration")
        builder.add_contribution("a2", "research", "The renewable energy data shows clear patterns", 0.6, "exploration")
        builder.add_contribution("a3", "analysis", "Financial markets are volatile today", 0.8, "analysis")

        removed = builder.deduplicate(similarity_threshold=0.5)
        assert removed >= 1
        assert builder.contribution_count < 3

    def test_no_duplicates(self):
        builder = CollaborativeContextBuilder()
        builder.add_contribution("a1", "research", "Topic alpha about science", 0.7, "exploration")
        builder.add_contribution("a2", "analysis", "Topic beta about finance", 0.8, "analysis")

        removed = builder.deduplicate()
        assert removed == 0
        assert builder.contribution_count == 2

    def test_empty_dedup(self):
        builder = CollaborativeContextBuilder()
        assert builder.deduplicate() == 0
