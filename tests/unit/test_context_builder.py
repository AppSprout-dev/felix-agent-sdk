"""Tests for CollaborativeContextBuilder."""

from __future__ import annotations


import time

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

class TestContextEfficiency:
    def test_skips_empty_content(self):
        builder = CollaborativeContextBuilder()
        builder.add_contribution("a1", "research", "   ", 0.7, "exploration")
        builder.add_contribution("a2", "research", "real findings", 0.8, "exploration")
        assert builder.contribution_count == 1

    def test_truncates_long_content_in_build(self):
        builder = CollaborativeContextBuilder(max_chars_per_entry=50)
        builder.add_contribution("a1", "research", "x" * 200, 0.9, "exploration")
        ctx = builder.build_context()
        # header + truncated body; body should not retain all 200 chars
        assert "x" * 200 not in ctx
        assert "…" in ctx or len(ctx) < 200

    def test_no_truncate_when_disabled(self):
        builder = CollaborativeContextBuilder(max_chars_per_entry=None)
        body = "y" * 120
        builder.add_contribution("a1", "research", body, 0.9, "exploration")
        ctx = builder.build_context()
        assert body in ctx


class TestScoringConfig:
    """Configurable recency / confidence scoring parameters."""

    def test_custom_recency_decay(self):
        """Faster decay means older contributions score lower."""
        slow = CollaborativeContextBuilder(recency_decay_rate=0.01)
        fast = CollaborativeContextBuilder(recency_decay_rate=1.0)

        now = time.time()
        old = Contribution("a1", "research", "old", 0.5, "exploration", timestamp=now - 30)
        new = Contribution("a2", "research", "new", 0.5, "exploration", timestamp=now)

        def recency_score(builder, c):
            age = time.time() - c.timestamp
            return max(0.0, builder._recency_max_weight - age * builder._recency_decay_rate)

        assert recency_score(slow, old) > 0.0  # slow decay keeps old score
        assert recency_score(fast, old) == 0.0  # fast decay zeros old
        assert recency_score(fast, new) > 0.0   # even fast decay keeps new

    def test_custom_confidence_weight(self):
        """Higher weight amplifies confidence in combined score."""
        low = CollaborativeContextBuilder(confidence_weight=0.3)
        high = CollaborativeContextBuilder(confidence_weight=0.9)
        low.add_contribution("a1", "research", "data", 0.8, "exploration")
        high.add_contribution("a1", "research", "data", 0.8, "exploration")

        # Both produce context (score > 0)
        assert len(low.build_context()) > 0
        assert len(high.build_context()) > 0

        # The higher-weight builder gives a strictly higher score for same input
        low_score = low._score_contributions()[0][1]
        high_score = high._score_contributions()[0][1]
        assert high_score > low_score

    def test_defaults_match_original_behavior(self):
        """Default parameter values preserve original magic-number behaviour."""
        default = CollaborativeContextBuilder()
        now = time.time()
        c = Contribution("a1", "r", "content", 0.7, "exploration", timestamp=now)
        age = time.time() - c.timestamp
        expected_recency = max(0.0, 0.5 - age * 0.01)
        expected = expected_recency + 0.7 * 0.5
        default._contributions.append(c)
        score = default._score_contributions()[0][1]
        assert abs(score - expected) < 0.001
