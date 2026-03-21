"""Tests for TeamSizeOptimizer."""

from __future__ import annotations

from felix_agent_sdk.spawning.optimizer import TeamSizeOptimizer


class TestTeamSizeOptimizer:
    def test_default_for_simple_task(self):
        opt = TeamSizeOptimizer()
        size = opt.recommend_team_size("Short task")
        assert size == 3

    def test_long_task_increases_size(self):
        opt = TeamSizeOptimizer()
        short = opt.recommend_team_size("Short")
        long_task = "x" * 250
        medium = opt.recommend_team_size(long_task)
        assert medium > short

    def test_very_long_task(self):
        opt = TeamSizeOptimizer()
        size = opt.recommend_team_size("x" * 600)
        assert size >= 5  # base 3 + 2 length signals

    def test_low_confidence_increases_size(self):
        opt = TeamSizeOptimizer()
        results = [{"confidence": 0.3, "content": "weak"} for _ in range(3)]
        size = opt.recommend_team_size("task", results)
        assert size > 3

    def test_high_confidence_keeps_base(self):
        opt = TeamSizeOptimizer()
        results = [{"confidence": 0.9, "content": "strong"} for _ in range(3)]
        size = opt.recommend_team_size("Short", results)
        assert size == 3

    def test_wide_spread_increases_size(self):
        opt = TeamSizeOptimizer()
        results = [
            {"confidence": 0.3, "content": "a"},
            {"confidence": 0.9, "content": "b"},
        ]
        size = opt.recommend_team_size("Short", results)
        # Low avg (0.6 < 0.7) + spread (0.6 > 0.3) = +2
        assert size > 3

    def test_respects_max_size(self):
        opt = TeamSizeOptimizer(max_size=4)
        size = opt.recommend_team_size("x" * 600, [{"confidence": 0.1, "content": "x"}])
        assert size <= 4

    def test_respects_min_size(self):
        opt = TeamSizeOptimizer(min_size=5)
        size = opt.recommend_team_size("Short")
        assert size >= 5
