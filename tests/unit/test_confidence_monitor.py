"""Tests for ConfidenceMonitor."""

from __future__ import annotations

from felix_agent_sdk.spawning.confidence_monitor import (
    ConfidenceMonitor,
    SpawnRecommendation,
)


class TestConfidenceMonitorBasics:
    def test_no_data_returns_hold(self):
        mon = ConfidenceMonitor(threshold=0.8)
        status = mon.get_status()
        assert status.team_average == 0.0
        assert status.recommendation == SpawnRecommendation.SPAWN  # big gap

    def test_above_threshold_holds(self):
        mon = ConfidenceMonitor(threshold=0.7)
        mon.record_round({"a": 0.8, "b": 0.9})
        assert mon.get_status().recommendation == SpawnRecommendation.HOLD

    def test_below_threshold_falling_spawns(self):
        mon = ConfidenceMonitor(threshold=0.8)
        mon.record_round({"a": 0.7, "b": 0.6})
        mon.record_round({"a": 0.5, "b": 0.4})
        status = mon.get_status()
        assert status.trend == "falling"
        assert status.recommendation == SpawnRecommendation.SPAWN

    def test_should_spawn_reflects_status(self):
        mon = ConfidenceMonitor(threshold=0.9)
        mon.record_round({"a": 0.3})
        assert mon.should_spawn() is True


class TestConfidenceMonitorTrend:
    def test_rising(self):
        mon = ConfidenceMonitor(threshold=0.9, stagnation_delta=0.01)
        mon.record_round({"a": 0.4})
        mon.record_round({"a": 0.6})
        assert mon.get_status().trend == "rising"

    def test_falling(self):
        mon = ConfidenceMonitor(threshold=0.9, stagnation_delta=0.01)
        mon.record_round({"a": 0.6})
        mon.record_round({"a": 0.4})
        assert mon.get_status().trend == "falling"

    def test_stable(self):
        mon = ConfidenceMonitor(threshold=0.9, stagnation_delta=0.05)
        mon.record_round({"a": 0.5})
        mon.record_round({"a": 0.51})
        assert mon.get_status().trend == "stable"


class TestConfidenceMonitorStagnation:
    def test_stagnation_detected(self):
        mon = ConfidenceMonitor(threshold=0.8, stagnation_window=3, stagnation_delta=0.02)
        mon.record_round({"a": 0.50})
        mon.record_round({"a": 0.50})
        mon.record_round({"a": 0.51})
        status = mon.get_status()
        assert status.is_stagnating is True
        assert status.recommendation == SpawnRecommendation.SPAWN

    def test_no_stagnation_if_improving(self):
        mon = ConfidenceMonitor(threshold=0.8, stagnation_window=3, stagnation_delta=0.02)
        mon.record_round({"a": 0.50})
        mon.record_round({"a": 0.55})
        mon.record_round({"a": 0.60})
        assert mon.get_status().is_stagnating is False

    def test_not_enough_data_for_stagnation(self):
        mon = ConfidenceMonitor(stagnation_window=5)
        mon.record_round({"a": 0.5})
        mon.record_round({"a": 0.5})
        assert mon.get_status().is_stagnating is False


class TestConfidenceMonitorCriticalGap:
    def test_large_gap_spawns_immediately(self):
        mon = ConfidenceMonitor(threshold=0.8)
        mon.record_round({"a": 0.3})  # gap = 0.5 > 0.2
        assert mon.get_status().recommendation == SpawnRecommendation.SPAWN


class TestConfidenceMonitorReset:
    def test_reset_clears_state(self):
        mon = ConfidenceMonitor(threshold=0.8)
        mon.record_round({"a": 0.5})
        mon.reset()
        status = mon.get_status()
        assert status.team_average == 0.0
