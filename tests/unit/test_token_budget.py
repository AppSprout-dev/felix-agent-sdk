"""Tests for felix_agent_sdk.tokens.budget — TokenBudget."""

from __future__ import annotations

import pytest

from felix_agent_sdk.core.helix import HelixGeometry, HelixPosition
from felix_agent_sdk.tokens.budget import TokenBudget


@pytest.fixture
def helix():
    return HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)


class TestTokenBudgetConstruction:
    def test_fields(self):
        b = TokenBudget(max_input_tokens=1000, max_output_tokens=500, reserved_tokens=64)
        assert b.max_input_tokens == 1000
        assert b.max_output_tokens == 500
        assert b.reserved_tokens == 64

    def test_total_property(self):
        b = TokenBudget(max_input_tokens=1000, max_output_tokens=500)
        assert b.total == 1500

    def test_frozen(self):
        b = TokenBudget(max_input_tokens=1, max_output_tokens=1)
        with pytest.raises(AttributeError):
            b.max_input_tokens = 99  # type: ignore[misc]


class TestTokenBudgetDefault:
    def test_default_returns_budget(self):
        b = TokenBudget.default()
        assert b.max_input_tokens > 0
        assert b.max_output_tokens > 0
        assert b.total > 0


class TestTokenBudgetFromHelixPosition:
    def test_exploration_favours_input(self, helix):
        pos = HelixPosition(helix, t=0.0)
        b = TokenBudget.from_helix_position(pos)
        assert b.max_input_tokens > b.max_output_tokens

    def test_synthesis_favours_output(self, helix):
        pos = HelixPosition(helix, t=1.0)
        b = TokenBudget.from_helix_position(pos)
        assert b.max_output_tokens > b.max_input_tokens

    def test_midpoint_balanced(self, helix):
        pos = HelixPosition(helix, t=0.5)
        b = TokenBudget.from_helix_position(pos)
        # At t=0.5: input_share = 0.55, output_share = 0.45
        assert b.max_input_tokens > b.max_output_tokens

    def test_output_increases_monotonically(self, helix):
        outputs = []
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            b = TokenBudget.from_helix_position(HelixPosition(helix, t))
            outputs.append(b.max_output_tokens)
        assert outputs == sorted(outputs)

    def test_total_within_context_window(self, helix):
        window = 128_000
        for t in [0.0, 0.5, 1.0]:
            b = TokenBudget.from_helix_position(
                HelixPosition(helix, t), model_context_window=window
            )
            assert b.total + b.reserved_tokens <= window

    def test_custom_context_window(self, helix):
        pos = HelixPosition(helix, t=0.0)
        b = TokenBudget.from_helix_position(pos, model_context_window=8192)
        assert b.total + b.reserved_tokens <= 8192

    def test_reserved_tokens_respected(self, helix):
        pos = HelixPosition(helix, t=0.5)
        b = TokenBudget.from_helix_position(pos, reserved_tokens=1024)
        assert b.reserved_tokens == 1024
