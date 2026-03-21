"""Tests for WorkflowSynthesizer."""

from __future__ import annotations

from unittest.mock import MagicMock


from felix_agent_sdk.agents.llm_agent import LLMResult
from felix_agent_sdk.providers.base import BaseProvider
from felix_agent_sdk.providers.types import CompletionResult
from felix_agent_sdk.workflows.config import SynthesisStrategy, WorkflowConfig
from felix_agent_sdk.workflows.synthesizer import WorkflowSynthesizer


def _make_result(agent_id: str, content: str, confidence: float) -> LLMResult:
    """Create a mock LLMResult for testing."""
    return LLMResult(
        agent_id=agent_id,
        task_id="test",
        content=content,
        position_info={"phase": "exploration", "progress": 0.5},
        completion_result=CompletionResult(
            content=content,
            model="test",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        ),
        processing_time=0.1,
        confidence=confidence,
        temperature_used=0.5,
        token_budget_used=20,
    )


def _make_mock_provider() -> BaseProvider:
    provider = MagicMock(spec=BaseProvider)
    provider.complete.return_value = CompletionResult(
        content="Synthesised final output combining all findings.",
        model="test-model",
        usage={"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
    )
    provider.count_tokens.return_value = 50
    return provider


class TestBestResultStrategy:
    def test_picks_highest_confidence(self):
        provider = _make_mock_provider()
        config = WorkflowConfig(synthesis_strategy=SynthesisStrategy.BEST_RESULT)
        synth = WorkflowSynthesizer(provider, config)

        results = [
            _make_result("a1", "Low confidence output", 0.3),
            _make_result("a2", "High confidence output", 0.9),
            _make_result("a3", "Medium confidence output", 0.6),
        ]

        output = synth.synthesize(results, "test task")
        assert output == "High confidence output"

    def test_does_not_call_provider(self):
        provider = _make_mock_provider()
        config = WorkflowConfig(synthesis_strategy=SynthesisStrategy.BEST_RESULT)
        synth = WorkflowSynthesizer(provider, config)

        results = [_make_result("a1", "content", 0.5)]
        synth.synthesize(results, "task")
        provider.complete.assert_not_called()


class TestRoundRobinStrategy:
    def test_returns_last_result(self):
        provider = _make_mock_provider()
        config = WorkflowConfig(synthesis_strategy=SynthesisStrategy.ROUND_ROBIN)
        synth = WorkflowSynthesizer(provider, config)

        results = [
            _make_result("a1", "First", 0.5),
            _make_result("a2", "Second", 0.6),
            _make_result("a3", "Third", 0.7),
        ]

        output = synth.synthesize(results, "task")
        assert output == "Third"


class TestCompressedMergeStrategy:
    def test_calls_provider(self):
        provider = _make_mock_provider()
        config = WorkflowConfig(synthesis_strategy=SynthesisStrategy.COMPRESSED_MERGE)
        synth = WorkflowSynthesizer(provider, config)

        results = [
            _make_result("a1", "Finding one", 0.6),
            _make_result("a2", "Finding two", 0.7),
        ]

        output = synth.synthesize(results, "Analyse the data")
        assert provider.complete.called
        assert len(output) > 0

    def test_without_compression(self):
        provider = _make_mock_provider()
        config = WorkflowConfig(
            synthesis_strategy=SynthesisStrategy.COMPRESSED_MERGE,
            enable_context_compression=False,
        )
        synth = WorkflowSynthesizer(provider, config)

        results = [_make_result("a1", "Data", 0.5)]
        synth.synthesize(results, "task")
        assert provider.complete.called


class TestEmptyResults:
    def test_empty_returns_empty(self):
        provider = _make_mock_provider()
        config = WorkflowConfig()
        synth = WorkflowSynthesizer(provider, config)
        assert synth.synthesize([], "task") == ""
