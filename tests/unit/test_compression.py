"""Tests for the ContextCompressor strategies."""

from __future__ import annotations


import pytest

from felix_agent_sdk.memory.compression import (
    CompressedContext,
    CompressionConfig,
    CompressionLevel,
    CompressionStrategy,
    ContextCompressor,
    RETENTION_RATIOS,
)


# ------------------------------------------------------------------
# Enums & data classes
# ------------------------------------------------------------------


class TestCompressionEnums:
    def test_strategy_values(self):
        assert CompressionStrategy.EXTRACTIVE_SUMMARY.value == "extractive_summary"
        assert len(CompressionStrategy) == 6

    def test_level_values(self):
        assert CompressionLevel.LIGHT.value == "light"
        assert len(CompressionLevel) == 4

    def test_retention_ratios(self):
        assert RETENTION_RATIOS[CompressionLevel.LIGHT] == 0.8
        assert RETENTION_RATIOS[CompressionLevel.MODERATE] == 0.6
        assert RETENTION_RATIOS[CompressionLevel.HEAVY] == 0.4
        assert RETENTION_RATIOS[CompressionLevel.EXTREME] == 0.2


class TestCompressedContext:
    def test_efficiency(self):
        ctx = CompressedContext(
            context_id="x",
            original_size=1000,
            compressed_size=400,
            compression_ratio=0.4,
            strategy_used=CompressionStrategy.EXTRACTIVE_SUMMARY,
            compression_level=CompressionLevel.MODERATE,
            content={"text": "short"},
            metadata={},
            relevance_scores={},
        )
        assert abs(ctx.get_compression_efficiency() - 0.6) < 1e-6

    def test_efficiency_zero_original(self):
        ctx = CompressedContext(
            context_id="x",
            original_size=0,
            compressed_size=0,
            compression_ratio=0.0,
            strategy_used=CompressionStrategy.EXTRACTIVE_SUMMARY,
            compression_level=CompressionLevel.LIGHT,
            content={},
            metadata={},
            relevance_scores={},
        )
        assert ctx.get_compression_efficiency() == 0.0


# ------------------------------------------------------------------
# ContextCompressor
# ------------------------------------------------------------------

_SAMPLE_CONTEXT = {
    "task": "Analyse the impact of renewable energy on grid stability",
    "objective": "Determine correlation between solar adoption and outages",
    "background": (
        "Renewable energy sources have grown significantly over the past decade. "
        "Solar and wind power now account for a substantial portion of electricity generation. "
        "Grid operators must balance supply and demand in real time. "
        "Battery storage technologies are evolving rapidly to address intermittency."
    ),
    "notes": "Check the 2024 NREL report for baseline data.",
    "count": 42,
}


class TestExtractiveSummary:
    def test_reduces_text(self, compressor):
        result = compressor.compress_context(
            _SAMPLE_CONTEXT,
            strategy=CompressionStrategy.EXTRACTIVE_SUMMARY,
        )
        assert result.compressed_size <= result.original_size
        assert result.strategy_used == CompressionStrategy.EXTRACTIVE_SUMMARY
        assert "background" in result.content

    def test_preserves_non_string_values(self, compressor):
        result = compressor.compress_context(
            _SAMPLE_CONTEXT,
            strategy=CompressionStrategy.EXTRACTIVE_SUMMARY,
        )
        assert result.content["count"] == 42


class TestKeywordExtraction:
    def test_extracts_keywords(self, compressor):
        result = compressor.compress_context(
            _SAMPLE_CONTEXT,
            strategy=CompressionStrategy.KEYWORD_EXTRACTION,
        )
        assert "keywords" in result.content
        assert isinstance(result.content["keywords"], list)
        assert len(result.content["keywords"]) > 0

    def test_extracts_concepts(self, compressor):
        result = compressor.compress_context(
            {"text": "The NREL report uses CamelCase identifiers and snake_case variables"},
            strategy=CompressionStrategy.KEYWORD_EXTRACTION,
        )
        concepts = result.content.get("key_concepts", [])
        assert isinstance(concepts, list)


class TestHierarchicalSummary:
    def test_three_levels(self, compressor):
        result = compressor.compress_context(
            _SAMPLE_CONTEXT,
            strategy=CompressionStrategy.HIERARCHICAL_SUMMARY,
        )
        assert "core" in result.content
        assert "supporting" in result.content
        assert "auxiliary" in result.content
        assert result.metadata["levels"] == 3

    def test_core_preserves_task_keys(self, compressor):
        result = compressor.compress_context(
            _SAMPLE_CONTEXT,
            strategy=CompressionStrategy.HIERARCHICAL_SUMMARY,
        )
        core = result.content["core"]
        assert "task" in core
        assert "objective" in core


class TestRelevanceFiltering:
    def test_filters_by_relevance(self, compressor):
        result = compressor.compress_context(
            _SAMPLE_CONTEXT,
            strategy=CompressionStrategy.RELEVANCE_FILTERING,
        )
        assert result.strategy_used == CompressionStrategy.RELEVANCE_FILTERING
        # Non-string values preserved
        assert result.content.get("count") == 42


class TestProgressiveRefinement:
    def test_multiple_passes(self, compressor):
        # Use a very small target to force multiple passes
        result = compressor.compress_context(
            _SAMPLE_CONTEXT,
            target_size=50,
            strategy=CompressionStrategy.PROGRESSIVE_REFINEMENT,
        )
        assert result.metadata["method"] == "progressive_refinement"
        assert result.metadata["passes"] > 0

    def test_no_passes_needed(self, compressor):
        # Target larger than content = 0 passes
        result = compressor.compress_context(
            {"short": "hi"},
            target_size=100000,
            strategy=CompressionStrategy.PROGRESSIVE_REFINEMENT,
        )
        assert result.metadata["passes"] == 0


class TestAbstractiveSummary:
    def test_not_implemented(self, compressor):
        with pytest.raises(NotImplementedError):
            compressor.compress_context(
                _SAMPLE_CONTEXT,
                strategy=CompressionStrategy.ABSTRACTIVE_SUMMARY,
            )


class TestDecompressContext:
    def test_returns_content_with_metadata(self, compressor):
        compressed = compressor.compress_context(
            _SAMPLE_CONTEXT,
            strategy=CompressionStrategy.HIERARCHICAL_SUMMARY,
        )
        result = compressor.decompress_context(compressed)
        assert "_compression_metadata" in result
        meta = result["_compression_metadata"]
        assert meta["strategy_used"] == "hierarchical_summary"
        assert compressed.access_count == 1


class TestCompressionStats:
    def test_returns_config(self, compressor):
        stats = compressor.get_compression_stats()
        assert stats["default_strategy"] == "hierarchical_summary"
        assert stats["default_level"] == "moderate"


class TestEdgeCases:
    def test_empty_context(self, compressor):
        result = compressor.compress_context(
            {},
            strategy=CompressionStrategy.EXTRACTIVE_SUMMARY,
        )
        assert result.content == {}

    def test_short_text(self, compressor):
        result = compressor.compress_context(
            {"text": "Hi"},
            strategy=CompressionStrategy.EXTRACTIVE_SUMMARY,
        )
        assert "text" in result.content

    def test_custom_config(self):
        config = CompressionConfig(
            max_context_size=1000,
            strategy=CompressionStrategy.KEYWORD_EXTRACTION,
            level=CompressionLevel.HEAVY,
            preserve_keywords=["renewable", "energy"],
            relevance_threshold=0.5,
        )
        c = ContextCompressor(config=config)
        result = c.compress_context(_SAMPLE_CONTEXT)
        assert result.strategy_used == CompressionStrategy.KEYWORD_EXTRACTION
