"""Context compression system for the Felix Agent SDK.

Intelligent summarisation and compression of large contexts through
extractive, keyword, hierarchical, relevance, and progressive strategies.

Algorithms ported from CalebisGross/felix ``src/memory/context_compression.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------


class CompressionStrategy(Enum):
    """Available compression strategies."""

    EXTRACTIVE_SUMMARY = "extractive_summary"
    ABSTRACTIVE_SUMMARY = "abstractive_summary"
    KEYWORD_EXTRACTION = "keyword_extraction"
    HIERARCHICAL_SUMMARY = "hierarchical_summary"
    RELEVANCE_FILTERING = "relevance_filtering"
    PROGRESSIVE_REFINEMENT = "progressive_refinement"


class CompressionLevel(Enum):
    """Compression intensity levels with retention ratios."""

    LIGHT = "light"  # 80 %
    MODERATE = "moderate"  # 60 %
    HEAVY = "heavy"  # 40 %
    EXTREME = "extreme"  # 20 %


RETENTION_RATIOS: dict[CompressionLevel, float] = {
    CompressionLevel.LIGHT: 0.8,
    CompressionLevel.MODERATE: 0.6,
    CompressionLevel.HEAVY: 0.4,
    CompressionLevel.EXTREME: 0.2,
}


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class CompressedContext:
    """Container for compressed context data."""

    context_id: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    strategy_used: CompressionStrategy
    compression_level: CompressionLevel
    content: dict[str, Any]
    metadata: dict[str, Any]
    relevance_scores: dict[str, float]
    created_at: float = field(default_factory=time.time)
    access_count: int = 0

    def get_compression_efficiency(self) -> float:
        """Higher is better (1.0 = full reduction)."""
        if self.original_size == 0:
            return 0.0
        return 1.0 - (self.compressed_size / self.original_size)


@dataclass
class CompressionConfig:
    """Configuration for context compression."""

    max_context_size: int = 4000
    strategy: CompressionStrategy = CompressionStrategy.HIERARCHICAL_SUMMARY
    level: CompressionLevel = CompressionLevel.MODERATE
    preserve_keywords: list[str] = field(default_factory=list)
    preserve_structure: bool = True
    maintain_coherence: bool = True
    relevance_threshold: float = 0.3


# ------------------------------------------------------------------
# Stopwords (shared with keyword extraction)
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
    "where",
    "much",
    "your",
    "from",
    "they",
    "know",
    "want",
    "been",
    "good",
    "much",
    "some",
    "time",
    "very",
    "when",
    "come",
    "here",
    "just",
    "like",
    "long",
    "make",
    "many",
    "over",
    "such",
    "take",
    "than",
    "them",
    "well",
    "were",
    "will",
    "with",
}


# ------------------------------------------------------------------
# ContextCompressor
# ------------------------------------------------------------------


class ContextCompressor:
    """Intelligent context compression with strategy dispatch.

    Args:
        config: Compression configuration. Uses defaults if ``None``.
    """

    def __init__(self, config: Optional[CompressionConfig] = None) -> None:
        self.config = config or CompressionConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress_context(
        self,
        context: dict[str, Any],
        target_size: Optional[int] = None,
        strategy: Optional[CompressionStrategy] = None,
    ) -> CompressedContext:
        """Compress *context* using the specified (or default) strategy."""
        target_size = target_size or self.config.max_context_size
        strategy = strategy or self.config.strategy

        original_size = len(json.dumps(context))

        dispatch = {
            CompressionStrategy.EXTRACTIVE_SUMMARY: self._extractive_summary,
            CompressionStrategy.ABSTRACTIVE_SUMMARY: self._abstractive_summary,
            CompressionStrategy.KEYWORD_EXTRACTION: self._keyword_extraction,
            CompressionStrategy.HIERARCHICAL_SUMMARY: self._hierarchical_summary,
            CompressionStrategy.RELEVANCE_FILTERING: self._relevance_filtering,
            CompressionStrategy.PROGRESSIVE_REFINEMENT: self._progressive_refinement,
        }

        handler = dispatch.get(strategy, self._hierarchical_summary)
        result = handler(context, target_size)

        compressed_size = len(json.dumps(result["content"]))
        ratio = compressed_size / original_size if original_size > 0 else 0.0

        return CompressedContext(
            context_id=self._generate_context_id(context),
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=ratio,
            strategy_used=strategy,
            compression_level=self.config.level,
            content=result["content"],
            metadata=result["metadata"],
            relevance_scores=result["relevance_scores"],
        )

    def decompress_context(self, compressed: CompressedContext) -> dict[str, Any]:
        """Return compressed content with compression metadata attached."""
        compressed.access_count += 1
        result = dict(compressed.content)
        result["_compression_metadata"] = {
            "original_size": compressed.original_size,
            "compressed_size": compressed.compressed_size,
            "compression_ratio": compressed.compression_ratio,
            "strategy_used": compressed.strategy_used.value,
            "compression_level": compressed.compression_level.value,
            "relevance_scores": compressed.relevance_scores,
        }
        return result

    def get_compression_stats(self) -> dict[str, Any]:
        """Return current configuration as stats."""
        return {
            "max_context_size": self.config.max_context_size,
            "default_strategy": self.config.strategy.value,
            "default_level": self.config.level.value,
            "preserve_keywords": len(self.config.preserve_keywords),
            "relevance_threshold": self.config.relevance_threshold,
        }

    # ------------------------------------------------------------------
    # Strategies (ported from Felix)
    # ------------------------------------------------------------------

    def _extractive_summary(self, context: dict[str, Any], target_size: int) -> dict[str, Any]:
        """Extract most important sentences/sections."""
        content: dict[str, Any] = {}
        metadata: dict[str, Any] = {"method": "extractive_summary"}
        relevance_scores: dict[str, float] = {}

        for key, value in context.items():
            if isinstance(value, str):
                sentences = self._split_into_sentences(value)
                scored = [(s, self._calculate_sentence_importance(s, context)) for s in sentences]
                for i, (_, score) in enumerate(scored):
                    relevance_scores[f"{key}_{i}"] = score

                scored.sort(key=lambda x: x[1], reverse=True)
                target_n = max(1, len(scored) // 3)
                content[key] = " ".join(s for s, _ in scored[:target_n])
            else:
                content[key] = value

        return {"content": content, "metadata": metadata, "relevance_scores": relevance_scores}

    def _abstractive_summary(self, context: dict[str, Any], target_size: int) -> dict[str, Any]:
        """Abstractive summary — requires an LLM provider (not yet wired).

        Raises :class:`NotImplementedError` so callers know to pick
        another strategy or integrate a provider.
        """
        raise NotImplementedError(
            "Abstractive summary requires a provider integration; "
            "use a different CompressionStrategy."
        )

    def _keyword_extraction(self, context: dict[str, Any], target_size: int) -> dict[str, Any]:
        """Extract key terms and keyword-rich sentences."""
        content: dict[str, Any] = {}
        metadata: dict[str, Any] = {"method": "keyword_extraction"}
        relevance_scores: dict[str, float] = {}

        all_text = " ".join(str(v) for v in context.values() if isinstance(v, str))
        keywords = self._extract_keywords(all_text)

        content["keywords"] = keywords[:20]
        content["key_concepts"] = self._extract_key_concepts(all_text)

        for key, value in context.items():
            if isinstance(value, str):
                sentences = self._split_into_sentences(value)
                keyword_sentences: list[str] = []
                for sentence in sentences:
                    count = sum(1 for kw in keywords[:10] if kw.lower() in sentence.lower())
                    if count > 0:
                        keyword_sentences.append(sentence)
                        relevance_scores[f"{key}_sentence"] = count / max(len(keywords[:10]), 1)
                if keyword_sentences:
                    content[f"{key}_summary"] = " ".join(keyword_sentences[:3])
            else:
                content[key] = value

        return {"content": content, "metadata": metadata, "relevance_scores": relevance_scores}

    def _hierarchical_summary(self, context: dict[str, Any], target_size: int) -> dict[str, Any]:
        """Three-level summary: core, supporting, auxiliary."""
        content: dict[str, Any] = {}
        metadata: dict[str, Any] = {"method": "hierarchical_summary", "levels": 3}
        relevance_scores: dict[str, float] = {}

        core_keys = {"task", "objective", "requirements", "constraints"}
        core_info: dict[str, Any] = {}
        for key, value in context.items():
            if key in core_keys:
                core_info[key] = value
                relevance_scores[f"core_{key}"] = 1.0
        content["core"] = core_info

        supporting_info: dict[str, Any] = {}
        for key, value in context.items():
            if key not in core_info and isinstance(value, str):
                if len(value) > 100:
                    supporting_info[key] = self._create_brief_summary(value)
                    relevance_scores[f"support_{key}"] = 0.7
                else:
                    supporting_info[key] = value
                    relevance_scores[f"support_{key}"] = 0.8
        content["supporting"] = supporting_info

        auxiliary_info: dict[str, Any] = {}
        for key, value in context.items():
            if key not in core_info and key not in supporting_info:
                auxiliary_info[key] = value
                relevance_scores[f"aux_{key}"] = 0.5
        content["auxiliary"] = auxiliary_info

        return {"content": content, "metadata": metadata, "relevance_scores": relevance_scores}

    def _relevance_filtering(self, context: dict[str, Any], target_size: int) -> dict[str, Any]:
        """Filter content by relevance to main topics."""
        content: dict[str, Any] = {}
        metadata: dict[str, Any] = {"method": "relevance_filtering"}
        relevance_scores: dict[str, float] = {}

        main_topics = self._identify_main_topics(context)

        for key, value in context.items():
            if isinstance(value, str):
                relevance = self._calculate_relevance_to_topics(value, main_topics)
                relevance_scores[key] = relevance
                if relevance >= self.config.relevance_threshold:
                    content[key] = value
                elif relevance >= self.config.relevance_threshold * 0.5:
                    content[f"{key}_brief"] = self._create_brief_summary(value)
            else:
                content[key] = value
                relevance_scores[key] = 0.8

        return {"content": content, "metadata": metadata, "relevance_scores": relevance_scores}

    def _progressive_refinement(self, context: dict[str, Any], target_size: int) -> dict[str, Any]:
        """Multiple compression passes for optimal size."""
        metadata: dict[str, Any] = {"method": "progressive_refinement", "passes": 0}
        relevance_scores: dict[str, float] = {}

        current = dict(context)
        passes = 0
        max_passes = 3
        strategies = [
            self._relevance_filtering,
            self._hierarchical_summary,
            self._keyword_extraction,
        ]

        while len(json.dumps(current)) > target_size and passes < max_passes:
            result = strategies[passes](current, target_size)
            current = result["content"]
            relevance_scores.update(result["relevance_scores"])
            passes += 1

        metadata["passes"] = passes
        return {"content": current, "metadata": metadata, "relevance_scores": relevance_scores}

    # ------------------------------------------------------------------
    # Text helpers (ported from Felix)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_context_id(content: dict[str, Any]) -> str:
        content_str = json.dumps(content, sort_keys=True)
        hash_input = f"{content_str}:{time.time()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_sentence_importance(self, sentence: str, context: dict[str, Any]) -> float:
        score = 0.0
        words = sentence.lower().split()

        # Length bonus
        if 10 <= len(words) <= 25:
            score += 0.2

        # Keyword presence
        for word in words:
            if word in self.config.preserve_keywords:
                score += 0.3

        # Numbers / technical terms
        if any(c.isdigit() for c in sentence):
            score += 0.1
        if any(w.isupper() for w in words):
            score += 0.1

        return score

    def _create_brief_summary(self, text: str) -> str:
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return text

        summary = sentences[0]
        key_info: list[str] = []
        for sentence in sentences[1:]:
            numbers = re.findall(r"\b\d+(?:\.\d+)?\b", sentence)
            key_info.extend(numbers)
            caps = re.findall(r"\b[A-Z][A-Za-z]*\b", sentence)
            key_info.extend(caps[:2])

        if key_info:
            unique = list(set(key_info))[:5]
            summary += f" Key details: {', '.join(unique)}"
        return summary

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        words = re.findall(r"\b\w{4,}\b", text.lower())
        keywords = [w for w in words if w not in _STOPWORDS]
        freq: dict[str, int] = {}
        for w in keywords:
            freq[w] = freq.get(w, 0) + 1
        return sorted(freq, key=lambda x: freq[x], reverse=True)[:30]

    @staticmethod
    def _extract_key_concepts(text: str) -> list[str]:
        patterns = [
            r"\b[A-Z]{2,}\b",
            r"\b\w+[A-Z]\w*\b",
            r"\b\w+_\w+\b",
            r"\b\d+\.?\d*[a-zA-Z]+\b",
        ]
        concepts: list[str] = []
        for pat in patterns:
            concepts.extend(re.findall(pat, text))
        return list(set(concepts))[:15]

    def _identify_main_topics(self, context: dict[str, Any]) -> list[str]:
        priority_keys = ("task", "objective", "goal", "purpose", "requirements")
        topics: list[str] = []
        for key in priority_keys:
            if key in context and isinstance(context[key], str):
                topics.extend(self._extract_keywords(context[key])[:5])
        if not topics:
            all_text = " ".join(str(v) for v in context.values() if isinstance(v, str))
            topics = self._extract_keywords(all_text)[:10]
        return topics

    @staticmethod
    def _calculate_relevance_to_topics(text: str, topics: list[str]) -> float:
        if not topics:
            return 0.5
        text_lower = text.lower()
        matches = sum(1 for t in topics if t.lower() in text_lower)
        relevance = matches / len(topics)
        total_mentions = sum(text_lower.count(t.lower()) for t in topics)
        if total_mentions > matches:
            relevance += 0.1 * (total_mentions - matches)
        return min(1.0, relevance)
