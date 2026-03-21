"""Tests for the KnowledgeStore memory system."""

from __future__ import annotations


import pytest

from felix_agent_sdk.memory.knowledge_store import (
    ConfidenceLevel,
    KnowledgeEntry,
    KnowledgeQuery,
    KnowledgeType,
)


# ------------------------------------------------------------------
# KnowledgeEntry
# ------------------------------------------------------------------


class TestKnowledgeEntry:
    def test_construction(self):
        entry = KnowledgeEntry(
            knowledge_id="abc123",
            knowledge_type=KnowledgeType.AGENT_INSIGHT,
            content={"text": "insight"},
            confidence_level=ConfidenceLevel.HIGH,
            source_agent="research-1",
            domain="testing",
        )
        assert entry.knowledge_id == "abc123"
        assert entry.success_rate == 1.0
        assert entry.is_deleted is False

    def test_to_dict_from_dict_roundtrip(self):
        entry = KnowledgeEntry(
            knowledge_id="rt1",
            knowledge_type=KnowledgeType.TASK_RESULT,
            content={"key": "value", "nested": [1, 2]},
            confidence_level=ConfidenceLevel.VERIFIED,
            source_agent="agent-x",
            domain="science",
            tags=["alpha", "beta"],
            related_entries=["other-1"],
        )
        d = entry.to_dict()
        restored = KnowledgeEntry.from_dict(d)
        assert restored.knowledge_id == entry.knowledge_id
        assert restored.knowledge_type == entry.knowledge_type
        assert restored.content == entry.content
        assert restored.tags == entry.tags
        assert restored.related_entries == entry.related_entries


class TestKnowledgeTypes:
    def test_all_types_exist(self):
        expected = {
            "task_result", "agent_insight", "pattern_recognition",
            "failure_analysis", "optimization_data", "domain_expertise",
            "tool_instruction", "file_location",
        }
        assert {t.value for t in KnowledgeType} == expected

    def test_confidence_ordering(self):
        levels = list(ConfidenceLevel)
        assert levels == [
            ConfidenceLevel.LOW,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.HIGH,
            ConfidenceLevel.VERIFIED,
        ]


# ------------------------------------------------------------------
# KnowledgeStore CRUD
# ------------------------------------------------------------------


class TestKnowledgeStoreCRUD:
    def test_add_and_retrieve(self, knowledge_store):
        kid = knowledge_store.add_entry(
            knowledge_type=KnowledgeType.AGENT_INSIGHT,
            content={"text": "hello world"},
            confidence_level=ConfidenceLevel.HIGH,
            source_agent="agent-1",
            domain="test",
        )
        entry = knowledge_store.get_entry_by_id(kid)
        assert entry is not None
        assert entry.content["text"] == "hello world"
        assert entry.domain == "test"

    def test_deterministic_id(self, knowledge_store):
        """Same content + agent + domain → same ID (dedup)."""
        kid1 = knowledge_store.add_entry(
            knowledge_type=KnowledgeType.TASK_RESULT,
            content={"a": 1},
            confidence_level=ConfidenceLevel.LOW,
            source_agent="agt",
            domain="d",
        )
        kid2 = knowledge_store.add_entry(
            knowledge_type=KnowledgeType.TASK_RESULT,
            content={"a": 1},
            confidence_level=ConfidenceLevel.LOW,
            source_agent="agt",
            domain="d",
        )
        assert kid1 == kid2

    def test_update_entry(self, knowledge_store):
        kid = knowledge_store.add_entry(
            knowledge_type=KnowledgeType.DOMAIN_EXPERTISE,
            content={"old": True},
            confidence_level=ConfidenceLevel.MEDIUM,
            source_agent="a",
            domain="d",
        )
        assert knowledge_store.update_entry(kid, {"content": {"new": True}})
        entry = knowledge_store.get_entry_by_id(kid)
        assert entry.content == {"new": True}

    def test_update_confidence(self, knowledge_store):
        kid = knowledge_store.add_entry(
            knowledge_type=KnowledgeType.AGENT_INSIGHT,
            content={"x": 1},
            confidence_level=ConfidenceLevel.LOW,
            source_agent="a",
            domain="d",
        )
        knowledge_store.update_entry(kid, {"confidence_level": ConfidenceLevel.VERIFIED})
        entry = knowledge_store.get_entry_by_id(kid)
        assert entry.confidence_level == ConfidenceLevel.VERIFIED

    def test_update_nonexistent(self, knowledge_store):
        assert knowledge_store.update_entry("nope", {"domain": "x"}) is False

    def test_soft_delete(self, knowledge_store):
        kid = knowledge_store.add_entry(
            knowledge_type=KnowledgeType.TASK_RESULT,
            content={"x": 1},
            confidence_level=ConfidenceLevel.LOW,
            source_agent="a",
            domain="d",
        )
        assert knowledge_store.delete_entry(kid) is True
        assert knowledge_store.get_entry_by_id(kid) is None

    def test_delete_nonexistent(self, knowledge_store):
        assert knowledge_store.delete_entry("nope") is False

    def test_batch_add(self, knowledge_store):
        entries = [
            {
                "knowledge_type": KnowledgeType.AGENT_INSIGHT,
                "content": {"n": i},
                "confidence_level": ConfidenceLevel.MEDIUM,
                "source_agent": f"a-{i}",
                "domain": "batch",
            }
            for i in range(5)
        ]
        ids = knowledge_store.batch_add(entries)
        assert len(ids) == 5


# ------------------------------------------------------------------
# KnowledgeStore queries
# ------------------------------------------------------------------


class TestKnowledgeStoreQueries:
    @pytest.fixture(autouse=True)
    def _seed(self, knowledge_store):
        self.store = knowledge_store
        self.store.add_entry(
            KnowledgeType.AGENT_INSIGHT, {"text": "insight one"},
            ConfidenceLevel.HIGH, "agent-1", "science", tags=["physics"],
        )
        self.store.add_entry(
            KnowledgeType.TASK_RESULT, {"text": "result two"},
            ConfidenceLevel.LOW, "agent-2", "engineering",
        )
        self.store.add_entry(
            KnowledgeType.AGENT_INSIGHT, {"text": "insight three"},
            ConfidenceLevel.VERIFIED, "agent-3", "science", tags=["chemistry"],
        )

    def test_by_type(self):
        entries = self.store.get_entries_by_type(KnowledgeType.AGENT_INSIGHT)
        assert len(entries) == 2

    def test_by_domain(self):
        entries = self.store.get_entries_by_domain("science")
        assert len(entries) == 2

    def test_min_confidence(self):
        entries = self.store.search(
            KnowledgeQuery(min_confidence=ConfidenceLevel.HIGH)
        )
        assert all(
            e.confidence_level in (ConfidenceLevel.HIGH, ConfidenceLevel.VERIFIED)
            for e in entries
        )

    def test_content_keywords(self):
        entries = self.store.search(
            KnowledgeQuery(content_keywords=["insight"])
        )
        assert len(entries) == 2

    def test_text_search(self):
        entries = self.store.search(
            KnowledgeQuery(search_text="insight")
        )
        assert len(entries) >= 1

    def test_empty_result(self):
        entries = self.store.search(
            KnowledgeQuery(domains=["nonexistent"])
        )
        assert entries == []


# ------------------------------------------------------------------
# Relationships
# ------------------------------------------------------------------


class TestKnowledgeStoreRelationships:
    def test_add_and_get_relationship(self, knowledge_store):
        kid1 = knowledge_store.add_entry(
            KnowledgeType.AGENT_INSIGHT, {"x": 1},
            ConfidenceLevel.HIGH, "a", "d",
        )
        kid2 = knowledge_store.add_entry(
            KnowledgeType.AGENT_INSIGHT, {"x": 2},
            ConfidenceLevel.HIGH, "b", "d",
        )
        knowledge_store.add_relationship(kid1, kid2, "supports", confidence=0.9)

        rels = knowledge_store.get_relationships(kid1)
        assert len(rels) >= 1
        assert any(r["target_id"] == kid2 for r in rels)

    def test_bidirectional_lookup(self, knowledge_store):
        kid1 = knowledge_store.add_entry(
            KnowledgeType.TASK_RESULT, {"a": 1},
            ConfidenceLevel.LOW, "a", "d",
        )
        kid2 = knowledge_store.add_entry(
            KnowledgeType.TASK_RESULT, {"b": 2},
            ConfidenceLevel.LOW, "b", "d",
        )
        knowledge_store.add_relationship(kid1, kid2)
        # Query from the target side
        rels = knowledge_store.get_relationships(kid2)
        assert len(rels) >= 1


# ------------------------------------------------------------------
# Success rate / cleanup / summary
# ------------------------------------------------------------------


class TestKnowledgeStoreUtilities:
    def test_update_success_rate(self, knowledge_store):
        kid = knowledge_store.add_entry(
            KnowledgeType.OPTIMIZATION_DATA, {"x": 1},
            ConfidenceLevel.MEDIUM, "a", "d",
        )
        knowledge_store.update_success_rate(kid, 0.42)
        entry = knowledge_store.get_entry_by_id(kid)
        assert abs(entry.success_rate - 0.42) < 1e-6

    def test_cleanup_old_entries(self, knowledge_store):
        kid = knowledge_store.add_entry(
            KnowledgeType.TASK_RESULT, {"old": True},
            ConfidenceLevel.LOW, "a", "d",
        )
        knowledge_store.delete_entry(kid)
        # With max_age_days=0 it should purge immediately
        removed = knowledge_store.cleanup_old_entries(max_age_days=0)
        assert removed >= 1

    def test_summary(self, knowledge_store):
        knowledge_store.add_entry(
            KnowledgeType.AGENT_INSIGHT, {"x": 1},
            ConfidenceLevel.HIGH, "a", "d",
        )
        summary = knowledge_store.get_summary()
        assert summary["total_entries"] >= 1
        assert "relationships" in summary

    def test_semantic_search_not_implemented(self, knowledge_store):
        with pytest.raises(NotImplementedError):
            knowledge_store.semantic_search([0.1, 0.2, 0.3])
