"""Knowledge storage system for the Felix Agent SDK.

Persistent storage and retrieval of knowledge entries with confidence tags,
domain organisation, relationship tracking, and full-text search.

Algorithms ported from CalebisGross/felix ``src/memory/knowledge_store.py``.
Refactored to use the pluggable :class:`BaseBackend` interface.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

from felix_agent_sdk.memory.backends.base import BaseBackend
from felix_agent_sdk.memory.backends.sqlite import SQLiteBackend

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------


class KnowledgeType(Enum):
    """Types of knowledge that can be stored."""

    TASK_RESULT = "task_result"
    AGENT_INSIGHT = "agent_insight"
    PATTERN_RECOGNITION = "pattern_recognition"
    FAILURE_ANALYSIS = "failure_analysis"
    OPTIMIZATION_DATA = "optimization_data"
    DOMAIN_EXPERTISE = "domain_expertise"
    TOOL_INSTRUCTION = "tool_instruction"
    FILE_LOCATION = "file_location"


class ConfidenceLevel(Enum):
    """Confidence levels for knowledge entries."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"


# Numeric ordering for confidence comparisons
_CONFIDENCE_ORDER: dict[ConfidenceLevel, int] = {
    ConfidenceLevel.LOW: 0,
    ConfidenceLevel.MEDIUM: 1,
    ConfidenceLevel.HIGH: 2,
    ConfidenceLevel.VERIFIED: 3,
}


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class KnowledgeEntry:
    """Single entry in the knowledge base."""

    knowledge_id: str
    knowledge_type: KnowledgeType
    content: dict[str, Any]
    confidence_level: ConfidenceLevel
    source_agent: str
    domain: str
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    success_rate: float = 1.0
    related_entries: list[str] = field(default_factory=list)
    is_deleted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat dict suitable for backend storage."""
        data = asdict(self)
        data["knowledge_type"] = self.knowledge_type.value
        data["confidence_level"] = self.confidence_level.value
        data["content"] = json.dumps(self.content)
        data["tags"] = json.dumps(self.tags)
        data["related_entries"] = json.dumps(self.related_entries)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeEntry:
        """Reconstruct from a backend dict."""
        d = dict(data)
        # Pop backend internal key
        d.pop("_id", None)
        d["knowledge_type"] = KnowledgeType(d["knowledge_type"])
        d["confidence_level"] = ConfidenceLevel(d["confidence_level"])
        if isinstance(d.get("content"), str):
            d["content"] = json.loads(d["content"])
        if isinstance(d.get("tags"), str):
            d["tags"] = json.loads(d["tags"])
        if isinstance(d.get("related_entries"), str):
            d["related_entries"] = json.loads(d["related_entries"])
        if "is_deleted" not in d:
            d["is_deleted"] = False
        elif isinstance(d["is_deleted"], int):
            d["is_deleted"] = bool(d["is_deleted"])
        return cls(**d)


@dataclass
class KnowledgeQuery:
    """Query structure for knowledge retrieval."""

    knowledge_types: Optional[list[KnowledgeType]] = None
    domains: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    min_confidence: Optional[ConfidenceLevel] = None
    min_success_rate: Optional[float] = None
    content_keywords: Optional[list[str]] = None
    time_range: Optional[tuple[float, float]] = None
    search_text: Optional[str] = None
    limit: int = 100

    def build_filters(self) -> dict[str, Any]:
        """Translate query fields into backend filter dict."""
        filters: dict[str, Any] = {"is_deleted": 0}

        if self.knowledge_types:
            filters["knowledge_type"] = {"$in": [kt.value for kt in self.knowledge_types]}
        if self.domains:
            filters["domain"] = {"$in": self.domains}
        if self.min_confidence is not None:
            min_level = _CONFIDENCE_ORDER[self.min_confidence]
            valid = [
                level.value for level, order in _CONFIDENCE_ORDER.items() if order >= min_level
            ]
            filters["confidence_level"] = {"$in": valid}
        if self.min_success_rate is not None:
            filters["success_rate"] = {"$gte": self.min_success_rate}
        if self.time_range:
            filters["created_at"] = {"$gte": self.time_range[0], "$lte": self.time_range[1]}

        return filters


# ------------------------------------------------------------------
# Schema definitions
# ------------------------------------------------------------------

_ENTRIES_SCHEMA: dict[str, str] = {
    "knowledge_type": "TEXT",
    "content": "TEXT",
    "confidence_level": "TEXT",
    "source_agent": "TEXT",
    "domain": "TEXT",
    "tags": "TEXT",
    "created_at": "REAL",
    "updated_at": "REAL",
    "access_count": "INTEGER",
    "success_rate": "REAL",
    "related_entries": "TEXT",
    "is_deleted": "INTEGER",
}

_RELATIONSHIPS_SCHEMA: dict[str, str] = {
    "source_id": "TEXT",
    "target_id": "TEXT",
    "relationship_type": "TEXT",
    "confidence": "REAL",
    "created_at": "REAL",
}

_TABLE = "knowledge_entries"
_REL_TABLE = "knowledge_relationships"


# ------------------------------------------------------------------
# KnowledgeStore
# ------------------------------------------------------------------


class KnowledgeStore:
    """Persistent knowledge storage with pluggable backend.

    Args:
        backend: Storage backend. Defaults to an in-memory
            :class:`SQLiteBackend` if not supplied.
    """

    def __init__(self, backend: Optional[BaseBackend] = None) -> None:
        self._backend = backend or SQLiteBackend()
        self._backend.initialize(_TABLE, _ENTRIES_SCHEMA)
        self._backend.initialize(_REL_TABLE, _RELATIONSHIPS_SCHEMA)

    # ------------------------------------------------------------------
    # ID generation (ported from Felix)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_id(content: dict[str, Any], source_agent: str, domain: str) -> str:
        content_str = json.dumps(content, sort_keys=True)
        hash_input = f"{domain}:{content_str}:{source_agent}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_entry(
        self,
        knowledge_type: KnowledgeType,
        content: dict[str, Any],
        confidence_level: ConfidenceLevel,
        source_agent: str,
        domain: str,
        tags: Optional[list[str]] = None,
    ) -> str:
        """Store a new knowledge entry. Returns the knowledge ID."""
        tags = tags or []
        knowledge_id = self._generate_id(content, source_agent, domain)
        now = time.time()

        entry = KnowledgeEntry(
            knowledge_id=knowledge_id,
            knowledge_type=knowledge_type,
            content=content,
            confidence_level=confidence_level,
            source_agent=source_agent,
            domain=domain,
            tags=tags,
            created_at=now,
            updated_at=now,
        )

        self._backend.store(_TABLE, knowledge_id, entry.to_dict())
        return knowledge_id

    def get_entry_by_id(self, knowledge_id: str) -> Optional[KnowledgeEntry]:
        """Retrieve a single entry by ID."""
        data = self._backend.get(_TABLE, knowledge_id)
        if data is None:
            return None
        entry = KnowledgeEntry.from_dict(data)
        if entry.is_deleted:
            return None
        self._increment_access_count(knowledge_id)
        return entry

    def update_entry(self, knowledge_id: str, updates: dict[str, Any]) -> bool:
        """Update fields on an existing entry.

        *updates* may contain: ``content``, ``confidence_level``, ``domain``,
        ``tags``, ``success_rate``.
        """
        data = self._backend.get(_TABLE, knowledge_id)
        if data is None:
            return False

        entry = KnowledgeEntry.from_dict(data)
        if entry.is_deleted:
            return False

        if "content" in updates:
            entry.content = updates["content"]
        if "confidence_level" in updates:
            level = updates["confidence_level"]
            entry.confidence_level = (
                level if isinstance(level, ConfidenceLevel) else ConfidenceLevel(level)
            )
        if "domain" in updates:
            entry.domain = updates["domain"]
        if "tags" in updates:
            entry.tags = updates["tags"]
        if "success_rate" in updates:
            entry.success_rate = updates["success_rate"]

        entry.updated_at = time.time()
        self._backend.store(_TABLE, knowledge_id, entry.to_dict())
        return True

    def delete_entry(self, knowledge_id: str) -> bool:
        """Soft-delete a knowledge entry."""
        data = self._backend.get(_TABLE, knowledge_id)
        if data is None:
            return False
        entry = KnowledgeEntry.from_dict(data)
        entry.is_deleted = True
        entry.updated_at = time.time()
        self._backend.store(_TABLE, knowledge_id, entry.to_dict())
        return True

    def batch_add(
        self,
        entries: list[dict[str, Any]],
    ) -> list[str]:
        """Add multiple entries. Each dict must contain the fields expected
        by :meth:`add_entry`. Returns a list of knowledge IDs."""
        ids: list[str] = []
        for e in entries:
            kid = self.add_entry(
                knowledge_type=e["knowledge_type"],
                content=e["content"],
                confidence_level=e["confidence_level"],
                source_agent=e["source_agent"],
                domain=e["domain"],
                tags=e.get("tags"),
            )
            ids.append(kid)
        return ids

    # ------------------------------------------------------------------
    # Query / search
    # ------------------------------------------------------------------

    def search(self, query: KnowledgeQuery) -> list[KnowledgeEntry]:
        """Retrieve entries matching *query*."""
        if query.search_text:
            rows = self._backend.search_text(
                _TABLE,
                query.search_text,
                ["content", "domain", "tags"],
                limit=query.limit,
            )
        else:
            rows = self._backend.query(
                _TABLE,
                filters=query.build_filters(),
                order_by="created_at",
                ascending=False,
                limit=query.limit,
            )

        entries: list[KnowledgeEntry] = []
        for row in rows:
            entry = KnowledgeEntry.from_dict(row)
            if entry.is_deleted:
                continue
            # Content-keyword post-filter
            if query.content_keywords:
                content_str = json.dumps(entry.content).lower()
                if not any(kw.lower() in content_str for kw in query.content_keywords):
                    continue
            entries.append(entry)

        return entries

    def get_entries_by_type(
        self, knowledge_type: KnowledgeType, limit: int = 100
    ) -> list[KnowledgeEntry]:
        """Retrieve entries of a specific type."""
        return self.search(KnowledgeQuery(knowledge_types=[knowledge_type], limit=limit))

    def get_entries_by_domain(self, domain: str, limit: int = 100) -> list[KnowledgeEntry]:
        """Retrieve entries in a specific domain."""
        return self.search(KnowledgeQuery(domains=[domain], limit=limit))

    def semantic_search(
        self,
        query_embedding: list[float],
        limit: int = 10,
    ) -> list[KnowledgeEntry]:
        """Semantic / embedding-based search.

        .. note::
            Not implemented in the SDK yet. Requires an embedding
            provider integration (future phase).
        """
        raise NotImplementedError(
            "Semantic search requires an embedding provider; "
            "use search() with text queries instead."
        )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str = "related",
        confidence: float = 0.7,
    ) -> bool:
        """Create a relationship between two entries."""
        rel_id = hashlib.sha256(
            f"{source_id}:{target_id}:{relationship_type}".encode()
        ).hexdigest()[:16]

        self._backend.store(
            _REL_TABLE,
            rel_id,
            {
                "source_id": source_id,
                "target_id": target_id,
                "relationship_type": relationship_type,
                "confidence": confidence,
                "created_at": time.time(),
            },
        )

        # Also update the related_entries list on the source entry
        data = self._backend.get(_TABLE, source_id)
        if data:
            entry = KnowledgeEntry.from_dict(data)
            if target_id not in entry.related_entries:
                entry.related_entries.append(target_id)
                self._backend.store(_TABLE, source_id, entry.to_dict())
        return True

    def get_relationships(self, knowledge_id: str) -> list[dict[str, Any]]:
        """Get all relationships for an entry (bidirectional)."""
        outgoing = self._backend.query(_REL_TABLE, filters={"source_id": knowledge_id})
        incoming = self._backend.query(_REL_TABLE, filters={"target_id": knowledge_id})
        return outgoing + incoming

    # ------------------------------------------------------------------
    # Success rate
    # ------------------------------------------------------------------

    def update_success_rate(self, knowledge_id: str, success_rate: float) -> bool:
        """Update the success rate for a knowledge entry."""
        return self.update_entry(knowledge_id, {"success_rate": success_rate})

    # ------------------------------------------------------------------
    # Cleanup / summary
    # ------------------------------------------------------------------

    def cleanup_old_entries(self, max_age_days: int = 90) -> int:
        """Hard-delete soft-deleted entries older than *max_age_days*."""
        cutoff = time.time() - max_age_days * 86400
        deleted = self._backend.query(
            _TABLE,
            filters={"is_deleted": 1, "updated_at": {"$lt": cutoff}},
        )
        for row in deleted:
            self._backend.delete(_TABLE, row["_id"])
        return len(deleted)

    def get_summary(self) -> dict[str, Any]:
        """Aggregate statistics about the knowledge store."""
        total = self._backend.count(_TABLE, filters={"is_deleted": 0})
        deleted = self._backend.count(_TABLE, filters={"is_deleted": 1})
        relationships = self._backend.count(_REL_TABLE)
        return {
            "total_entries": total,
            "deleted_entries": deleted,
            "relationships": relationships,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _increment_access_count(self, knowledge_id: str) -> None:
        data = self._backend.get(_TABLE, knowledge_id)
        if data is None:
            return
        entry = KnowledgeEntry.from_dict(data)
        entry.access_count += 1
        self._backend.store(_TABLE, knowledge_id, entry.to_dict())
