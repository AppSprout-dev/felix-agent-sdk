"""Abstract storage backend for Felix memory systems.

Backends provide typed table-like storage. Each 'table' is identified
by a string name. Records are dicts with a required 'id' key.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseBackend(ABC):
    """Abstract base class for memory storage backends.

    Subclasses implement persistence for KnowledgeStore and TaskMemory.
    The interface is intentionally high-level (CRUD + operators) so that
    non-SQL backends (Redis, in-memory, cloud) can implement it without
    SQL translation.
    """

    # Supported filter operators for query()
    OPERATORS = ("$gt", "$lt", "$gte", "$lte", "$in", "$contains")

    @abstractmethod
    def initialize(self, table: str, schema: dict[str, str]) -> None:
        """Create or verify a table/collection.

        Args:
            table: Logical table name (e.g. "knowledge_entries").
            schema: Column name -> type hint mapping. Not enforced at the
                backend level but used for indexing hints.
        """

    @abstractmethod
    def store(self, table: str, record_id: str, data: dict[str, Any]) -> None:
        """Insert or update (upsert) a record."""

    @abstractmethod
    def get(self, table: str, record_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a single record by ID. Returns ``None`` if missing."""

    @abstractmethod
    def query(
        self,
        table: str,
        filters: Optional[dict[str, Any]] = None,
        order_by: Optional[str] = None,
        ascending: bool = True,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query records matching filters.

        Filters is ``{field: value}`` for exact match, or
        ``{field: {"$op": value}}`` for operators
        (``$gt``, ``$lt``, ``$gte``, ``$lte``, ``$in``, ``$contains``).
        """

    @abstractmethod
    def delete(self, table: str, record_id: str) -> bool:
        """Delete a record. Returns ``True`` if it existed."""

    @abstractmethod
    def count(self, table: str, filters: Optional[dict[str, Any]] = None) -> int:
        """Count records matching filters."""

    @abstractmethod
    def search_text(
        self,
        table: str,
        query: str,
        fields: list[str],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Full-text search across *fields*. Returns records ranked by relevance."""

    def close(self) -> None:
        """Release resources. Default is a no-op."""
