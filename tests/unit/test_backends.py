"""Tests for the pluggable backend interface and SQLiteBackend."""

from __future__ import annotations

import pytest

from felix_agent_sdk.memory.backends.base import BaseBackend
from felix_agent_sdk.memory.backends.sqlite import SQLiteBackend

_SCHEMA = {
    "name": "TEXT",
    "value": "REAL",
    "tags": "TEXT",
}


class TestBaseBackendContract:
    """Verify that BaseBackend cannot be instantiated directly."""

    def test_abstract(self):
        with pytest.raises(TypeError):
            BaseBackend()  # type: ignore[abstract]


class TestSQLiteBackendInitialization:
    def test_in_memory_default(self):
        backend = SQLiteBackend()
        backend.initialize("test_table", _SCHEMA)
        assert backend.count("test_table") == 0
        backend.close()

    def test_file_backed(self, tmp_path):
        db = str(tmp_path / "test.db")
        backend = SQLiteBackend(db_path=db)
        backend.initialize("t", {"col": "TEXT"})
        backend.store("t", "1", {"col": "hello"})
        backend.close()

        # Reopen and verify persistence
        backend2 = SQLiteBackend(db_path=db)
        backend2.initialize("t", {"col": "TEXT"})
        assert backend2.get("t", "1")["col"] == "hello"
        backend2.close()

    def test_idempotent_init(self, memory_backend):
        memory_backend.initialize("t", _SCHEMA)
        memory_backend.initialize("t", _SCHEMA)  # should not raise
        assert memory_backend.count("t") == 0

    def test_context_manager_closes_connection(self):
        with SQLiteBackend() as backend:
            backend.initialize("t", _SCHEMA)
            backend.store("t", "1", {"name": "x", "value": 1.0, "tags": ""})
        import sqlite3

        with pytest.raises(sqlite3.ProgrammingError):
            backend.get("t", "1")


class TestSQLiteBackendHardening:
    def test_order_by_rejects_injection(self, memory_backend):
        memory_backend.initialize("t", _SCHEMA)
        with pytest.raises(ValueError, match="identifier"):
            memory_backend.query("t", order_by="name; DROP TABLE t--")

    def test_order_by_rejects_unknown_column(self, memory_backend):
        memory_backend.initialize("t", _SCHEMA)
        with pytest.raises(ValueError, match="Unknown order_by"):
            memory_backend.query("t", order_by="not_a_column")

    def test_filter_field_rejects_injection(self, memory_backend):
        memory_backend.initialize("t", _SCHEMA)
        with pytest.raises(ValueError, match="identifier"):
            memory_backend.query("t", filters={"name = '' OR 1=1 --": "x"})

    def test_table_name_rejects_injection(self, memory_backend):
        with pytest.raises(ValueError, match="identifier"):
            memory_backend.initialize("t]; DROP TABLE t--", _SCHEMA)

    def test_multithreaded_access(self, memory_backend):
        import threading

        memory_backend.initialize("t", _SCHEMA)
        errors: list[Exception] = []

        def writer(start: int) -> None:
            try:
                for i in range(start, start + 25):
                    memory_backend.store(
                        "t", f"r{i}", {"name": f"n{i}", "value": float(i), "tags": ""}
                    )
                    memory_backend.get("t", f"r{i}")
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(n * 25,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert memory_backend.count("t") == 100


class TestSQLiteBackendCRUD:
    def test_store_and_get(self, memory_backend):
        memory_backend.initialize("t", _SCHEMA)
        memory_backend.store("t", "r1", {"name": "alpha", "value": 1.0, "tags": "a,b"})
        row = memory_backend.get("t", "r1")
        assert row is not None
        assert row["name"] == "alpha"
        assert row["value"] == 1.0

    def test_upsert(self, memory_backend):
        memory_backend.initialize("t", _SCHEMA)
        memory_backend.store("t", "r1", {"name": "v1", "value": 1.0})
        memory_backend.store("t", "r1", {"name": "v2", "value": 2.0})
        row = memory_backend.get("t", "r1")
        assert row["name"] == "v2"
        assert memory_backend.count("t") == 1

    def test_get_missing(self, memory_backend):
        memory_backend.initialize("t", _SCHEMA)
        assert memory_backend.get("t", "nope") is None

    def test_delete_existing(self, memory_backend):
        memory_backend.initialize("t", _SCHEMA)
        memory_backend.store("t", "r1", {"name": "x"})
        assert memory_backend.delete("t", "r1") is True
        assert memory_backend.get("t", "r1") is None

    def test_delete_missing(self, memory_backend):
        memory_backend.initialize("t", _SCHEMA)
        assert memory_backend.delete("t", "nope") is False

    def test_json_round_trip(self, memory_backend):
        memory_backend.initialize("t", _SCHEMA)
        memory_backend.store("t", "r1", {"name": "x", "tags": ["a", "b"]})
        row = memory_backend.get("t", "r1")
        assert row["tags"] == ["a", "b"]


class TestSQLiteBackendQuery:
    @pytest.fixture(autouse=True)
    def _setup(self, memory_backend):
        self.b = memory_backend
        self.b.initialize("t", _SCHEMA)
        self.b.store("t", "1", {"name": "alpha", "value": 1.0})
        self.b.store("t", "2", {"name": "beta", "value": 2.0})
        self.b.store("t", "3", {"name": "gamma", "value": 3.0})

    def test_no_filter(self):
        assert len(self.b.query("t")) == 3

    def test_exact_match(self):
        rows = self.b.query("t", filters={"name": "beta"})
        assert len(rows) == 1
        assert rows[0]["name"] == "beta"

    def test_gt_operator(self):
        rows = self.b.query("t", filters={"value": {"$gt": 1.5}})
        assert len(rows) == 2

    def test_lt_operator(self):
        rows = self.b.query("t", filters={"value": {"$lt": 2.0}})
        assert len(rows) == 1

    def test_gte_lte_operators(self):
        rows = self.b.query("t", filters={"value": {"$gte": 2.0, "$lte": 3.0}})
        assert len(rows) == 2

    def test_in_operator(self):
        rows = self.b.query("t", filters={"name": {"$in": ["alpha", "gamma"]}})
        assert len(rows) == 2

    def test_contains_operator(self):
        rows = self.b.query("t", filters={"name": {"$contains": "eta"}})
        assert len(rows) == 1

    def test_order_by_asc(self):
        rows = self.b.query("t", order_by="value", ascending=True)
        assert [r["value"] for r in rows] == [1.0, 2.0, 3.0]

    def test_order_by_desc(self):
        rows = self.b.query("t", order_by="value", ascending=False)
        assert [r["value"] for r in rows] == [3.0, 2.0, 1.0]

    def test_limit_offset(self):
        rows = self.b.query("t", order_by="value", ascending=True, limit=2, offset=1)
        assert len(rows) == 2
        assert rows[0]["value"] == 2.0

    def test_count_with_filter(self):
        assert self.b.count("t", filters={"value": {"$gt": 1.5}}) == 2

    def test_count_no_filter(self):
        assert self.b.count("t") == 3


class TestSQLiteBackendTextSearch:
    def test_fts_search(self, memory_backend):
        memory_backend.initialize("docs", {"title": "TEXT", "body": "TEXT"})
        memory_backend.store("docs", "1", {"title": "python guide", "body": "learn python programming"})
        memory_backend.store("docs", "2", {"title": "rust guide", "body": "learn rust systems"})
        memory_backend.store("docs", "3", {"title": "cooking tips", "body": "how to cook pasta"})

        results = memory_backend.search_text("docs", "python", ["title", "body"])
        assert len(results) >= 1
        assert any("python" in str(r.get("title", "")).lower() for r in results)

    def test_like_fallback(self, memory_backend):
        """LIKE search when no FTS table exists (schema has no TEXT cols marked)."""
        memory_backend.initialize("nums", {"val": "INTEGER"})
        # search_text should not crash
        results = memory_backend.search_text("nums", "hello", ["val"])
        assert results == []


class TestSQLiteBackendClose:
    def test_close_then_reopen(self, tmp_path):
        db = str(tmp_path / "close_test.db")
        b = SQLiteBackend(db_path=db)
        b.initialize("t", {"x": "TEXT"})
        b.store("t", "1", {"x": "v"})
        b.close()

        b2 = SQLiteBackend(db_path=db)
        b2.initialize("t", {"x": "TEXT"})
        assert b2.get("t", "1")["x"] == "v"
        b2.close()


class TestOrderByOnDiskColumns:
    def test_order_by_allows_columns_outside_session_schema(self, tmp_path):
        """Schema evolution: a column present on disk but absent from this
        session's (narrower) initialize() schema is a valid order_by."""
        db = str(tmp_path / "evolve.db")
        with SQLiteBackend(db_path=db) as b1:
            b1.initialize("t", {"name": "TEXT", "extra_col": "TEXT"})
            b1.store("t", "1", {"name": "a", "extra_col": "x"})

        with SQLiteBackend(db_path=db) as b2:
            b2.initialize("t", {"name": "TEXT"})
            rows = b2.query("t", order_by="extra_col")
            assert len(rows) == 1

    def test_order_by_unknown_column_still_rejected(self, tmp_path):
        db = str(tmp_path / "evolve2.db")
        with SQLiteBackend(db_path=db) as b:
            b.initialize("t", {"name": "TEXT"})
            with pytest.raises(ValueError, match="Unknown order_by"):
                b.query("t", order_by="never_existed")
