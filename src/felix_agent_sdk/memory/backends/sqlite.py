"""SQLite storage backend with FTS5 full-text search.

Default backend for Felix memory systems. Uses a single SQLite database
file (or ``:memory:`` for testing) with dynamic schema initialisation,
operator-based filter translation, and FTS5 for text search.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from typing import Any, Optional

from felix_agent_sdk.memory.backends.base import BaseBackend

logger = logging.getLogger(__name__)

# Maps operator tokens to SQL comparison operators
_OP_SQL = {
    "$gt": ">",
    "$lt": "<",
    "$gte": ">=",
    "$lte": "<=",
}

# Column/table names are interpolated into SQL and must be plain identifiers.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str, context: str) -> None:
    """Reject anything that is not a bare SQL identifier.

    Raises:
        ValueError: If *name* contains characters outside [A-Za-z0-9_]
            or does not start with a letter/underscore.
    """
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid SQL identifier for {context}: {name!r}")


class SQLiteBackend(BaseBackend):
    """SQLite-backed storage with FTS5 support.

    A single connection is shared across threads; all access is serialised
    through an internal lock, so the backend is safe for multi-threaded use
    (though writes from many threads will contend on that lock).

    Can be used as a context manager to guarantee the connection is closed.

    Args:
        db_path: Path to the database file, or ``":memory:"`` for an
            in-memory database (the default).
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        except Exception:
            self._conn.close()
            raise
        self._fts_tables: set[str] = set()
        self._fts_text_cols: dict[str, list[str]] = {}
        self._initialized_tables: set[str] = set()
        self._table_columns: dict[str, set[str]] = {}

    def __enter__(self) -> SQLiteBackend:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def initialize(self, table: str, schema: dict[str, str]) -> None:
        with self._lock:
            self._initialize_locked(table, schema)

    def _initialize_locked(self, table: str, schema: dict[str, str]) -> None:
        if table in self._initialized_tables:
            return

        _validate_identifier(table, "table name")
        for col_name in schema:
            _validate_identifier(col_name, "column name")

        # Build column definitions from schema hints
        col_defs = ["_id TEXT PRIMARY KEY"]
        for col_name, col_type in schema.items():
            if col_name == "_id":
                continue
            sql_type = self._map_type(col_type)
            col_defs.append(f"{col_name} {sql_type}")

        self._conn.execute(f"CREATE TABLE IF NOT EXISTS [{table}] ({', '.join(col_defs)})")

        # Create indices on commonly filtered columns
        for col_name in schema:
            if col_name == "_id":
                continue
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS [idx_{table}_{col_name}] ON [{table}]({col_name})"
            )

        # Standalone FTS5 table for text search (not content-sync)
        text_cols = [c for c, t in schema.items() if t in ("TEXT", "text")]
        if text_cols:
            fts_name = f"{table}_fts"
            fts_col_defs = ", ".join(["record_id"] + text_cols)
            try:
                self._conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS [{fts_name}] "
                    f"USING fts5({fts_col_defs}, tokenize='porter unicode61')"
                )
                self._fts_tables.add(table)
                self._fts_text_cols[table] = text_cols
            except sqlite3.OperationalError:
                logger.debug("FTS5 not available; text search will use LIKE")

        self._conn.commit()
        self._initialized_tables.add(table)
        self._table_columns[table] = {"_id"} | set(schema.keys())

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def store(self, table: str, record_id: str, data: dict[str, Any]) -> None:
        data_copy = dict(data)
        data_copy["_id"] = record_id

        # Serialise non-scalar values to JSON
        for key, val in data_copy.items():
            if isinstance(val, (dict, list)):
                data_copy[key] = json.dumps(val)

        # Filter to only columns that exist in the table
        valid_cols = self._table_columns.get(table)
        if valid_cols:
            data_copy = {k: v for k, v in data_copy.items() if k in valid_cols}
        else:
            for col_name in data_copy:
                _validate_identifier(col_name, "column name")

        cols = list(data_copy.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)

        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO [{table}] ({col_names}) VALUES ({placeholders})",
                [data_copy[c] for c in cols],
            )

            # Sync FTS
            if table in self._fts_tables:
                self._sync_fts(table, record_id, data_copy)

            self._conn.commit()

    def get(self, table: str, record_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(f"SELECT * FROM [{table}] WHERE _id = ?", (record_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_dict(cur.description, row)

    def delete(self, table: str, record_id: str) -> bool:
        with self._lock:
            # Delete FTS entry first
            if table in self._fts_tables:
                fts_name = f"{table}_fts"
                self._conn.execute(
                    f"DELETE FROM [{fts_name}] WHERE record_id = ?",
                    (record_id,),
                )

            cur = self._conn.execute(f"DELETE FROM [{table}] WHERE _id = ?", (record_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        table: str,
        filters: Optional[dict[str, Any]] = None,
        order_by: Optional[str] = None,
        ascending: bool = True,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        sql = f"SELECT * FROM [{table}]"
        params: list[Any] = []

        where_parts = self._build_where(filters, params)
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)

        if order_by:
            _validate_identifier(order_by, "order_by")
            known_cols = self._table_columns.get(table)
            if known_cols and order_by not in known_cols:
                raise ValueError(
                    f"Unknown order_by column {order_by!r} for table {table!r}"
                )
            direction = "ASC" if ascending else "DESC"
            sql += f" ORDER BY [{order_by}] {direction}"

        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
            if offset:
                sql += " OFFSET ?"
                params.append(offset)

        with self._lock:
            cur = self._conn.execute(sql, params)
            return [self._row_to_dict(cur.description, row) for row in cur.fetchall()]

    def count(self, table: str, filters: Optional[dict[str, Any]] = None) -> int:
        sql = f"SELECT COUNT(*) FROM [{table}]"
        params: list[Any] = []

        where_parts = self._build_where(filters, params)
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)

        with self._lock:
            return int(self._conn.execute(sql, params).fetchone()[0])

    def search_text(
        self,
        table: str,
        query: str,
        fields: list[str],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if table in self._fts_tables:
            return self._fts_search(table, query, limit)
        return self._like_search(table, query, fields, limit)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_type(hint: str) -> str:
        hint_lower = hint.lower()
        if hint_lower in ("int", "integer", "bool"):
            return "INTEGER"
        if hint_lower in ("float", "real", "number"):
            return "REAL"
        return "TEXT"

    def _build_where(
        self,
        filters: Optional[dict[str, Any]],
        params: list[Any],
    ) -> list[str]:
        if not filters:
            return []

        parts: list[str] = []
        for field, value in filters.items():
            _validate_identifier(field, "filter field")
            if isinstance(value, dict):
                # Operator filter: {"$gt": 5}
                for op, op_val in value.items():
                    if op == "$in":
                        placeholders = ", ".join(["?"] * len(op_val))
                        parts.append(f"{field} IN ({placeholders})")
                        params.extend(op_val)
                    elif op == "$contains":
                        parts.append(f"{field} LIKE ?")
                        params.append(f"%{op_val}%")
                    elif op in _OP_SQL:
                        parts.append(f"{field} {_OP_SQL[op]} ?")
                        params.append(op_val)
            else:
                parts.append(f"{field} = ?")
                params.append(value)
        return parts

    def _sync_fts(self, table: str, record_id: str, data: dict[str, Any]) -> None:
        """Sync standalone FTS index after an upsert."""
        fts_name = f"{table}_fts"
        text_cols = self._fts_text_cols.get(table, [])
        if not text_cols:
            return

        # Delete old FTS row
        self._conn.execute(
            f"DELETE FROM [{fts_name}] WHERE record_id = ?",
            (record_id,),
        )

        # Insert new FTS row
        vals = [record_id] + [str(data.get(c, "")) for c in text_cols]
        col_names = ", ".join(["record_id"] + text_cols)
        placeholders = ", ".join(["?"] * len(vals))
        self._conn.execute(
            f"INSERT INTO [{fts_name}] ({col_names}) VALUES ({placeholders})",
            vals,
        )

    def _fts_search(self, table: str, query: str, limit: int) -> list[dict[str, Any]]:
        fts_name = f"{table}_fts"
        sql = (
            f"SELECT t.* FROM [{table}] t "
            f"JOIN [{fts_name}] f ON t._id = f.record_id "
            f"WHERE [{fts_name}] MATCH ? "
            f"ORDER BY rank LIMIT ?"
        )
        with self._lock:
            cur = self._conn.execute(sql, (query, limit))
            return [self._row_to_dict(cur.description, row) for row in cur.fetchall()]

    def _like_search(
        self,
        table: str,
        query: str,
        fields: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        # Filter to columns that actually exist
        valid = self._table_columns.get(table, set())
        safe_fields = [f for f in fields if f in valid]
        if not safe_fields:
            return []
        conditions = [f"{f} LIKE ?" for f in safe_fields]
        sql = f"SELECT * FROM [{table}] WHERE " + " OR ".join(conditions) + " LIMIT ?"
        params = [f"%{query}%"] * len(safe_fields) + [limit]
        with self._lock:
            cur = self._conn.execute(sql, params)
            return [self._row_to_dict(cur.description, row) for row in cur.fetchall()]

    @staticmethod
    def _row_to_dict(description: Any, row: tuple[Any, ...]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for i, col_info in enumerate(description):
            col_name = col_info[0]
            value = row[i]
            # Try to deserialise JSON strings
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, (dict, list)):
                        value = parsed
                except ValueError:
                    pass
            result[col_name] = value
        return result
