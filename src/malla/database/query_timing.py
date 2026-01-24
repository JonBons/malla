"""
Query timing for diagnosing database slowdowns in production.

When enabled via config (log_query_times, slow_query_threshold_ms), wraps
SQLite cursors to log each execute() duration. PRAGMAs and schema-introspection
queries are skipped to reduce noise.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import sqlite3

logger = logging.getLogger(__name__)

# Queries we skip to avoid log noise (connection setup, migrations, introspection)
_SKIP_PATTERNS = (
    re.compile(r"^\s*PRAGMA\s", re.IGNORECASE),
    re.compile(r"sqlite_master", re.IGNORECASE),
    re.compile(r"PRAGMA\s+table_info\s*\(", re.IGNORECASE),
)

_SQL_TRUNCATE = 400  # Max chars of SQL to include in log line


def _should_skip(sql: str) -> bool:
    """Return True if this query should not be logged."""
    s = sql.strip()
    for pat in _SKIP_PATTERNS:
        if pat.search(s):
            return True
    return False


def _format_sql_for_log(
    sql: str,
    parameters: tuple[Any, ...] | list[Any] | None,
    executemany_rows: int | None = None,
) -> str:
    """Produce a short, log-safe representation of the query."""
    if executemany_rows is not None:
        suffix = f" [+{executemany_rows} rows]"
    elif parameters:
        n = len(parameters)
        suffix = f" [+{n} param(s)]"
    else:
        suffix = ""
    # Collapse whitespace for readability, truncate
    collapsed = " ".join(sql.split())
    if len(collapsed) > _SQL_TRUNCATE:
        collapsed = collapsed[:_SQL_TRUNCATE] + "..."
    return collapsed + suffix


def _log_query(
    duration_ms: float,
    sql: str,
    parameters: tuple[Any, ...] | list[Any] | None,
    threshold_ms: float,
    executemany_rows: int | None = None,
) -> None:
    """Log a single query if it meets the threshold."""
    if _should_skip(sql):
        return
    if threshold_ms > 0 and duration_ms < threshold_ms:
        return
    formatted = _format_sql_for_log(sql, parameters, executemany_rows)
    logger.info(
        "query_timing duration_ms=%.2f sql=%s",
        duration_ms,
        formatted,
        extra={"duration_ms": duration_ms, "sql": formatted},
    )


class TimingCursor:
    """Cursor wrapper that times execute/executemany and logs durations."""

    def __init__(
        self,
        cursor: sqlite3.Cursor,
        threshold_ms: float,
    ) -> None:
        self._cursor = cursor
        self._threshold_ms = threshold_ms

    def execute(self, sql: str, parameters: tuple[Any, ...] | list[Any] = ()) -> sqlite3.Cursor:
        t0 = time.perf_counter()
        try:
            return self._cursor.execute(sql, parameters)
        finally:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            _log_query(duration_ms, sql, parameters if parameters else None, self._threshold_ms)

    def executemany(
        self, sql: str, parameters_iter: list[tuple[Any, ...]] | list[list[Any]]
    ) -> sqlite3.Cursor:
        t0 = time.perf_counter()
        try:
            return self._cursor.executemany(sql, parameters_iter)
        finally:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            n = len(parameters_iter) if hasattr(parameters_iter, "__len__") else None
            _log_query(
                duration_ms,
                sql,
                None,
                self._threshold_ms,
                executemany_rows=n,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


class TimingConnectionWrapper:
    """Connection wrapper whose cursor() returns timing-enabled cursors."""

    def __init__(self, conn: sqlite3.Connection, threshold_ms: float) -> None:
        self._conn = conn
        self._threshold_ms = threshold_ms

    def cursor(self, factory: type[sqlite3.Cursor] | None = None) -> TimingCursor:
        c = self._conn.cursor() if factory is None else self._conn.cursor(factory)
        return TimingCursor(c, self._threshold_ms)

    def close(self) -> None:
        self._conn.close()

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def __enter__(self) -> TimingConnectionWrapper:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)
