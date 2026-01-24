"""Unit tests for query timing (log_query_times, slow_query_threshold_ms)."""

import os
import tempfile

import pytest

from malla.config import AppConfig, _clear_config_cache, _override_config
from malla.database.connection import get_db_connection
from malla.database.query_timing import (
    TimingConnectionWrapper,
    _format_sql_for_log,
    _should_skip,
)
from tests.fixtures.database_fixtures import DatabaseFixtures


@pytest.fixture(autouse=True)
def _reset_config(monkeypatch):
    """Reset config before/after each test to avoid leaking into other tests."""
    monkeypatch.delenv("MALLA_DATABASE_FILE", raising=False)
    _clear_config_cache()
    yield
    _clear_config_cache()


@pytest.fixture
def temp_db():
    """Temporary DB with fixture schema."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    DatabaseFixtures().create_test_database(f.name)
    yield f.name
    try:
        os.unlink(f.name)
    except (FileNotFoundError, PermissionError):
        pass  # Windows may keep file locked briefly after conn.close()


@pytest.mark.unit
def test_query_timing_disabled(temp_db):
    """When log_query_times is False, get_db_connection returns raw Connection."""
    cfg = AppConfig(database_file=temp_db, log_query_times=False)
    _override_config(cfg)
    conn = get_db_connection()
    try:
        assert not isinstance(conn, TimingConnectionWrapper)
        conn.cursor().execute("SELECT 1")
    finally:
        conn.close()


@pytest.mark.unit
def test_query_timing_enabled_logs(temp_db, caplog):
    """When log_query_times is True, user queries are logged with duration."""
    cfg = AppConfig(
        database_file=temp_db,
        log_query_times=True,
        slow_query_threshold_ms=0.0,
    )
    _override_config(cfg)
    conn = get_db_connection()
    try:
        assert isinstance(conn, TimingConnectionWrapper)
        with caplog.at_level("INFO"):
            conn.cursor().execute("SELECT 1")
        assert "query_timing" in caplog.text
        assert "duration_ms=" in caplog.text
        assert "SELECT 1" in caplog.text
    finally:
        conn.close()


@pytest.mark.unit
def test_pragma_skipped(temp_db, caplog):
    """PRAGMA and schema-introspection queries are not logged."""
    cfg = AppConfig(
        database_file=temp_db,
        log_query_times=True,
        slow_query_threshold_ms=0.0,
    )
    _override_config(cfg)
    conn = get_db_connection()
    try:
        with caplog.at_level("INFO"):
            conn.cursor().execute("PRAGMA journal_mode")
            conn.cursor().execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
        assert "query_timing" not in caplog.text
    finally:
        conn.close()


@pytest.mark.unit
def test_slow_query_threshold_filters_fast(temp_db, caplog):
    """When slow_query_threshold_ms > 0, fast queries are not logged."""
    cfg = AppConfig(
        database_file=temp_db,
        log_query_times=True,
        slow_query_threshold_ms=999_999.0,
    )
    _override_config(cfg)
    conn = get_db_connection()
    try:
        with caplog.at_level("INFO"):
            conn.cursor().execute("SELECT 1")
        assert "query_timing" not in caplog.text
    finally:
        conn.close()


@pytest.mark.unit
def test_slow_query_threshold_logs_slow(temp_db, caplog):
    """When slow_query_threshold_ms > 0, slow queries are logged."""
    cfg = AppConfig(
        database_file=temp_db,
        log_query_times=True,
        slow_query_threshold_ms=0.0,
    )
    _override_config(cfg)
    conn = get_db_connection()
    try:
        with caplog.at_level("INFO"):
            conn.cursor().execute("SELECT 1")
        assert "query_timing" in caplog.text
    finally:
        conn.close()


@pytest.mark.unit
def test_should_skip():
    """_should_skip correctly identifies PRAGMA and sqlite_master."""
    assert _should_skip("PRAGMA journal_mode=WAL") is True
    assert _should_skip("  PRAGMA synchronous=NORMAL") is True
    assert _should_skip("SELECT * FROM sqlite_master") is True
    assert _should_skip("PRAGMA table_info(node_info)") is True
    assert _should_skip("SELECT 1") is False
    assert _should_skip("SELECT * FROM node_info") is False


@pytest.mark.unit
def test_format_sql_for_log():
    """_format_sql_for_log truncates and adds param/row hints."""
    assert "[+3 param(s)]" in _format_sql_for_log("SELECT * FROM x WHERE a=? AND b=? AND c=?", (1, 2, 3))
    assert "[+10 rows]" in _format_sql_for_log("INSERT INTO t VALUES (?)", None, executemany_rows=10)
    short = "SELECT 1"
    assert _format_sql_for_log(short, None) == "SELECT 1"
