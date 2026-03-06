"""Tests for the data lake module."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ai_congress.datalake.connection import OraclePoolManager
from ai_congress.datalake.logger import EventLogger, Event
from ai_congress.datalake.schema import DDL_STATEMENTS, INDEX_STATEMENTS, SCHEMA_VERSION


class TestOraclePoolManager:
    def test_defaults_from_env(self):
        with patch.dict("os.environ", {}, clear=False):
            pm = OraclePoolManager()
            assert pm.host == "localhost"
            assert pm.port == 1521
            assert pm.service == "FREEPDB1"
            assert pm.pool_min == 2
            assert pm.pool_max == 5
            assert pm.dsn == "localhost:1521/FREEPDB1"

    def test_custom_values(self):
        pm = OraclePoolManager(
            host="dbhost", port=1522, service="PDB1",
            user="myuser", password="pass", pool_min=1, pool_max=3,
        )
        assert pm.dsn == "dbhost:1522/PDB1"
        assert pm.user == "myuser"
        assert pm.pool_min == 1

    def test_env_overrides(self):
        env = {
            "ORACLE_HOST": "remotehost",
            "ORACLE_PORT": "1523",
            "ORACLE_SERVICE": "MYPDB",
            "ORACLE_USER": "envuser",
            "ORACLE_PASSWORD": "envpass",
            "ORACLE_POOL_MIN": "3",
            "ORACLE_POOL_MAX": "10",
        }
        with patch.dict("os.environ", env, clear=False):
            pm = OraclePoolManager()
            assert pm.host == "remotehost"
            assert pm.port == 1523
            assert pm.service == "MYPDB"
            assert pm.user == "envuser"
            assert pm.pool_min == 3
            assert pm.pool_max == 10

    def test_not_available_before_start(self):
        pm = OraclePoolManager()
        assert not pm.is_available
        assert pm.pool is None


class TestEventLogger:
    def setup_method(self):
        self.pm = OraclePoolManager()
        self.pm._pool = None  # Not connected
        self.logger = EventLogger(self.pm)

    def test_new_session_returns_uuid(self):
        sid = self.logger.new_session()
        assert len(sid) == 36  # UUID format
        assert "-" in sid

    def test_log_queues_event(self):
        self.logger.log("test_event", "session1", foo="bar")
        assert not self.logger._queue.empty()
        event = self.logger._queue.get_nowait()
        assert event.event_type == "test_event"
        assert event.session_id == "session1"
        assert event.event_data == {"foo": "bar"}

    def test_log_without_session(self):
        self.logger.log("standalone_event", key="value")
        event = self.logger._queue.get_nowait()
        assert event.session_id == ""
        assert event.event_data == {"key": "value"}

    def test_log_does_not_raise_on_full_queue(self):
        """Queue overflow should not crash."""
        small_logger = EventLogger(self.pm, batch_size=5)
        small_logger._queue = asyncio.Queue(maxsize=2)
        # Fill queue
        small_logger.log("e1")
        small_logger.log("e2")
        # This should not raise
        small_logger.log("e3_overflow")
        assert small_logger._queue.qsize() == 2  # Still 2

    def test_sequence_counter_increments(self):
        assert self.logger._next_seq("s1") == 1
        assert self.logger._next_seq("s1") == 2
        assert self.logger._next_seq("s2") == 1
        assert self.logger._next_seq("s1") == 3


class TestEventDataclass:
    def test_event_defaults(self):
        e = Event(event_type="test")
        assert e.event_type == "test"
        assert e.session_id == ""
        assert e.event_data == {}
        assert len(e.id) == 36
        assert e.created_at > 0


class TestEventLoggerAsync:
    @pytest.fixture
    def mock_pool_manager(self):
        pm = OraclePoolManager()
        mock_pool = MagicMock()

        # conn.cursor() is sync, but cursor.execute() and conn.commit() are async
        mock_cursor = MagicMock()
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[])

        mock_conn = MagicMock()
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_conn.commit = AsyncMock()

        # pool.acquire() returns an async context manager
        acm = MagicMock()
        acm.__aenter__ = AsyncMock(return_value=mock_conn)
        acm.__aexit__ = AsyncMock(return_value=None)
        mock_pool.acquire = MagicMock(return_value=acm)
        pm._pool = mock_pool
        pm._mock_conn = mock_conn
        pm._mock_cursor = mock_cursor
        return pm

    @pytest.mark.asyncio
    async def test_log_session(self, mock_pool_manager):
        logger = EventLogger(mock_pool_manager)
        await logger.log_session("sid1", "hello", "multi_model", "semantic", ["m1", "m2"])
        mock_pool_manager._pool.acquire.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_model_response(self, mock_pool_manager):
        logger = EventLogger(mock_pool_manager)
        await logger.log_model_response("sid1", "mistral:7b", 0.7, "response", 500)
        mock_pool_manager._pool.acquire.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_vote(self, mock_pool_manager):
        logger = EventLogger(mock_pool_manager)
        await logger.log_vote("sid1", "semantic", "mistral:7b", 0.85, 2, {"clusters": []})
        mock_pool_manager._pool.acquire.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_debate_round(self, mock_pool_manager):
        logger = EventLogger(mock_pool_manager)
        await logger.log_debate_round("sid1", 1, "mistral:7b", "response", 1, False, 1.2)
        mock_pool_manager._pool.acquire.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_session_silently_fails_when_unavailable(self):
        """Logging should not raise when Oracle is down."""
        pm = OraclePoolManager()
        pm._pool = None
        logger = EventLogger(pm)
        # Should not raise
        await logger.log_session("sid", "prompt", "mode", "classic", [])
        await logger.log_model_response("sid", "m", 0.7, "r", 100)
        await logger.log_vote("sid", "classic", "m", 0.5)
        await logger.log_debate_round("sid", 1, "m", "r")

    @pytest.mark.asyncio
    async def test_flush_batch_writes_to_oracle(self, mock_pool_manager):
        logger = EventLogger(mock_pool_manager)
        logger.log("event1", "s1", data="val1")
        logger.log("event2", "s1", data="val2")
        await logger._flush_batch()
        # Should have acquired connection and committed
        mock_pool_manager._mock_conn.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_batch_noop_when_empty(self, mock_pool_manager):
        logger = EventLogger(mock_pool_manager)
        await logger._flush_batch()
        mock_pool_manager._pool.acquire.assert_not_called()


class TestSchema:
    def test_ddl_has_all_tables(self):
        table_names = []
        for ddl in DDL_STATEMENTS:
            name = ddl.split("CREATE TABLE")[1].split("(")[0].strip()
            table_names.append(name)

        assert "CONGRESS_META" in table_names
        assert "CONGRESS_SESSIONS" in table_names
        assert "CONGRESS_EVENTS" in table_names
        assert "CONGRESS_MODEL_RESPONSES" in table_names
        assert "CONGRESS_VOTES" in table_names
        assert "CONGRESS_DEBATES" in table_names

    def test_schema_version_defined(self):
        assert SCHEMA_VERSION == "1.0.0"

    def test_indexes_reference_congress_tables(self):
        for idx in INDEX_STATEMENTS:
            assert "CONGRESS_" in idx
