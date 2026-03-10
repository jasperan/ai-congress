"""Tests for the Stare Decisis (Precedent-Based Reasoning) system."""
import pytest
import numpy as np


class TestPrecedentSchema:
    """Test that CONGRESS_PRECEDENTS DDL is defined."""

    def test_precedents_ddl_exists(self):
        from src.ai_congress.datalake.schema import DDL_STATEMENTS
        ddl_text = "\n".join(DDL_STATEMENTS)
        assert "CONGRESS_PRECEDENTS" in ddl_text

    def test_precedents_has_vector_column(self):
        from src.ai_congress.datalake.schema import DDL_STATEMENTS
        ddl_text = "\n".join(DDL_STATEMENTS)
        assert "VECTOR(384" in ddl_text

    def test_precedents_has_superseded_by(self):
        from src.ai_congress.datalake.schema import DDL_STATEMENTS
        ddl_text = "\n".join(DDL_STATEMENTS)
        assert "superseded_by" in ddl_text

    def test_precedents_indexes_exist(self):
        from src.ai_congress.datalake.schema import INDEX_STATEMENTS
        idx_text = "\n".join(INDEX_STATEMENTS)
        assert "idx_precedents_domain" in idx_text


from unittest.mock import AsyncMock, MagicMock


class TestPrecedentStore:
    """Test PrecedentStore without Oracle connection."""

    def test_import(self):
        from src.ai_congress.core.precedent.precedent_store import PrecedentStore, Precedent
        assert PrecedentStore is not None
        assert Precedent is not None

    def test_precedent_dataclass(self):
        from src.ai_congress.core.precedent.precedent_store import Precedent
        p = Precedent(
            id="abc-123",
            session_id="sess-1",
            query_text="What is 2+2?",
            ruling_text="4",
            domain="math",
            consensus=0.95,
            models_used=["phi3", "mistral"],
            debate_rounds=1,
            similarity=0.88,
            created_at="2026-03-10T00:00:00",
        )
        assert p.id == "abc-123"
        assert p.consensus == 0.95
        assert p.similarity == 0.88

    def test_precedent_to_dict(self):
        from src.ai_congress.core.precedent.precedent_store import Precedent
        p = Precedent(
            id="abc", session_id="s1", query_text="q", ruling_text="r",
            domain="math", consensus=0.9, models_used=["m1"],
        )
        d = p.to_dict()
        assert d["id"] == "abc"
        assert d["domain"] == "math"
        assert isinstance(d["models_used"], list)

    @pytest.mark.asyncio
    async def test_store_precedent_generates_embedding(self):
        from src.ai_congress.core.precedent.precedent_store import PrecedentStore

        mock_pool = MagicMock()
        mock_pool.is_available = True
        mock_cursor = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)
        mock_pool.pool.acquire.return_value = mock_conn

        mock_embedder = MagicMock()
        mock_embedder.generate_embedding.return_value = np.zeros(384, dtype=np.float32)

        store = PrecedentStore(mock_pool, mock_embedder)
        pid = await store.store_precedent(
            session_id="sess-1",
            query_text="What is 2+2?",
            ruling_text="The answer is 4.",
            domain="math",
            consensus=0.95,
            models_used=["phi3"],
            vote_data={"winner": "phi3"},
            debate_rounds=1,
        )
        assert pid is not None
        assert len(pid) > 0
        mock_embedder.generate_embedding.assert_called_once_with("What is 2+2?")

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_unavailable(self):
        from src.ai_congress.core.precedent.precedent_store import PrecedentStore

        mock_pool = MagicMock()
        mock_pool.is_available = False
        mock_embedder = MagicMock()

        store = PrecedentStore(mock_pool, mock_embedder)
        results = await store.search_precedents("What is 2+2?")
        assert results == []

    @pytest.mark.asyncio
    async def test_supersede_sets_superseded_by(self):
        from src.ai_congress.core.precedent.precedent_store import PrecedentStore

        mock_pool = MagicMock()
        mock_pool.is_available = True
        mock_cursor = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)
        mock_pool.pool.acquire.return_value = mock_conn

        mock_embedder = MagicMock()

        store = PrecedentStore(mock_pool, mock_embedder)
        await store.supersede("old-id", "new-id")
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args
        assert "superseded_by" in call_args[0][0]
