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


class TestPrecedentInjector:
    """Test PrecedentInjector logic without LLM calls."""

    def test_import(self):
        from src.ai_congress.core.precedent.precedent_injector import (
            PrecedentInjector, PrecedentAction,
        )
        assert PrecedentInjector is not None

    def test_no_precedent_action(self):
        from src.ai_congress.core.precedent.precedent_injector import (
            PrecedentInjector, PrecedentAction,
        )
        injector = PrecedentInjector()
        action = injector.classify_action([])
        assert action == PrecedentAction.NO_PRECEDENT

    def test_soft_cite_action(self):
        from src.ai_congress.core.precedent.precedent_injector import (
            PrecedentInjector, PrecedentAction,
        )
        from src.ai_congress.core.precedent.precedent_store import Precedent

        p = Precedent(
            id="p1", session_id="s1", query_text="q", ruling_text="r",
            domain="general", consensus=0.80, models_used=["m1"],
            similarity=0.82,
        )
        injector = PrecedentInjector()
        action = injector.classify_action([p])
        assert action == PrecedentAction.SOFT_CITE

    def test_fast_follow_action(self):
        from src.ai_congress.core.precedent.precedent_injector import (
            PrecedentInjector, PrecedentAction,
        )
        from src.ai_congress.core.precedent.precedent_store import Precedent

        p = Precedent(
            id="p1", session_id="s1", query_text="q", ruling_text="r",
            domain="math", consensus=0.95, models_used=["m1"],
            similarity=0.96,
        )
        injector = PrecedentInjector()
        action = injector.classify_action([p])
        assert action == PrecedentAction.FAST_FOLLOW

    def test_below_threshold_no_precedent(self):
        from src.ai_congress.core.precedent.precedent_injector import (
            PrecedentInjector, PrecedentAction,
        )
        from src.ai_congress.core.precedent.precedent_store import Precedent

        p = Precedent(
            id="p1", session_id="s1", query_text="q", ruling_text="r",
            domain="general", consensus=0.90, models_used=["m1"],
            similarity=0.60,
        )
        injector = PrecedentInjector()
        action = injector.classify_action([p])
        assert action == PrecedentAction.NO_PRECEDENT

    def test_augment_system_prompt_soft_cite(self):
        from src.ai_congress.core.precedent.precedent_injector import (
            PrecedentInjector, PrecedentAction,
        )
        from src.ai_congress.core.precedent.precedent_store import Precedent

        p = Precedent(
            id="p1", session_id="s1",
            query_text="What is gravity?",
            ruling_text="Gravity is a fundamental force.",
            domain="science", consensus=0.87, models_used=["m1"],
            similarity=0.85,
        )
        injector = PrecedentInjector()
        result = injector.augment_system_prompt(
            "You are a helpful assistant.", [p], PrecedentAction.SOFT_CITE,
        )
        assert "PRIOR CONGRESS RULING" in result
        assert "FOLLOW" in result
        assert "DISTINGUISH" in result
        assert "Gravity is a fundamental force" in result

    def test_augment_does_nothing_for_no_precedent(self):
        from src.ai_congress.core.precedent.precedent_injector import (
            PrecedentInjector, PrecedentAction,
        )
        injector = PrecedentInjector()
        base = "You are a helpful assistant."
        result = injector.augment_system_prompt(base, [], PrecedentAction.NO_PRECEDENT)
        assert result == base

    def test_detect_distinguish_true(self):
        from src.ai_congress.core.precedent.precedent_injector import PrecedentInjector
        injector = PrecedentInjector()
        text = "I DISTINGUISH from the prior ruling because the question context differs significantly."
        assert injector.detect_distinguish(text) is True

    def test_detect_distinguish_overrule(self):
        from src.ai_congress.core.precedent.precedent_injector import PrecedentInjector
        injector = PrecedentInjector()
        text = "I believe we should overrule the previous decision."
        assert injector.detect_distinguish(text) is True

    def test_detect_distinguish_false(self):
        from src.ai_congress.core.precedent.precedent_injector import PrecedentInjector
        injector = PrecedentInjector()
        text = "I agree with this assessment. The answer is indeed 42."
        assert injector.detect_distinguish(text) is False

    def test_detect_follow_is_not_distinguish(self):
        from src.ai_congress.core.precedent.precedent_injector import PrecedentInjector
        injector = PrecedentInjector()
        text = "I FOLLOW the prior ruling. The answer remains consistent."
        assert injector.detect_distinguish(text) is False

    def test_build_fast_follow_response(self):
        from src.ai_congress.core.precedent.precedent_injector import PrecedentInjector
        from src.ai_congress.core.precedent.precedent_store import Precedent

        p = Precedent(
            id="p1", session_id="s1", query_text="q", ruling_text="The answer is 4.",
            domain="math", consensus=0.95, models_used=["m1"], similarity=0.96,
        )
        injector = PrecedentInjector()
        resp = injector.build_fast_follow_response(p)
        assert resp["final_answer"] == "The answer is 4."
        assert resp["confidence"] == 0.95
        assert resp["precedent"]["action"] == "fast_follow"
        assert resp["precedent"]["disposition"] == "followed"


class TestOrchestratorIntegration:
    """Test that EnhancedOrchestrator has precedent attributes."""

    def test_orchestrator_has_precedent_store(self):
        from src.ai_congress.core.enhanced_orchestrator import EnhancedOrchestrator
        import inspect
        init_src = inspect.getsource(EnhancedOrchestrator.__init__)
        assert "precedent_store" in init_src

    def test_orchestrator_has_precedent_injector(self):
        from src.ai_congress.core.enhanced_orchestrator import EnhancedOrchestrator
        import inspect
        init_src = inspect.getsource(EnhancedOrchestrator.__init__)
        assert "precedent_injector" in init_src

    def test_build_result_has_precedent_field(self):
        from src.ai_congress.core.enhanced_orchestrator import EnhancedOrchestrator
        import inspect
        src = inspect.getsource(EnhancedOrchestrator._build_result)
        assert "precedent" in src
