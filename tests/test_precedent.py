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
