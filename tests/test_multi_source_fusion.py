"""Tests for multi-source RAG context fusion."""

import pytest

from src.ai_congress.core.rag.multi_source_fusion import MultiSourceFusion


class TestMultiSourceFusion:
    @pytest.mark.asyncio
    async def test_fuse_sources_orders_by_reliability_and_deduplicates(self):
        fusion = MultiSourceFusion()

        result = await fusion.fuse_sources(
            query="ignored",
            rag_chunks=[{"content": "shared context about vector search"}],
            web_results=[{"snippet": "shared context about vector search"}],
            memory_results=[{"text": "unique memory from earlier discussion"}],
        )

        assert result["total_sources"] == 2
        assert [source["source_type"] for source in result["sources"]] == [
            "rag",
            "memory",
        ]
        assert result["fused_context"] == (
            "shared context about vector search\n\n"
            "unique memory from earlier discussion"
        )

    def test_deduplicate_keeps_higher_reliability_source(self):
        fusion = MultiSourceFusion()
        items = [
            {
                "source_type": "memory",
                "content": "same answer from multiple sources",
                "reliability": 0.6,
            },
            {
                "source_type": "web",
                "content": "same answer from multiple sources",
                "reliability": 0.8,
            },
        ]

        deduped = fusion.deduplicate(items)

        assert len(deduped) == 1
        assert deduped[0]["source_type"] == "web"
