"""Multi-source knowledge fusion from RAG, web search, and memory."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MultiSourceFusion:
    """Merges and deduplicates contexts from multiple knowledge sources.

    Sources are scored by reliability: RAG chunks (1.0) > web results (0.8) > memory (0.6).
    """

    # Reliability scores by source type
    RELIABILITY_SCORES: dict[str, float] = {
        "rag": 1.0,
        "web": 0.8,
        "memory": 0.6,
    }

    async def fuse_sources(
        self,
        query: str,
        rag_chunks: Optional[list[dict]] = None,
        web_results: Optional[list[dict]] = None,
        memory_results: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Merge and deduplicate contexts from multiple sources.

        Args:
            query: The original query for context.
            rag_chunks: Retrieved document chunks.
            web_results: Web search results.
            memory_results: Memory/conversation history results.

        Returns:
            Dict with fused_context string, sources list, and total_sources count.
        """
        all_items = [
            *self._source_items("rag", rag_chunks, ("content", "text")),
            *self._source_items("web", web_results, ("content", "snippet")),
            *self._source_items("memory", memory_results, ("content", "text")),
        ]

        deduped = self.deduplicate(all_items)
        deduped.sort(key=lambda x: x["reliability"], reverse=True)

        sources = [
            {
                "source_type": item["source_type"],
                "content": item["content"],
                "reliability": item["reliability"],
            }
            for item in deduped
            if item["content"]
        ]

        fused_context = "\n\n".join(source["content"] for source in sources)

        logger.info(
            "Fused %d sources (from %d total, %d after dedup) for query: %s",
            len(sources),
            len(all_items),
            len(deduped),
            query[:80],
        )

        return {
            "fused_context": fused_context,
            "sources": sources,
            "total_sources": len(sources),
        }

    @classmethod
    def _source_items(
        cls,
        source_type: str,
        records: Optional[list[dict]],
        content_keys: tuple[str, ...],
    ) -> list[dict]:
        """Normalize raw source records into scored fusion items."""
        return [
            {
                "source_type": source_type,
                "content": cls._first_value(record, content_keys),
                "reliability": cls.RELIABILITY_SCORES[source_type],
                "metadata": record,
            }
            for record in (records or [])
        ]

    @staticmethod
    def _first_value(record: dict, keys: tuple[str, ...]) -> str:
        """Return the first non-empty content value from a source record."""
        for key in keys:
            value = record.get(key)
            if value:
                return value
        return ""

    def deduplicate(self, items: list[dict]) -> list[dict]:
        """Remove near-duplicate items based on word overlap.

        Items with > 0.8 word overlap similarity are considered duplicates.
        The higher-reliability item is kept.

        Args:
            items: List of source item dicts with content and reliability.

        Returns:
            Deduplicated list of items.
        """
        if not items:
            return []

        sorted_items = sorted(items, key=lambda x: x["reliability"], reverse=True)
        kept: list[dict] = []

        for item in sorted_items:
            content = item.get("content", "")
            is_duplicate = any(
                self._word_overlap(content, existing.get("content", "")) > 0.8
                for existing in kept
            )

            if not is_duplicate:
                kept.append(item)

        return kept

    @staticmethod
    def _word_overlap(first: str, second: str) -> float:
        """Return Jaccard word overlap for two content strings."""
        first_words = set(first.lower().split())
        second_words = set(second.lower().split())
        union = first_words | second_words
        if not union:
            return 0.0
        return len(first_words & second_words) / len(union)
