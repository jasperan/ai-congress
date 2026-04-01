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

    def __init__(self) -> None:
        """Initialize the multi-source fusion engine."""
        pass

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
        all_items: list[dict] = []

        for chunk in (rag_chunks or []):
            all_items.append({
                "source_type": "rag",
                "content": chunk.get("content", chunk.get("text", "")),
                "reliability": self.RELIABILITY_SCORES["rag"],
                "metadata": chunk,
            })

        for result in (web_results or []):
            all_items.append({
                "source_type": "web",
                "content": result.get("content", result.get("snippet", "")),
                "reliability": self.RELIABILITY_SCORES["web"],
                "metadata": result,
            })

        for mem in (memory_results or []):
            all_items.append({
                "source_type": "memory",
                "content": mem.get("content", mem.get("text", "")),
                "reliability": self.RELIABILITY_SCORES["memory"],
                "metadata": mem,
            })

        # Deduplicate
        deduped = self.deduplicate(all_items)

        # Sort by reliability (highest first)
        deduped.sort(key=lambda x: x["reliability"], reverse=True)

        # Build fused context
        context_parts: list[str] = []
        sources: list[dict] = []
        for item in deduped:
            content = item["content"]
            if content:
                context_parts.append(content)
                sources.append({
                    "source_type": item["source_type"],
                    "content": content,
                    "reliability": item["reliability"],
                })

        fused_context = "\n\n".join(context_parts)

        logger.info(
            "Fused %d sources (from %d total, %d after dedup) for query",
            len(sources),
            len(all_items),
            len(deduped),
        )

        return {
            "fused_context": fused_context,
            "sources": sources,
            "total_sources": len(sources),
        }

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

        # Sort by reliability descending so we keep higher-reliability items
        sorted_items = sorted(items, key=lambda x: x["reliability"], reverse=True)
        kept: list[dict] = []

        for item in sorted_items:
            content = item.get("content", "")
            is_duplicate = False
            words_new = set(content.lower().split())

            for existing in kept:
                words_existing = set(existing.get("content", "").lower().split())
                union = words_new | words_existing
                if not union:
                    continue
                overlap = len(words_new & words_existing) / len(union)
                if overlap > 0.8:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(item)

        return kept
