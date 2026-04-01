"""Memory-augmented agents with short-term and long-term memory stores."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict


class AgentMemory:
    """Provides short-term and long-term memory for agents.

    Short-term memory holds recent (query, response) pairs.
    Long-term memory persists frequently-accessed exchanges keyed by query hash.
    """

    def __init__(self, max_short_term: int = 20, max_long_term: int = 100) -> None:
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term

        # Short-term: list of recent exchanges
        self._short_term: list[dict] = []

        # Long-term: query_hash -> {query, response, timestamp, relevance_count}
        self._long_term: OrderedDict[str, dict] = OrderedDict()

    @staticmethod
    def _hash_query(query: str) -> str:
        """Create a stable hash for a query string."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    @staticmethod
    def _word_overlap_similarity(text_a: str, text_b: str) -> float:
        """Compute simple word-overlap similarity between two texts."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def add_exchange(self, query: str, response: str) -> None:
        """Add a (query, response) exchange to memory.

        Adds to short-term memory. If a similar query has been seen before
        (in short-term), promotes it to long-term memory.
        """
        exchange = {
            "query": query,
            "response": response,
            "timestamp": time.time(),
        }

        # Check if a similar query exists in short-term memory
        query_hash = self._hash_query(query)
        similar_found = False
        for existing in self._short_term:
            similarity = self._word_overlap_similarity(query, existing["query"])
            if similarity > 0.5:
                similar_found = True
                break

        # Add to short-term
        self._short_term.append(exchange)
        if len(self._short_term) > self.max_short_term:
            self._short_term.pop(0)

        # Promote to long-term if similar query seen before
        if similar_found or query_hash in self._long_term:
            if query_hash in self._long_term:
                self._long_term[query_hash]["relevance_count"] += 1
                self._long_term[query_hash]["response"] = response
                self._long_term[query_hash]["timestamp"] = time.time()
            else:
                self._long_term[query_hash] = {
                    "query": query,
                    "response": response,
                    "timestamp": time.time(),
                    "relevance_count": 1,
                }
            # Enforce long-term size limit (evict oldest)
            while len(self._long_term) > self.max_long_term:
                self._long_term.popitem(last=False)

    def recall_relevant(self, query: str, top_k: int = 3) -> list[dict]:
        """Find the most relevant past exchanges using word overlap similarity.

        Searches both short-term and long-term memory.

        Returns:
            List of up to top_k dicts with keys: query, response, similarity, source.
        """
        candidates = []

        # Search short-term
        for exchange in self._short_term:
            sim = self._word_overlap_similarity(query, exchange["query"])
            if sim > 0.1:
                candidates.append({
                    "query": exchange["query"],
                    "response": exchange["response"],
                    "similarity": sim,
                    "source": "short_term",
                })

        # Search long-term
        for _qhash, entry in self._long_term.items():
            sim = self._word_overlap_similarity(query, entry["query"])
            if sim > 0.1:
                candidates.append({
                    "query": entry["query"],
                    "response": entry["response"],
                    "similarity": sim,
                    "source": "long_term",
                })

        # Sort by similarity descending, return top_k
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:top_k]

    def build_memory_context(self, query: str) -> str:
        """Format relevant memories as a context string for prompting.

        Args:
            query: The current query to find relevant memories for.

        Returns:
            Formatted string with relevant past exchanges, or empty string
            if no relevant memories found.
        """
        relevant = self.recall_relevant(query)
        if not relevant:
            return ""

        parts = ["Relevant past exchanges:"]
        for i, mem in enumerate(relevant, 1):
            parts.append(
                f"\n[Memory {i}] (similarity: {mem['similarity']:.2f})\n"
                f"Q: {mem['query']}\n"
                f"A: {mem['response'][:500]}"
            )
        return "\n".join(parts)

    def clear_short_term(self) -> None:
        """Clear all short-term memory."""
        self._short_term.clear()

    def get_stats(self) -> dict:
        """Return memory statistics.

        Returns:
            Dict with short_term_count, long_term_count, and
            top entries by relevance_count.
        """
        top_entries = sorted(
            self._long_term.values(),
            key=lambda x: x["relevance_count"],
            reverse=True,
        )[:5]

        return {
            "short_term_count": len(self._short_term),
            "long_term_count": len(self._long_term),
            "max_short_term": self.max_short_term,
            "max_long_term": self.max_long_term,
            "top_long_term": [
                {"query": e["query"][:80], "relevance_count": e["relevance_count"]}
                for e in top_entries
            ],
        }
