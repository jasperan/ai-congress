"""Chunk-level attribution tracking for RAG responses."""

import logging
import time
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


class AttributionTracker:
    """Tracks which document chunks contributed to model responses.

    Enables source attribution display and coverage analysis.
    """

    def __init__(self) -> None:
        """Initialize the attribution tracker."""
        self._attributions: list[dict] = []

    def record_attribution(
        self,
        session_id: str,
        model: str,
        chunk_ids: list[str],
        response: str,
    ) -> None:
        """Record which chunks a model used in its response.

        Args:
            session_id: The session identifier.
            model: The model identifier.
            chunk_ids: List of chunk identifiers used.
            response: The model's response text.
        """
        entry = {
            "session_id": session_id,
            "model": model,
            "chunk_ids": list(chunk_ids),
            "response": response,
            "timestamp": time.time(),
        }
        self._attributions.append(entry)
        logger.debug(
            "Recorded attribution: model=%s, chunks=%d, session=%s",
            model,
            len(chunk_ids),
            session_id,
        )

    def get_session_attributions(self, session_id: str) -> list[dict]:
        """Get all attributions for a specific session.

        Args:
            session_id: The session identifier.

        Returns:
            List of attribution dicts for the session.
        """
        return [a for a in self._attributions if a["session_id"] == session_id]

    def format_attribution_display(self, attributions: list[dict]) -> str:
        """Format attributions into a human-readable source citation string.

        Args:
            attributions: List of attribution dicts.

        Returns:
            Formatted string like 'Sources used: doc1.pdf (chunks 3, 7), doc2.xlsx (chunk 2)'.
        """
        if not attributions:
            return "No sources used."

        # Group chunk IDs by source document (extract doc name from chunk ID)
        source_chunks: dict[str, list[str]] = defaultdict(list)
        for attr in attributions:
            for chunk_id in attr.get("chunk_ids", []):
                # Chunk IDs may be formatted as "doc_name:chunk_num" or just IDs
                if ":" in chunk_id:
                    doc, chunk_num = chunk_id.rsplit(":", 1)
                    source_chunks[doc].append(chunk_num)
                else:
                    source_chunks["document"].append(chunk_id)

        parts: list[str] = []
        for source, chunks in sorted(source_chunks.items()):
            chunk_list = ", ".join(sorted(set(chunks)))
            label = "chunk" if len(set(chunks)) == 1 else "chunks"
            parts.append(f"{source} ({label} {chunk_list})")

        return "Sources used: " + ", ".join(parts)

    def compute_attribution_coverage(
        self, response: str, chunks: list[dict]
    ) -> float:
        """Compute what fraction of response content traces to source chunks.

        Uses word overlap to estimate how much of the response is derived
        from the provided source chunks.

        Args:
            response: The model's response text.
            chunks: List of chunk dicts with 'content' key.

        Returns:
            Coverage fraction between 0.0 and 1.0.
        """
        if not response or not chunks:
            return 0.0

        response_words = set(response.lower().split())
        if not response_words:
            return 0.0

        chunk_words: set[str] = set()
        for chunk in chunks:
            content = chunk.get("content", chunk.get("text", ""))
            chunk_words.update(content.lower().split())

        if not chunk_words:
            return 0.0

        overlap = response_words & chunk_words
        coverage = len(overlap) / len(response_words)
        return round(min(coverage, 1.0), 3)
