"""RAG integration for swarm query augmentation."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SwarmRAGIntegrator:
    """Integrates RAG retrieval into the swarm query pipeline.

    Augments user prompts with relevant document chunks retrieved
    from the RAG engine before distributing to models.
    """

    def __init__(self, rag_engine: Any) -> None:
        """Initialize with an existing RAG engine instance.

        Args:
            rag_engine: An instance of RAGEngine with a retrieve() or search() method.
        """
        self.rag_engine = rag_engine

    async def augment_swarm_prompt(
        self,
        prompt: str,
        document_ids: Optional[list[str]] = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Retrieve relevant chunks and build an augmented prompt.

        Args:
            prompt: The original user query.
            document_ids: Optional list of document IDs to restrict search to.
            top_k: Number of top chunks to retrieve.

        Returns:
            Dict with augmented_prompt, chunks_used, and original_prompt.
        """
        chunks: list[dict] = []

        try:
            # Try common RAG engine method signatures
            if hasattr(self.rag_engine, "retrieve"):
                raw_chunks = await self.rag_engine.retrieve(prompt, top_k=top_k)
            elif hasattr(self.rag_engine, "search"):
                raw_chunks = await self.rag_engine.search(prompt, top_k=top_k)
            elif hasattr(self.rag_engine, "query"):
                raw_chunks = await self.rag_engine.query(prompt, top_k=top_k)
            else:
                logger.warning("RAG engine has no recognized retrieval method")
                raw_chunks = []

            # Normalize chunk format
            for chunk in raw_chunks:
                if isinstance(chunk, dict):
                    chunks.append({
                        "content": chunk.get("content", chunk.get("text", "")),
                        "source": chunk.get("source", chunk.get("document_id", "unknown")),
                        "similarity": chunk.get("similarity", chunk.get("score", 0.0)),
                    })
                elif isinstance(chunk, str):
                    chunks.append({
                        "content": chunk,
                        "source": "unknown",
                        "similarity": 0.0,
                    })

            # Filter by document_ids if specified
            if document_ids:
                chunks = [c for c in chunks if c["source"] in document_ids]

        except Exception as e:
            logger.error("RAG retrieval failed: %s", e)
            chunks = []

        if chunks:
            context = self.format_chunks_for_context(chunks)
            augmented_prompt = (
                f"Based on the following context:\n{context}\n\nQuestion: {prompt}"
            )
        else:
            augmented_prompt = prompt

        return {
            "augmented_prompt": augmented_prompt,
            "chunks_used": chunks,
            "original_prompt": prompt,
        }

    def format_chunks_for_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks with source attribution.

        Args:
            chunks: List of chunk dicts with content and source.

        Returns:
            Formatted context string with source annotations.
        """
        formatted_parts: list[str] = []
        for i, chunk in enumerate(chunks):
            source = chunk.get("source", "unknown")
            content = chunk.get("content", "")
            formatted_parts.append(f"[Source: {source}, chunk {i + 1}] {content}")
        return "\n\n".join(formatted_parts)
